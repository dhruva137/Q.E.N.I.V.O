"""World comparators, each run in its own process with a hard wall-clock limit.

Every comparator is optional: if its package is missing or it crashes, the row records why and
the campaign moves on. Where the comparator returns a primal point, QENIVO's own KKT check scores
its feasibility on the original model, so every solver is held to the same standard.

    HiGHS (University of Edinburgh, UK)       highs_simplex, highs_ipm, highs_pdlp
    OR-Tools (Google, USA)                    ortools_glop, ortools_pdlp
    SCIP (Zuse Institute Berlin, Germany)     scip
    cuPDLPx (MIT, USA; GPU)                   cupdlpx          (CLI binary via CUPDLPX_BIN)
    NVIDIA cuOpt (USA; GPU)                   cuopt            (best effort: API is version dependent)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PRE = "import json,sys,time,numpy as np\npath,tl,tol=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])\n"
_OUT = "\nprint('RESULT '+json.dumps(res))\n"

CODE = {
    "highs_simplex": _PRE + """
import highspy
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('time_limit',tl)
h.setOptionValue('solver','simplex'); h.readModel(path); t=time.perf_counter(); h.run(); dt=time.perf_counter()-t
st=h.modelStatusToString(h.getModelStatus()); x=list(h.getSolution().col_value)
res={'status':st,'objective':h.getInfo().objective_function_value if st=='Optimal' else None,'time':dt,'x':x}""" + _OUT,
    "highs_ipm": _PRE + """
import highspy
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('time_limit',tl)
h.setOptionValue('solver','ipm'); h.readModel(path); t=time.perf_counter(); h.run(); dt=time.perf_counter()-t
st=h.modelStatusToString(h.getModelStatus()); x=list(h.getSolution().col_value)
res={'status':st,'objective':h.getInfo().objective_function_value if st=='Optimal' else None,'time':dt,'x':x}""" + _OUT,
    "highs_pdlp": _PRE + """
import highspy
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('time_limit',tl)
h.setOptionValue('solver','pdlp'); h.setOptionValue('pdlp_d_gap_tol',tol); h.readModel(path)
t=time.perf_counter(); h.run(); dt=time.perf_counter()-t
st=h.modelStatusToString(h.getModelStatus()); x=list(h.getSolution().col_value)
res={'status':st,'objective':h.getInfo().objective_function_value,'time':dt,'x':x}""" + _OUT,
    "ortools_glop": _PRE + """
from ortools.linear_solver.python import model_builder as mb
m=mb.Model(); m.import_from_mps_file(path); s=mb.Solver('glop'); s.set_time_limit_in_seconds(tl)
t=time.perf_counter(); st=s.solve(m); dt=time.perf_counter()-t
ok=st in (mb.SolveStatus.OPTIMAL, mb.SolveStatus.FEASIBLE)
res={'status':str(st).split('.')[-1],'objective':s.objective_value if ok else None,'time':dt,
     'x':[float(v) for v in s.values(m.get_variables())] if ok else None}""" + _OUT,
    "ortools_pdlp": _PRE + """
from ortools.linear_solver.python import model_builder as mb
m=mb.Model(); m.import_from_mps_file(path); s=mb.Solver('pdlp'); s.set_time_limit_in_seconds(tl)
t=time.perf_counter(); st=s.solve(m); dt=time.perf_counter()-t
ok=st in (mb.SolveStatus.OPTIMAL, mb.SolveStatus.FEASIBLE)
res={'status':str(st).split('.')[-1],'objective':s.objective_value if ok else None,'time':dt,
     'x':[float(v) for v in s.values(m.get_variables())] if ok else None}""" + _OUT,
    "scip": _PRE + """
from pyscipopt import Model
m=Model(); m.hideOutput(); m.readProblem(path); m.setParam('limits/time',tl)
t=time.perf_counter(); m.optimize(); dt=time.perf_counter()-t
st=m.getStatus(); ok=m.getNSols()>0
res={'status':st,'objective':m.getObjVal() if ok else None,'time':dt,
     'x':[m.getVal(v) for v in m.getVars(transformed=False)] if ok else None,'names':[v.name for v in m.getVars(transformed=False)] if ok else None}""" + _OUT,
}


def _check(path, res):
    """Score a comparator's primal point with our own residual check (feasibility only)."""
    x = res.get("x")
    if not x:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np

    import qenivo
    from qenivo.certify.kkt import rhs_norm
    p = qenivo.read(path)
    if res.get("names"):
        idx = {n: i for i, n in enumerate(res["names"])}
        _, cn = p.names()
        if all(c in idx for c in cn):
            x = [res["x"][idx[c]] for c in cn]
    x = np.asarray(x, float)
    if x.size != p.n:
        res["our_check"] = "size mismatch"
        return
    ax = p.A @ x
    viol = np.concatenate([np.maximum(p.lc - ax, 0) + np.maximum(ax - p.uc, 0), np.maximum(p.lx - x, 0) + np.maximum(x - p.ux, 0)])
    res["our_rel_primal"] = float(np.linalg.norm(viol) / (1 + rhs_norm(p.lc, p.uc)))
    res["our_objective"] = p.objective(x)


def _py():
    """Interpreter that has the comparator packages. The notebook installs them into a separate
    virtual environment (CMP_PYTHON) so they can never disturb the solver's own packages."""
    exe = os.environ.get("CMP_PYTHON")
    return exe if exe and Path(exe).exists() else sys.executable


def run(name: str, path, tol: float = 1e-6, time_limit: float = 300.0) -> dict:
    t0 = time.perf_counter()
    if name == "cupdlpx":
        return _cupdlpx(path, tol, time_limit)
    if name == "cuopt":
        return _cuopt(path, tol, time_limit)
    code = CODE[name]
    try:
        p = subprocess.run([_py(), "-c", code, str(path), str(time_limit), str(tol)],
                           capture_output=True, text=True, timeout=time_limit * 1.5 + 60)
    except subprocess.TimeoutExpired:
        return {"solver": name, "status": "killed_at_wall", "time": time.perf_counter() - t0}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
            try:
                _check(path, res)
            except Exception as e:  # noqa: BLE001
                res["our_check"] = f"error {e}"
            res.pop("x", None); res.pop("names", None)
            res["solver"] = name
            return res
    err = (p.stderr.strip().splitlines() or ["no output"])[-1][:300]
    return {"solver": name, "status": "unavailable" if "ModuleNotFoundError" in p.stderr else "error", "error": err}


def _cupdlpx(path, tol, tl):
    exe = os.environ.get("CUPDLPX_BIN")
    if not exe or not Path(exe).exists():
        return {"solver": "cupdlpx", "status": "unavailable", "error": "CUPDLPX_BIN not set"}
    with tempfile.TemporaryDirectory() as td:
        t = time.perf_counter()
        try:
            p = subprocess.run([exe, "-q", "--eps_opt", str(tol), "--eps_feas", str(tol), "--time_limit", str(tl), str(path), td],
                               capture_output=True, text=True, timeout=tl + 120)
        except subprocess.TimeoutExpired:
            return {"solver": "cupdlpx", "status": "killed_at_wall", "time": tl + 120}
        res = {"solver": "cupdlpx", "status": "ran" if p.returncode == 0 else "error", "time": time.perf_counter() - t}
        stem = Path(path).name.split(".")[0]
        summ = Path(td) / f"{stem}_summary.txt"
        if summ.exists():
            txt = summ.read_text()
            res["summary"] = txt[:2000]
            for line in txt.splitlines():
                low = line.lower()
                if "status" in low and ":" in line:
                    res["status"] = line.split(":", 1)[1].strip()
                if ("primal objective" in low or "objective" in low) and ":" in line and "objective" not in res:
                    try:
                        res["objective"] = float(line.split(":", 1)[1].split()[0])
                    except (ValueError, IndexError):
                        pass
        xs = Path(td) / f"{stem}_primal_solution.txt"
        if xs.exists():
            import numpy as np
            res["x"] = np.loadtxt(xs).reshape(-1).tolist()
            try:
                _check(path, res)
            except Exception as e:  # noqa: BLE001
                res["our_check"] = f"error {e}"
            res.pop("x", None)
        return res


def _cuopt(path, tol, tl):
    code = _PRE + """
try:
    from cuopt import linear_programming as lp
    dm = lp.ParseMps(path) if hasattr(lp, 'ParseMps') else None
    from cuopt.linear_programming import solver as S, solver_settings as SS
    st = SS.SolverSettings(); st.set_optimality_tolerance(tol); st.set_parameter('time_limit', tl)
    t=time.perf_counter(); sol = S.Solve(dm, st); dt=time.perf_counter()-t
    res={'status':str(sol.get_termination_status()),'objective':float(sol.get_primal_objective()),'time':dt}
except Exception as e:
    res={'status':'unavailable','error':repr(e)[:300]}""" + _OUT
    try:
        p = subprocess.run([_py(), "-c", code, str(path), str(tl), str(tol)], capture_output=True, text=True,
                           timeout=tl * 1.5 + 60)
    except subprocess.TimeoutExpired:
        return {"solver": "cuopt", "status": "killed_at_wall"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:]); r["solver"] = "cuopt"; return r
    return {"solver": "cuopt", "status": "error", "error": p.stderr[-300:]}


def available() -> dict:
    """Which comparators can run here (import test in a subprocess, no side effects)."""
    mods = {"highs_simplex": "highspy", "highs_ipm": "highspy", "highs_pdlp": "highspy",
            "ortools_glop": "ortools.linear_solver.python.model_builder",
            "ortools_pdlp": "ortools.linear_solver.python.model_builder", "scip": "pyscipopt",
            "cuopt": "cuopt"}
    out = {}
    for k, m in mods.items():
        r = subprocess.run([_py(), "-c", f"import {m}"], capture_output=True)
        out[k] = r.returncode == 0
    out["cupdlpx"] = bool(os.environ.get("CUPDLPX_BIN")) and Path(os.environ.get("CUPDLPX_BIN", "")).exists()
    return out
