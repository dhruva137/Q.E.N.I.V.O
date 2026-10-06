"""Shadow mode: run the incumbent solver and QENIVO on the same model and diff the answers.

    python -m qenivo.compat.shadow plan.mps --external "my_solver_wrapper {model} {solution}"
                                   [--report diff.json] [--cert plan.cert.json] [--engine auto]

The external solver is never imported: it runs as a separate, user-configured command (also
read from the QENIVO_SHADOW_CMD environment variable). `{model}` is replaced by the model path
and `{solution}` by a JSON file the command must write; without `{solution}` the command's
standard output is read as that JSON. The file holds

    {"status": "optimal" | "infeasible" | "unbounded" | <anything else>,
     "objective": number, "x": {column: value}, "y": {row: value}}      (y optional)

with row duals in the user's objective sense, so that c - A'y are the reduced costs (the
convention of qenivo.Solution.y). A short wrapper turns any solver's output into this file.

QENIVO's answer carries its certificate (written to --cert). The external answer is checked on
the same model by QENIVO's own residual code: its claimed objective against its x, primal
feasibility (and integrality for a MILP), and when duals are given, dual feasibility and the
duality gap. Then verdicts, objectives, primal and dual values are compared. Different primal
or dual vectors with equal objectives and both answers checked are reported as alternative
optima, not as mismatches. Exit code 0 when the answers agree, 1 on a mismatch, 2 on a usage
error, 3 when the external command fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "qenivo.shadow/1"
_PROVEN = ("optimal", "infeasible", "unbounded")


@dataclass
class ShadowReport:
    model: str
    qenivo: dict
    external: dict
    checks: list = field(default_factory=list)     # (name, ok, detail)
    notes: list = field(default_factory=list)
    cert_path: str | None = None

    @property
    def agree(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def mismatches(self) -> list:
        return [f"{n}: {d}" for n, ok, d in self.checks if not ok]

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "model": self.model, "agree": self.agree, "checks": [
            {"check": n, "ok": bool(ok), "detail": d} for n, ok, d in self.checks], "notes": self.notes,
            "qenivo": self.qenivo, "external": self.external, "qenivo_certificate": self.cert_path}

    def text(self) -> str:
        lines = [f"Shadow run on {self.model}: " + ("AGREE" if self.agree else "MISMATCH")]
        lines.append(f"  QENIVO   {self.qenivo.get('verdict')}  objective {self.qenivo.get('objective')}"
                     f"  engine {self.qenivo.get('engine')}  {self.qenivo.get('time', 0):.3f}s")
        lines.append(f"  external {self.external.get('status')}  objective {self.external.get('objective')}"
                     f"  {self.external.get('time', 0):.3f}s")
        for n, ok, d in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {n}: {d}")
        lines += [f"  note: {n}" for n in self.notes]
        if self.cert_path:
            lines.append(f"  QENIVO certificate: {self.cert_path}")
        return "\n".join(lines)


def _split(command: str) -> list[str]:
    words = shlex.split(command, posix=os.name != "nt")
    return [w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]


def run_external(command, model: str, timeout: float = 600.0) -> dict:
    """Run the external command and return its parsed answer plus wall time. Raises RuntimeError."""
    argv = _split(command) if isinstance(command, str) else [str(c) for c in command]
    if not argv:
        raise RuntimeError("empty external command")
    with tempfile.TemporaryDirectory(prefix="qenivo_shadow_") as tmp:
        sol_path = str(Path(tmp) / "external.json")
        use_file = any("{solution}" in a for a in argv)
        argv = [a.replace("{model}", str(model)).replace("{solution}", sol_path) for a in argv]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise RuntimeError(f"external command not found: {argv[0]}") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"external command timed out after {timeout:g}s") from None
        wall = time.perf_counter() - t0
        if p.returncode != 0:
            raise RuntimeError(f"external command exited with {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
        try:
            raw = Path(sol_path).read_text(encoding="utf-8") if use_file else p.stdout
            ans = json.loads(raw)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"external answer is not readable JSON: {e}") from None
    if not isinstance(ans, dict) or "status" not in ans:
        raise RuntimeError("external answer must be a JSON object with a 'status'")
    ans["status"] = str(ans["status"]).lower()
    ans["time"] = wall
    ans["command"] = argv
    return ans


def _vector(named, names, what) -> tuple[np.ndarray | None, list]:
    if named is None:
        return None, []
    if isinstance(named, list):
        v = np.asarray(named, dtype=float)
        return (v, []) if v.size == len(names) else (None, [f"{what} has {v.size} values, model has {len(names)}"])
    missing = [nm for nm in names if nm not in named]
    v = np.array([float(named.get(nm, np.nan) if named.get(nm) is not None else np.nan) for nm in names])
    return v, missing


def check_external(prob, ans: dict, tol: float = 1e-6) -> dict:
    """QENIVO's residuals for the external answer on the model (no trust in its own claims)."""
    from ..certify.kkt import kkt_residuals
    rn, cn = prob.names()
    out: dict = {"problems": []}
    x, miss_x = _vector(ans.get("x"), cn, "x")
    y, miss_y = _vector(ans.get("y"), rn, "y")
    if miss_x:
        out["problems"].append(f"{len(miss_x)} columns missing from external x (first {miss_x[0]})")
    if miss_y:
        out["problems"].append(f"{len(miss_y)} rows missing from external y (first {miss_y[0]})")
    if x is None or not np.all(np.isfinite(x)):
        return out
    ax = prob.A @ x
    scale = 1.0 + max([float(np.max(np.abs(v[np.isfinite(v)]), initial=0)) for v in (prob.lc, prob.uc, prob.lx, prob.ux)])
    viol = max(float(np.max(np.maximum(prob.lc - ax, 0), initial=0)), float(np.max(np.maximum(ax - prob.uc, 0), initial=0)),
               float(np.max(np.maximum(prob.lx - x, 0), initial=0)), float(np.max(np.maximum(x - prob.ux, 0), initial=0)))
    out.update(x=x, objective_of_x=prob.objective(x), rel_primal_violation=viol / scale)
    if prob.is_mip:
        out["max_integrality"] = float(np.max(np.abs(x[prob.integer] - np.round(x[prob.integer])), initial=0))
    if y is not None and np.all(np.isfinite(y)) and not prob.is_mip:
        k = kkt_residuals(prob, x, prob.obj_sign * y)
        out.update(y=y, rel_dual=float(k["rel_dual"]), rel_gap=float(k["rel_gap"]))
    return out


def _close(a, b, rtol) -> bool:
    return abs(a - b) <= rtol * (1.0 + max(abs(a), abs(b)))


def compare(prob, sol, ans: dict, chk: dict, obj_tol: float = 1e-6, feas_tol: float = 1e-6) -> tuple[list, list]:
    checks, notes = [], []
    ext_status = ans["status"]
    checks.append(("QENIVO verdict is proven", sol.is_proven,
                   f"verdict {sol.verdict}" + ("" if sol.is_proven else f" ({sol.engine.get('note') or sol.status})")))
    ext_kind = ext_status if ext_status in _PROVEN else "other"
    checks.append(("verdicts agree", sol.verdict == ext_kind, f"QENIVO {sol.verdict}, external {ext_status}"))
    for p in chk["problems"]:
        checks.append(("external answer is complete", False, p))
    if sol.verdict != "optimal" or ext_kind != "optimal":
        return checks, notes
    eo = ans.get("objective")
    if eo is None:
        checks.append(("external objective given", False, "no 'objective' in the external answer"))
    else:
        eo = float(eo)
        checks.append(("objectives agree", _close(sol.objective, eo, obj_tol),
                       f"QENIVO {sol.objective:.12g}, external {eo:.12g}, |diff| {abs(sol.objective - eo):.3e} "
                       f"(tolerance {obj_tol:g} relative)"))
        if "objective_of_x" in chk:
            checks.append(("external objective matches its x", _close(chk["objective_of_x"], eo, obj_tol),
                           f"claimed {eo:.12g}, c'x on the model {chk['objective_of_x']:.12g}"))
    if "x" not in chk:
        checks.append(("external primal values given", False, "no finite x in the external answer"))
        return checks, notes
    checks.append(("external x is feasible on the model", chk["rel_primal_violation"] <= feas_tol,
                   f"relative violation {chk['rel_primal_violation']:.3e}"))
    if "max_integrality" in chk:
        checks.append(("external x is integral", chk["max_integrality"] <= 1e-6,
                       f"max distance to an integer {chk['max_integrality']:.3e}"))
    dx = float(np.max(np.abs(sol.x - chk["x"]), initial=0)) if sol.x is not None else float("nan")
    xscale = 1.0 + float(np.max(np.abs(sol.x), initial=0)) if sol.x is not None else 1.0
    if dx <= 1e-6 * xscale:
        checks.append(("primal values agree", True, f"max |dx| {dx:.3e}"))
    else:
        notes.append(f"primal vectors differ (max |dx| {dx:.3e}) with equal checked objectives: alternative optimum")
    if "y" in chk:
        checks.append(("external duals pass QENIVO's dual check", max(chk["rel_dual"], chk["rel_gap"]) <= feas_tol * 10,
                       f"rel_dual {chk['rel_dual']:.3e}, rel_gap {chk['rel_gap']:.3e}"))
        if sol.y is not None:
            dy = float(np.max(np.abs(sol.y - chk["y"]), initial=0))
            yscale = 1.0 + float(np.max(np.abs(sol.y), initial=0))
            if dy <= 1e-6 * yscale:
                checks.append(("dual values agree", True, f"max |dy| {dy:.3e}"))
            else:
                notes.append(f"dual vectors differ (max |dy| {dy:.3e}); both checked: degenerate primal, "
                             "several dual optima")
    elif not prob.is_mip:
        notes.append("external answer has no duals; dual comparison skipped")
    return checks, notes


def shadow(model: str, command, *, engine: str = "auto", tol: float = 1e-6, time_limit: float = 600.0,
           timeout: float = 600.0, obj_tol: float = 1e-6, cert_path: str | None = None) -> ShadowReport:
    """Solve `model` with QENIVO and with the external command, then compare (raises RuntimeError
    when the external command fails)."""
    from ..api import read, solve
    prob = read(model)
    t0 = time.perf_counter()
    sol = solve(prob, engine=engine, tol=tol, time_limit=time_limit)
    qt = time.perf_counter() - t0
    sol.source = str(model)
    cert_path = cert_path or str(Path(model).name.split(".")[0] + ".cert.json")
    sol.save_certificate(cert_path)
    ans = run_external(command, model, timeout=timeout)
    chk = check_external(prob, ans, tol)
    checks, notes = compare(prob, sol, ans, chk, obj_tol=obj_tol, feas_tol=max(tol, 1e-9))
    ext = {k: ans.get(k) for k in ("status", "objective", "time", "command")}
    ext["check"] = {k: v for k, v in chk.items() if k not in ("x", "y")}
    q = {"verdict": sol.verdict, "objective": sol.objective, "engine": sol.engine.get("engine"), "time": qt,
         "residuals": {k: v for k, v in sol.residuals.items() if k.startswith("rel_") or k == "max_rel"}}
    return ShadowReport(model=str(model), qenivo=q, external=ext, checks=checks, notes=notes, cert_path=cert_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qenivo-shadow", description="diff an external solver against QENIVO")
    ap.add_argument("model")
    ap.add_argument("--external", default=os.environ.get("QENIVO_SHADOW_CMD"),
                    help="command template with {model} and optionally {solution} (default: $QENIVO_SHADOW_CMD)")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--obj-tol", type=float, default=1e-6)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds allowed for the external command")
    ap.add_argument("--cert", help="QENIVO certificate path (default <model>.cert.json here)")
    ap.add_argument("--report", help="write the comparison as JSON")
    a = ap.parse_args(argv)
    if not a.external:
        print("error: no external command (--external or QENIVO_SHADOW_CMD)", file=sys.stderr)
        return 2
    if not Path(a.model).exists():
        print(f"error: no such file {a.model}", file=sys.stderr)
        return 2
    try:
        rep = shadow(a.model, a.external, engine=a.engine, tol=a.tol, time_limit=a.time_limit,
                     timeout=a.timeout, obj_tol=a.obj_tol, cert_path=a.cert)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(rep.text())
    if a.report:
        Path(a.report).write_text(json.dumps(rep.to_json(), indent=1, default=str) + "\n", encoding="utf-8")
    return 0 if rep.agree else 1


if __name__ == "__main__":
    sys.exit(main())
