"""qenivo command line.

    qenivo solve     model.mps [--engine auto|simplex|ipm|pdhg|pdqp|milp] [--tol 1e-6]
                               [--backend auto|numpy|cupy] [--cert out.json] [--sol out.sol] [--json]
    qenivo verify    model.mps cert.json [--exact]
    qenivo explain   model.mps                      why is this plan infeasible?
    qenivo range     model.mps [--top 20]           exact cost and RHS ranging at the optimum
    qenivo crude-value model.mps --cargo COL --price-range -20%:+20% --out report.xlsx|.json|.md
    qenivo cases     model.mps cases.json [--out report.json]
    qenivo recursion haverly1|haverly2|haverly3 [--method pdhg_warm|pdhg_cold|ipm|simplex|simplex_step]
    qenivo demo                                      a two-minute tour on built-in refinery models
    qenivo serve [--port 8765]                       local planner console (browser) and JSON API
    qenivo model     [name] [-p k=v] [-o out.mps]    sector model library
    qenivo info                                      version, backends, GPU, provenance guard
    qenivo bench     <set> [--engine] [--time-limit] [--out]
    qenivo convert   in.mps|in.json out.mps|out.json|out.lp
    qenivo presolve  in.mps out.mps [--report]
    qenivo doctor                                    environment diagnostics with fix lines
    qenivo engines                                   list built-in and plugin engines
    qenivo completion bash|zsh|powershell

Global: --json for machine-readable output on every command.

Exit codes: 0 proven verdict (optimal / infeasible / unbounded), 1 not proven,
2 usage error, 3 read error, 4 verification rejected.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path


def _quiet_cupy_path_warning() -> None:
    """CuPy warns when no CUDA toolkit root is set. The pip wheels still load, and gpu_available()
    is the real check. Hide only this line so `qenivo info` stays readable on Windows."""
    warnings.filterwarnings(
        "ignore",
        message=r"CUDA path could not be detected\..*",
        category=UserWarning,
        module=r"cupy\._environment",
    )


def _emit(obj, *, as_json: bool, text: str | None = None) -> None:
    if as_json:
        print(json.dumps(obj, indent=1, default=str))
    elif text is not None:
        print(text)
    else:
        print(json.dumps(obj, indent=1, default=str))


def _load(path):
    from .api import read
    try:
        return read(path)
    except FileNotFoundError:
        print(f"error: no such file {path}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(3)


def cmd_solve(a):
    from .api import solve
    prob = _load(a.model)
    t0 = time.perf_counter()
    if a.engine == "race":
        from .certify.certificate import save
        from .workload.parallel import race
        r = race(prob, tol=a.tol, time_limit=a.time_limit)
        w = r["winner"]
        if w is None:
            payload = {"verdict": "not_proven", "race": r}
            if a.json:
                _emit(payload, as_json=True)
            else:
                print(f"race: no engine proved a verdict in {r['wall']:.1f}s")
            return 1
        if a.json:
            c = w.get("certificate") or {}
            _emit({k: c.get(k) for k in ("model", "verdict", "objective", "residuals", "engine")}
                  if c else {"verdict": w.get("verdict"), "engine": w.get("engine"), "wall": r["wall"]},
                  as_json=True)
        else:
            print(w["summary"])
            print(f"  race: {w['engine']} won in {r['wall']:.3f}s (" + ", ".join(
                f"{f['engine_requested']} {f['verdict']} {f['time']:.2f}s"
                for f in r["finished"] if "time" in f) + ")")
        if a.cert:
            save(w["certificate"], a.cert)
        return 0
    sol = solve(prob, engine=a.engine, tol=a.tol, backend=a.backend, time_limit=a.time_limit,
                verbose=a.verbose)
    sol.source = a.model
    if a.cert:
        sol.save_certificate(a.cert)
    if a.sol and sol.x is not None:
        from .io.mps import write_sol
        write_sol(a.sol, prob, sol.x, None if sol.y is None else prob.obj_sign * sol.y,
                  status=sol.verdict, kkt={k: v for k, v in sol.residuals.items() if k.startswith("rel_")})
    if a.json:
        c = sol.certificate(include_solution=False)
        print(json.dumps({k: c[k] for k in ("model", "verdict", "objective", "residuals", "engine")}, indent=1,
                         default=str))
    else:
        print(sol.summary())
        print(f"  wall {time.perf_counter() - t0:.3f}s" + (f"   certificate -> {a.cert}" if a.cert else ""))
    return 0 if sol.is_proven else 1


def cmd_verify(a):
    from .certify.certificate import load
    from .certify.verify import format_report, verify
    rep = verify(a.model, load(a.cert), exact=a.exact, tol=a.tol)
    if a.json:
        _emit(rep, as_json=True)
    else:
        print(format_report(rep))
    return 0 if rep["passed"] else 4


def cmd_explain(a):
    from .workload.explain import explain_infeasibility, format_explanation
    rep = explain_infeasibility(_load(a.model), top=a.top)
    if a.json:
        _emit(rep, as_json=True)
    else:
        print(format_explanation(rep))
    return 0 if rep.get("feasible") is not None else 1


def cmd_crude_value(a):
    """Exact break-even / valuation curve along a cargo price or capacity ray."""
    try:
        from .workload.crude_value import crude_value, parse_price_range, run_tiny_demo
    except ImportError:
        raise SystemExit("crude-value ships in the full edition of QENIVO (see docs/EDITIONS.md)")

    if getattr(a, "demo", False):
        payload = run_tiny_demo(level=int(a.demo_level), out=a.out)
        if a.json:
            _emit(payload, as_json=True)
        else:
            if payload.get("ran"):
                print(payload.get("summary") or payload)
                if a.out:
                    print(f"  report -> {a.out}")
            else:
                print(f"demo skipped: {payload.get('reason')}")
        return 0 if (not payload.get("ran") or payload.get("cert_valid") is not False) else 1

    modes = [bool(a.cargo), bool(a.column_price_ray), bool(a.rhs_ray)]
    if sum(modes) != 1:
        print("error: choose exactly one of --cargo / --column-price-ray / --rhs-ray", file=sys.stderr)
        return 2
    if not a.model:
        print("error: model path required (or pass --demo)", file=sys.stderr)
        return 2
    if not a.price_range:
        print("error: --price-range LO:HI is required (e.g. -20%:+20% or -5:5)", file=sys.stderr)
        return 2
    try:
        t_range, relative = parse_price_range(a.price_range)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # --cargo implies relative when the range used %; absolute ranges stay absolute.
    # --column-price-ray always uses the parsed relative flag from the range string.
    if a.cargo and not relative and ("%" not in a.price_range):
        # cargo without %: still treat as absolute cost shift (same as column-price-ray)
        pass

    prob = _load(a.model)
    try:
        if a.rhs_ray:
            result = crude_value(prob, None, t_range, rhs_row=a.rhs_ray, rhs_side=a.rhs_side,
                                tol=a.tol, unit=a.unit)
        else:
            col = a.cargo or a.column_price_ray
            # cargo + percent range => relative fractional move on current cost
            use_rel = relative if a.cargo else relative
            result = crude_value(prob, col, t_range, relative=use_rel, tol=a.tol, unit=a.unit)
    except Exception as exc:  # noqa: BLE001
        print(f"error: crude-value failed: {exc}", file=sys.stderr)
        return 1

    payload = result.to_json()
    if a.out:
        try:
            result.write(a.out)
        except Exception as exc:  # noqa: BLE001
            print(f"error: cannot write {a.out}: {exc}", file=sys.stderr)
            return 3
    if a.json:
        _emit(payload, as_json=True)
    else:
        print(result.explain())
        print(f"  segments {len(result.segments)}  breakpoints {len(result.breakpoints)}"
              f"  pivots {result.curve.total_pivots}"
              f"  cert {'OK' if result.valid else ('N/A' if result.valid is None else 'FAIL')}")
        if a.out:
            print(f"  report -> {a.out}")
    if result.kind == "rhs":
        return 0 if result.segments else 1
    return 0 if result.valid else 1


def cmd_range(a):
    from .api import solve
    from .workload.ranging import cost_ranging, rhs_ranging
    prob = _load(a.model)
    sol = solve(prob, engine="simplex", tol=1e-9)
    if sol.verdict != "optimal":
        msg = f"model is {sol.verdict}; ranging needs an optimal basis"
        if a.json:
            _emit({"verdict": sol.verdict, "error": msg}, as_json=True)
        else:
            print(msg)
        return 1
    cr = sorted(cost_ranging(prob, sol), key=lambda d: -abs(d["value"]))[: a.top]
    rr = [d for d in rhs_ranging(prob, sol) if d["binding"]][: a.top]
    if a.json:
        _emit({"verdict": sol.verdict, "objective": sol.objective, "cost_ranging": cr, "binding_rows": rr},
              as_json=True)
        return 0
    print(sol.summary())
    print("\nCOST RANGING (basis stays optimal while the objective coefficient is in [min, max])")
    print(f"  {'column':24s} {'value':>14s} {'coef':>12s} {'min':>12s} {'max':>12s}  status")
    for d in cr:
        print(f"  {d['column']:24s} {d['value']:14.6g} {d['cost']:12.6g} {d['cost_min']:12.6g} {d['cost_max']:12.6g}  {d['status']}")
    print("\nBINDING ROWS (marginal value holds while the limit is in [min, max])")
    print(f"  {'row':24s} {'marginal':>14s} {'limit':>12s} {'min':>12s} {'max':>12s}")
    for d in rr:
        print(f"  {d['row']:24s} {d['marginal']:14.6g} {d['limit']:12.6g} {d['limit_min']:12.6g} {d['limit_max']:12.6g}")
    return 0


def cmd_cases(a):
    from .workload.cases import Case, solve_cases
    prob = _load(a.model)
    if a.cases.lower().endswith((".csv", ".xlsx")):
        from .io.sheets import read_case_table
        cases = read_case_table(a.cases)
    else:
        spec = json.load(open(a.cases, encoding="utf-8"))
        cases = [Case(name=c["name"], cost=c.get("cost", {}), col_lo=c.get("col_lo", {}),
                      col_hi=c.get("col_hi", {}), row_lo=c.get("row_lo", {}), row_hi=c.get("row_hi", {}))
                 for c in spec["cases"]]
    rep = solve_cases(prob, cases, tol=a.tol, backend=a.backend, engine=getattr(a, "engine", "auto"))
    payload = {"base": rep.base, "engine": rep.engine, "wall_time": rep.wall_time, "table": rep.table(),
               "certificates": [s.certificate(include_solution=False) for s in rep.cases]}
    if a.out and a.out.lower().endswith((".csv", ".xlsx")):
        from .io.sheets import write_case_report
        write_case_report(a.out, rep)
    elif a.out:
        json.dump(payload, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    if a.json:
        _emit(payload, as_json=True)
    else:
        print(rep.summary())
    return 0 if all(s.is_proven for s in rep.cases) else 1


def cmd_recursion(a):
    from .workload.recursion import HAVERLY_OPTIMA, haverly, run_recursion
    inst = int(a.instance[-1])
    prob, pools = haverly(inst)
    methods = [a.method] if a.method != "all" else ["pdhg_warm", "pdhg_cold", "ipm", "simplex", "simplex_step"]
    rows = []
    if not a.json:
        print(f"Haverly {inst}: published global optimum {HAVERLY_OPTIMA[inst]:g}; start pool sulphur {a.q0}")
    for mth in methods:
        r = run_recursion(prob, pools, method=mth, q0=a.q0, verbose=a.verbose)
        rows.append({"method": mth, "summary": r.summary(), "result": getattr(r, "__dict__", str(r))})
        if not a.json:
            print("  " + r.summary())
    if a.json:
        _emit({"instance": inst, "published_optimum": HAVERLY_OPTIMA[inst], "q0": a.q0, "runs": rows}, as_json=True)
    return 0


def cmd_info(a):
    from . import __version__
    from .certify.certificate import environment
    from .kernels.backend import gpu_available
    from .provenance import guard
    env = environment()
    g = guard()
    payload = {"qenivo": __version__, **env, "gpu_available": gpu_available(),
               "provenance_guard": "CLEAN" if g["clean"] else "VIOLATION",
               "forbidden_packages": g["forbidden_packages"] or [],
               "tripwire_hits": len(g["tripwire_hits"])}
    if a.json:
        _emit(payload, as_json=True)
        return 0
    print(f"qenivo {__version__}")
    for k, v in env.items():
        print(f"  {k:18s} {v}")
    print(f"  {'gpu_available':18s} {gpu_available()}")
    print(f"  {'provenance_guard':18s} {'CLEAN' if g['clean'] else 'VIOLATION'}  packages {g['forbidden_packages'] or 'none'}"
          f"  tripwire calls {len(g['tripwire_hits'])}")
    return 0


def cmd_demo(a):
    from .demo import run_demo
    return run_demo()


def cmd_model(a):
    """Build a sector model from the library and write it as MPS (or list the library)."""
    from .io.mps import write_mps
    from .models.industry import LIBRARY
    if not a.name:
        items = {k: ((fn.__doc__ or "").strip().splitlines() or [""])[0] for k, fn in LIBRARY.items()}
        if a.json:
            _emit({"models": items}, as_json=True)
        else:
            for k, doc in items.items():
                print(f"{k:22s} {doc}")
        return 0
    kw = dict(kv.split("=", 1) for kv in a.param)
    p = LIBRARY[a.name](**{k: int(v) for k, v in kw.items()})
    out = a.out or f"{a.name}.mps"
    write_mps(p, out, name=a.name)
    kind = "QP" if p.Q is not None else "MILP" if p.integer is not None else "LP"
    payload = {"out": out, "kind": kind, "rows": p.m, "cols": p.n, "nnz": p.nnz}
    if a.json:
        _emit(payload, as_json=True)
    else:
        print(f"wrote {out}: {kind}, {p.m} rows, {p.n} columns, {p.nnz} nonzeros")
    return 0


def cmd_serve(a):
    from .security import AccessPolicy
    from .server import serve
    pol = AccessPolicy.from_args(a.token_file, a.proxy_user_header, a.trusted_proxy, a.audit_dir, a.retention_days)
    serve(a.host, a.port, pol)
    return 0


def cmd_bench(a):
    from .benchmarks import available_sets, run_benchmark
    known = available_sets()
    if a.set not in known:
        msg = f"unknown set {a.set!r}; known: {', '.join(known) or '(none — bench/ missing)'}"
        if a.json:
            _emit({"error": msg, "sets": known}, as_json=True)
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 2
    try:
        result = run_benchmark(
            a.set, engine=a.engine, time_limit=a.time_limit, out=a.out,
            backend=getattr(a, "backend", "numpy"), no_ref=not a.with_ref, jobs=a.jobs,
        )
    except Exception as e:  # noqa: BLE001
        if a.json:
            _emit({"error": str(e)}, as_json=True)
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1
    if a.json:
        _emit(result, as_json=True)
    else:
        if result.get("summary"):
            print(result["summary"])
        print(f"results -> {result['out']}")
    return 0


def cmd_convert(a):
    src, dst = Path(a.source), Path(a.dest)
    in_kind = _format_kind(src)
    out_kind = _format_kind(dst)

    if out_kind == "lp":
        msg = ("LP format writer is not available yet (owned by the I/O task). "
               "Convert to .mps or .json, or wait for qenivo.io.lpformat.")
        if a.json:
            _emit({"error": msg, "exit": 2}, as_json=True)
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 2

    try:
        prob = _read_any(src, in_kind)
    except FileNotFoundError:
        print(f"error: no such file {src}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot read {src}: {e}", file=sys.stderr)
        return 3

    try:
        _write_any(prob, dst, out_kind)
    except Exception as e:  # noqa: BLE001
        print(f"error: cannot write {dst}: {e}", file=sys.stderr)
        return 3

    payload = {"source": str(src), "dest": str(dst), "rows": prob.m, "cols": prob.n, "nnz": prob.nnz}
    if a.json:
        _emit(payload, as_json=True)
    else:
        print(f"wrote {dst}: {prob.m} rows, {prob.n} columns, {prob.nnz} nonzeros")
    return 0


def _format_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".json") or name.endswith(".json.gz"):
        return "json"
    if name.endswith(".lp") or name.endswith(".lp.gz"):
        return "lp"
    if ".mps" in name or name.endswith(".mps.gz") or name.endswith(".mps.bz2"):
        return "mps"
    # fall back to suffix
    suf = path.suffix.lower()
    if suf == ".json":
        return "json"
    if suf == ".lp":
        return "lp"
    return "mps"


def _read_any(path: Path, kind: str):
    if kind == "json":
        from .benchmarks.json_format import read_json
        return read_json(path)
    if kind == "lp":
        try:
            from .io import lpformat  # type: ignore
        except ImportError:
            raise RuntimeError("LP reader not installed (qenivo.io.lpformat absent)") from None
        return lpformat.read_lp(path)
    from .api import read
    return read(path)


def _write_any(prob, path: Path, kind: str) -> None:
    if kind == "json":
        from .benchmarks.json_format import write_json
        write_json(prob, path)
        return
    if kind == "lp":
        raise RuntimeError("LP writer not available")
    from .io.mps import write_mps
    write_mps(prob, path, name=prob.name or path.stem)


def cmd_presolve(a):
    from .engines.presolve import presolve
    prob = _load(a.model)
    red = presolve(prob)
    report = {
        "status": red.status,
        "log": red.log,
        "original": {"rows": prob.m, "cols": prob.n, "nnz": prob.nnz},
        "reduced": {"rows": red.prob.m, "cols": red.prob.n, "nnz": red.prob.nnz},
        "n_fixed": len(red.fixed),
        "out_written": False,
        "out": a.out,
        "note": None,
    }
    written = False
    if a.out and red.status == "reduced":
        try:
            from .io.mps import write_mps
            write_mps(red.prob, a.out, name=(prob.name or "presolved") + "_presolved")
            written = True
            report["out_written"] = True
        except Exception as e:  # noqa: BLE001
            report["note"] = f"presolved model not written ({e}); report only"
    elif a.out and red.status != "reduced":
        report["note"] = f"model status is {red.status!r}; MPS not written"
    else:
        report["note"] = "no --out path; report only" if not a.out else None

    if a.json or a.report or not written:
        if a.json:
            _emit(report, as_json=True)
        else:
            print(json.dumps(report, indent=1, default=str))
            if report.get("note"):
                print(report["note"])
            if not written and a.out:
                print(f"presolved MPS was not written to {a.out}")
    else:
        print(f"wrote {a.out}: {red.prob.m} rows, {red.prob.n} cols (from {prob.m}x{prob.n})")
        if a.report:
            print(json.dumps(report, indent=1, default=str))
    return 0 if red.status != "infeasible" else 1


def cmd_doctor(a):
    checks = []

    def add(name, ok, detail, fix):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix if not ok else None})

    # Python
    py = sys.version_info
    add("python", py >= (3, 10), f"{py.major}.{py.minor}.{py.micro}",
        "Install Python 3.10 or newer and recreate the virtualenv.")

    # NumPy
    try:
        import numpy as np
        add("numpy", True, np.__version__, None)
    except Exception as e:  # noqa: BLE001
        add("numpy", False, str(e), "pip install 'numpy>=1.24'")

    # SciPy
    try:
        import scipy
        add("scipy", True, scipy.__version__, None)
    except Exception as e:  # noqa: BLE001
        add("scipy", False, str(e), "pip install 'scipy>=1.10'")

    # CuPy (optional)
    try:
        import cupy
        add("cupy", True, cupy.__version__, None)
    except Exception as e:  # noqa: BLE001
        add("cupy", False, f"not importable ({type(e).__name__}: {e})"[:200],
            "Optional: pip install 'qenivo[gpu]' (CUDA 12) if you need GPU engines.")

    # GPU probe
    try:
        from .kernels.backend import gpu_info, gpu_status
        status = gpu_status()
        info = gpu_info()
        ok = status == "ok"
        detail = status if not info else f"{status}; {info}"
        add("gpu_probe", ok, detail,
            "Install a CUDA driver + CuPy, or use --backend numpy. See kernels.backend.gpu_status().")
    except Exception as e:  # noqa: BLE001
        add("gpu_probe", False, str(e), "Use CPU backend (numpy); GPU is optional.")

    # Native simplex
    try:
        from .engines.native_simplex import status as native_status
        st = native_status()
        ok = st in ("ok", "ready", "loaded") or (isinstance(st, str) and "ok" in st.lower() and "fail" not in st.lower())
        # status() returns a reason string; treat anything that looks like compile failure as not ok
        low = str(st).lower()
        ok = "fail" not in low and "error" not in low and "not found" not in low and "missing" not in low
        add("native_simplex", ok, st,
            "Install a C++ compiler (MSVC Build Tools on Windows, or g++/clang) so "
            "engines.native_simplex can compile native/simplex_core.cpp on first use.")
    except Exception as e:  # noqa: BLE001
        add("native_simplex", False, str(e),
            "Install a C++ compiler and retry; native simplex compiles on first use.")

    # Compiler
    compilers = []
    for name in ("cl", "g++", "clang++", "clang", "c++", "gcc"):
        path = shutil.which(name)
        if path:
            compilers.append(f"{name}={path}")
    add("compiler", bool(compilers), ", ".join(compilers) or "none found on PATH",
        "Install Visual Studio Build Tools (cl) or a GCC/Clang toolchain and ensure it is on PATH.")

    # Provenance
    try:
        from .provenance import guard, install_tripwires
        install_tripwires()
        g = guard()
        add("provenance_guard", g["clean"],
            f"packages={g['forbidden_packages'] or []}; tripwire_hits={len(g['tripwire_hits'])}",
            "Uninstall forbidden solver packages (HiGHS/SCIP/OR-Tools/…) from this environment; "
            "they must not appear on the solve path.")
    except Exception as e:  # noqa: BLE001
        add("provenance_guard", False, str(e), "Reinstall qenivo in a clean virtualenv.")

    failed = [c for c in checks if not c["ok"]]
    # cupy/gpu are soft failures for doctor exit code — only hard deps fail the process
    hard = {"python", "numpy", "scipy", "provenance_guard"}
    hard_failed = [c for c in failed if c["name"] in hard]
    payload = {"checks": checks, "ok": not hard_failed, "n_failed": len(failed)}

    if a.json:
        _emit(payload, as_json=True)
    else:
        print("qenivo doctor")
        for c in checks:
            mark = "OK  " if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
            if c["fix"]:
                print(f"         fix: {c['fix']}")
        print("PASS" if not hard_failed else "FAIL")
    return 0 if not hard_failed else 1


def cmd_engines(a):
    from .engines.registry import engines
    # Also surface built-ins dispatched by api.solve (not only registry plugins).
    builtin = [
        {"name": "auto", "classes": ("LP", "QP", "MILP"), "description": "router chooses engine", "source": "builtin"},
        {"name": "simplex", "classes": ("LP",), "description": "bounded revised primal/dual simplex", "source": "builtin"},
        {"name": "ipm", "classes": ("LP",), "description": "Mehrotra interior-point", "source": "builtin"},
        {"name": "pdhg", "classes": ("LP",), "description": "batched restarted-Halpern PDHG", "source": "builtin"},
        {"name": "pdhg-rf", "classes": ("LP",), "description": "float32 PDHG + float64 refinement", "source": "builtin"},
        {"name": "pdqp", "classes": ("QP",), "description": "first-order convex QP", "source": "builtin"},
        {"name": "milp", "classes": ("MILP",), "description": "branch and cut", "source": "builtin"},
        {"name": "race", "classes": ("LP",), "description": "verifier-gated multi-engine race", "source": "builtin"},
    ]
    registered = []
    for name, spec in sorted(engines().items()):
        registered.append({
            "name": name,
            "classes": list(spec.classes),
            "description": spec.description or "",
            "source": "plugin",
        })
    # Deduplicate names: plugins override listing source
    by_name = {e["name"]: e for e in builtin}
    for e in registered:
        by_name[e["name"]] = e
    rows = list(by_name.values())
    if a.json:
        _emit({"engines": rows}, as_json=True)
    else:
        print(f"{'name':12s} {'classes':16s} {'source':8s} description")
        for e in rows:
            print(f"{e['name']:12s} {','.join(e['classes']):16s} {e['source']:8s} {e['description']}")
    return 0


def cmd_completion(a):
    shell = a.shell
    cmds = ("solve verify explain range crude-value cases recursion demo serve desktop io shadow model info "
            "bench convert presolve doctor engines completion")
    if shell == "bash":
        script = f"""# qenivo bash completion
_qenivo_completions() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}"
  local cmds="{cmds}"
  if [[ ${{COMP_CWORD}} -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$cmds" -- "$cur") )
  fi
}}
complete -F _qenivo_completions qenivo
"""
    elif shell == "zsh":
        script = f"""#compdef qenivo
_qenivo() {{
  local -a cmds
  cmds=({' '.join(f"'{c}'" for c in cmds.split())})
  _describe 'command' cmds
}}
compdef _qenivo qenivo
"""
    elif shell == "powershell":
        script = f"""Register-ArgumentCompleter -CommandName qenivo -ScriptBlock {{
  param($wordToComplete, $commandAst, $cursorPosition)
  $cmds = @({', '.join(repr(c) for c in cmds.split())})
  $cmds | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
  }}
}}
"""
    else:
        print(f"error: unknown shell {shell!r}", file=sys.stderr)
        return 2
    if a.json:
        _emit({"shell": shell, "script": script}, as_json=True)
    else:
        sys.stdout.write(script)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qenivo", description="Sovereign certified optimisation engine")
    p.add_argument("--json", action="store_true", help="JSON output (also accepted after the subcommand)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="solve an MPS model and certify the answer")
    s.add_argument("model")
    s.add_argument("--engine", default="auto", help="auto | simplex | ipm | pdhg | pdhg-rf | pdqp | milp | miqp | race | any registered plugin; "
                        "race runs simplex, interior point and PDHG on separate cores, the first certified answer wins")
    s.add_argument("--tol", type=float, default=1e-6)
    s.add_argument("--backend", default="auto", choices=["auto", "numpy", "cupy"])
    s.add_argument("--time-limit", type=float, default=3600.0)
    s.add_argument("--cert")
    s.add_argument("--sol")
    s.add_argument("--json", action="store_true")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_solve)

    v = sub.add_parser("verify", help="independently re-check a certificate")
    v.add_argument("model")
    v.add_argument("cert")
    v.add_argument("--exact", action="store_true", help="exact rational arithmetic")
    v.add_argument("--tol", type=float)
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_verify)

    e = sub.add_parser("explain", help="explain and prove infeasibility")
    e.add_argument("model")
    e.add_argument("--top", type=int, default=20)
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_explain)

    r = sub.add_parser("range", help="exact sensitivity ranging")
    r.add_argument("model")
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_range)

    cvv = sub.add_parser(
        "crude-value",
        help="exact cargo / column / RHS valuation curve with certificates and Excel/JSON/MD report",
    )
    cvv.add_argument("model", nargs="?", help="MPS/JSON model (omit with --demo)")
    cvv.add_argument("--cargo", help="cargo / purchase column (cost-price ray)")
    cvv.add_argument("--column-price-ray", help="any structural column for a cost ray")
    cvv.add_argument("--rhs-ray", help="row name for a parametric capacity / RHS ray")
    cvv.add_argument("--rhs-side", default="upper", choices=["lower", "upper", "both"])
    cvv.add_argument("--price-range", metavar="LO:HI",
                     help="parameter interval; use equals for negatives, e.g. --price-range=-20%%:+20%% or --price-range=-5:5")
    cvv.add_argument("--grid", type=int, help="unused placeholder (dense grid is bench-only)")
    cvv.add_argument("--cases", help="unused placeholder (multi-cargo cases.xlsx later)")
    cvv.add_argument("--tol", type=float, default=1e-9)
    cvv.add_argument("--unit", default="unit", help="unit label in break-even text")
    cvv.add_argument("--out", help="report path (.xlsx / .json / .md / .csv)")
    cvv.add_argument("--demo", action="store_true",
                     help="tiny refinery_level(1|2) demo if free RAM >= 1.5 GiB")
    cvv.add_argument("--demo-level", type=int, default=1, choices=[1, 2])
    cvv.add_argument("--json", action="store_true")
    cvv.set_defaults(fn=cmd_crude_value)

    c = sub.add_parser("cases", help="solve a what-if case stack (cases as JSON, CSV or .xlsx; results to .csv/.xlsx/.json)")
    c.add_argument("model")
    c.add_argument("cases")
    c.add_argument("--tol", type=float, default=1e-6)
    c.add_argument("--engine", default="auto",
                   help="auto (stack router) | simplex | ipm | pdhg | ...; forces a path when not auto")
    c.add_argument("--backend", default="auto", choices=["auto", "numpy", "cupy"])
    c.add_argument("--out")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_cases)

    rc = sub.add_parser("recursion", help="distributive recursion on the Haverly pooling problems")
    rc.add_argument("instance", choices=["haverly1", "haverly2", "haverly3"])
    rc.add_argument("--method", default="all",
                    choices=["all", "pdhg_warm", "pdhg_cold", "ipm", "simplex", "simplex_step"])
    rc.add_argument("--q0", type=float, default=2.0)
    rc.add_argument("--verbose", action="store_true")
    rc.add_argument("--json", action="store_true")
    rc.set_defaults(fn=cmd_recursion)

    # These keep their own option parsers; main() hands them their arguments before argparse runs.
    for name, text in (("desktop", "planner console in its own window, offline (per-launch token)"),
                       ("io", "command shell for scripts written for the classic optimizer shells"),
                       ("shadow", "run an external solver and QENIVO on one model and diff the answers")):
        sub.add_parser(name, help=text, add_help=False)

    sv = sub.add_parser("serve", help="local planner console and JSON API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--token-file", help="shared server: requests must send 'Authorization: Bearer <token>'")
    sv.add_argument("--proxy-user-header", help="company sign-in via reverse proxy, e.g. X-Remote-User")
    sv.add_argument("--trusted-proxy", action="append", default=[], help="address of that proxy (repeatable)")
    sv.add_argument("--audit-dir", help="write the audit trail here (one JSONL file per month)")
    sv.add_argument("--retention-days", type=int, default=180, help="audit retention (CERT-In: 180 days)")
    sv.add_argument("--json", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    mo = sub.add_parser("model", help="write a sector model (power, transport, supply chain, ...) as MPS")
    mo.add_argument("name", nargs="?", help="model name; omit to list the library")
    mo.add_argument("--param", "-p", action="append", default=[], help="size, e.g. -p units=20 -p periods=48")
    mo.add_argument("--out", "-o")
    mo.add_argument("--json", action="store_true")
    mo.set_defaults(fn=cmd_model)

    info = sub.add_parser("info", help="version, backends, GPU")
    info.add_argument("--json", action="store_true")
    info.set_defaults(fn=cmd_info)

    demo = sub.add_parser("demo", help="guided tour on built-in models")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(fn=cmd_demo)

    b = sub.add_parser("bench", help="run a declared benchmark set (wraps bench/run.py)")
    b.add_argument("set", help="netlib | netlib_infeas | kennington | gate | small | maros | ...")
    b.add_argument("--engine", default="auto")
    b.add_argument("--time-limit", type=float, default=120.0)
    b.add_argument("--out", help="results JSON path")
    b.add_argument("--backend", default="numpy", choices=["auto", "numpy", "cupy"])
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--with-ref", action="store_true", help="also run HiGHS comparator (requires highspy)")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_bench)

    cv = sub.add_parser("convert", help="convert between MPS and JSON (LP when available)")
    cv.add_argument("source")
    cv.add_argument("dest")
    cv.add_argument("--json", action="store_true")
    cv.set_defaults(fn=cmd_convert)

    pr = sub.add_parser("presolve", help="run engines.presolve and optionally write reduced MPS")
    pr.add_argument("model")
    pr.add_argument("out", nargs="?", default=None, help="output MPS path (optional)")
    pr.add_argument("--report", action="store_true", help="always print the JSON report")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(fn=cmd_presolve)

    doc = sub.add_parser("doctor", help="diagnose Python, NumPy, CuPy, GPU, native simplex, compiler, provenance")
    doc.add_argument("--json", action="store_true")
    doc.set_defaults(fn=cmd_doctor)

    eng = sub.add_parser("engines", help="list built-in and plugin engines")
    eng.add_argument("--json", action="store_true")
    eng.set_defaults(fn=cmd_engines)

    comp = sub.add_parser("completion", help="print shell completion script")
    comp.add_argument("shell", choices=["bash", "zsh", "powershell"])
    comp.add_argument("--json", action="store_true")
    comp.set_defaults(fn=cmd_completion)

    return p


PASSTHROUGH = {"desktop": ".desktop.app", "io": ".compat.shell", "shadow": ".compat.shadow"}


def main(argv=None):
    _quiet_cupy_path_warning()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from .banner import launch

        launch()
        return 0
    if argv and argv[0] in PASSTHROUGH:
        import importlib

        from .provenance import install_tripwires
        install_tripwires()
        return importlib.import_module(PASSTHROUGH[argv[0]], __package__).main(argv[1:])
    # Allow `qenivo --json <cmd> ...` and `<cmd> --json` uniformly.
    global_json = False
    if "--json" in argv:
        global_json = True
    p = build_parser()
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 2
        raise SystemExit(code) from None
    if global_json:
        a.json = True
    from .provenance import install_tripwires
    install_tripwires()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
