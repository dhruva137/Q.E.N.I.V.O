"""`qenivo demo`: a guided tour on built-in models (no downloads, CPU only, ~1-2 minutes)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path


def _h(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def run_demo() -> int:
    import numpy as np

    from .api import solve
    from .certify.certificate import load
    from .certify.verify import format_report, verify
    from .io.mps import write_mps
    from .model import ModelBuilder
    from .models.williams import PUBLISHED_OPTIMUM, williams_lp
    from .provenance import guard
    from .workload.cases import Case, solve_cases
    from .workload.explain import explain_infeasibility, format_explanation
    from .workload.recursion import HAVERLY_OPTIMA, haverly, run_recursion

    tmp = Path(tempfile.mkdtemp(prefix="qenivo_demo_"))
    t_all = time.perf_counter()

    _h("1. Solve the Williams refinery LP and certify the answer")
    p = williams_lp()
    sol = solve(p, tol=1e-9)
    print(sol.summary())
    print(f"  published optimum (Williams, Model Building in Math. Programming): {PUBLISHED_OPTIMUM:,.2f}")
    mps, cert = tmp / "williams.mps", tmp / "williams.cert.json"
    write_mps(p, mps)
    sol.save_certificate(cert)

    _h("2. Re-check that certificate with the independent verifier (own parser, exact arithmetic)")
    print(format_report(verify(mps, load(cert), exact=True, tol=1e-9)))

    _h("3. What-if case stack: crude prices and a reformer outage, every case certified")
    cases = [Case("base"),
             Case("crude1 +10% availability", col_hi={"crude1": 22000}),
             Case("reformer at 70%", row_hi={"reform_cap": 7000}),
             Case("premium price +1", cost={"PMF": 8.0}),
             Case("jet price -0.5", cost={"JF": 3.5})]
    rep = solve_cases(p, cases, tol=1e-8, backend="numpy")
    print(rep.summary())

    _h("4. An infeasible plan, explained in the planner's own terms and proven")
    b = ModelBuilder("blend_week", maximize=True)
    b.var("crude_light", 0, 400, obj=-60); b.var("crude_heavy", 0, 300, obj=-45)
    b.var("diesel", 0, np.inf, obj=90)
    b.row("unit_capacity", {"crude_light": 1, "crude_heavy": 1}, hi=600)
    b.row("diesel_yield", {"crude_light": 0.55, "crude_heavy": 0.40, "diesel": -1}, lo=0, hi=0)
    b.row("diesel_contract", {"diesel": 1}, lo=420)
    q = b.build()
    print(format_explanation(explain_infeasibility(q)))
    s2 = solve(q)
    print(f"  solve() verdict: {s2.verdict.upper()} (certificate carries the Farkas ray; "
          f"check {s2.ray_check and s2.ray_check.get('valid')})")

    _h("5. Distributive recursion on Haverly's pooling problem (published optimum 400)")
    hp, pools = haverly(1)
    for q0 in (1.0, 2.0, 3.0):
        print(f"  start pool sulphur {q0:g}%")
        for mth in ("pdhg_warm", "ipm", "simplex", "simplex_step"):
            r = run_recursion(hp, pools, method=mth, q0=q0, max_passes=40)
            print("    " + r.summary())
    print(f"  published global optimum: {HAVERLY_OPTIMA[1]:g}")

    _h("6. Provenance")
    g = guard()
    print(f"  solver packages imported: {g['forbidden_packages'] or 'none'}")
    print(f"  calls into sparse direct solvers / optimisers (tripwires {'armed' if g['tripwires_installed'] else 'NOT armed'}): "
          f"{len(g['tripwire_hits'])}")
    print(f"  -> {'CLEAN: every answer above came from QENIVO code' if g['clean'] else 'VIOLATION'}")
    print(f"\nDemo finished in {time.perf_counter() - t_all:.1f}s. Files in {tmp}")
    return 0
