"""Measured multi-core speed-up: a what-if case stack solved on 1, 2, 4, ... cores.

    python bench/multicore.py --level 2 --cases 32 --out bench/results/multicore.json
"""
import os

# one BLAS thread per process, so "N cores" means N processes, not N x (hidden BLAS threads)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None):
    from qenivo.models.refinery import LEVELS, refinery_lp
    from qenivo.workload.parallel import solve_parallel
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--cases", type=int, default=32)
    ap.add_argument("--engine", default="ipm")
    ap.add_argument("--out", default="bench/results/multicore.json")
    a = ap.parse_args(argv)
    base = refinery_lp(*LEVELS[a.level], seed=0)
    rng = np.random.default_rng(7)
    probs = [dataclasses.replace(base, c=base.c * rng.uniform(0.9, 1.1, base.n), name=f"case{k}") for k in range(a.cases)]
    cores = os.cpu_count() or 1
    rows, ref = [], None
    for w in [1, 2, 4, 8, 16, 32]:
        if w > cores:
            break
        r = solve_parallel(probs, workers=w, engine=a.engine)
        objs = np.array([x["objective"] for x in r["results"]])
        ref = objs if ref is None else ref
        rows.append({"workers": w, "wall": r["wall"], "certified": sum(x["verdict"] == "optimal" for x in r["results"]),
                     "max_obj_diff_vs_1core": float(np.max(np.abs(objs - ref) / (1 + np.abs(ref))))})
        print(f"{w:3d} cores: {r['wall']:7.2f}s  speed-up {rows[0]['wall'] / r['wall']:.2f}x  "
              f"certified {rows[-1]['certified']}/{a.cases}", flush=True)
    out = {"model": f"refinery L{a.level} ({base.m} rows, {base.n} cols)", "cases": a.cases, "engine": a.engine,
           "cpu_count": cores, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
