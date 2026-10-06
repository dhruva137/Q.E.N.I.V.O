"""Native simplex speed: baseline revision vs working tree vs HiGHS simplex, interleaved.

    PYTHONPATH=src python bench/native_speed.py 25fv47 bnl2 --reps 3 --baseline 373c72c \
        --out bench/results/native_simplex_speed_round2.jsonl

The baseline core is taken from `git show REV:qenivo/src/qenivo/native/simplex_core.cpp` and built
next to the current one (the DLL name is keyed by the source hash, so both coexist). Each repetition
runs HiGHS, baseline and current back to back, so machine load affects all three alike. Every answer
is checked: objective vs HiGHS and the float64 KKT residual on the original model. HiGHS is used
only here, as the reference; the solve path never calls a solver library.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qenivo import read  # noqa: E402
from qenivo.certify.kkt import kkt_residuals  # noqa: E402
from qenivo.engines import native_simplex as ns  # noqa: E402

DEFAULT = ["25fv47", "bnl2", "scfxm3", "ship08l", "degen2", "sctap3", "pilot4", "bnl1", "czprob", "d6cube",
           "pilot87", "greenbea", "80bau3b", "pilot.ja"]


def build(src: Path | None):
    """Load the native library for `src` (None: the working-tree source)."""
    saved = ns.SRC
    try:
        if src is not None:
            ns.SRC = src
        ns._LIB.clear()
        lib = ns.library()
        if lib is None:
            raise RuntimeError(ns.status())
        return lib
    finally:
        ns.SRC = saved
        ns._LIB.clear()


def run_native(lib, p, tl):
    ns._LIB.update(lib=lib, reason="ok")
    t = time.perf_counter()
    r = ns.solve_simplex_native(p, time_limit=tl)
    r["wall"] = time.perf_counter() - t
    return r


def run_highs(path):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("presolve", "off")
    h.readModel(str(path))
    t = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t
    info = h.getInfo()
    return dt, info.objective_function_value, h.modelStatusToString(h.getModelStatus()), info.simplex_iteration_count


def load(name):
    path = HERE / "instances" / f"{name}.mps"
    if not path.exists():
        from fetch import fetch
        fetch(name)
    return path, read(path)


def check(p, r, hobj):
    if r["status"] != "optimal":
        return None, None
    k = kkt_residuals(p, r["x"], r["y"])
    return k["max_rel"], abs(k["objective"] - hobj) / max(1.0, abs(hobj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline", default=None, help="git revision of the baseline core (omit: no baseline)")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = None
    if a.baseline:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                              cwd=HERE).stdout.strip()
        src = subprocess.run(["git", "show", f"{a.baseline}:qenivo/src/qenivo/native/simplex_core.cpp"],
                             capture_output=True, cwd=root, check=True).stdout
        tmp = Path(tempfile.gettempdir()) / f"qenivo_core_{a.baseline}.cpp"
        tmp.write_bytes(src)
        base = build(tmp)
    cur = build(None)
    rows = []
    for name in a.names:
        path, p = load(name)
        H, B, C = [], [], []
        for _ in range(a.reps):
            H.append(run_highs(path))
            if base is not None:
                B.append(run_native(base, p, a.time_limit))
            C.append(run_native(cur, p, a.time_limit))
        hobj = H[0][1]
        row = {"instance": name, "rows": p.m, "cols": p.n, "nnz": int(p.A.nnz),
               "highs_time": statistics.median(h[0] for h in H), "highs_obj": hobj, "highs_status": H[0][2],
               "highs_iters": H[0][3]}
        for tag, R in (("before", B), ("after", C)):
            if not R:
                continue
            r = R[-1]
            mr, rd = check(p, r, hobj)
            row.update({f"{tag}_time": statistics.median(x["wall"] for x in R), f"{tag}_status": r["status"],
                        f"{tag}_iters": r["iterations"], f"{tag}_max_rel": mr, f"{tag}_rel_obj_diff": rd})
        row["ratio_after"] = row["after_time"] / row["highs_time"]
        if B:
            row["ratio_before"] = row["before_time"] / row["highs_time"]
        rows.append(row)
        fmt = lambda v: "-" if v is None else (f"{v:.1e}" if isinstance(v, float) else str(v))  # noqa: E731
        print(f"{name:10s} m={p.m:5d} highs {row['highs_time']:7.3f}s ({row['highs_iters']:6d} it)"
              + (f" | before {row['before_time']:7.3f}s {row['before_iters']:6d} it {row['before_status'][:4]}"
                 f" x{row['ratio_before']:6.1f}" if B else "")
              + f" | after {row['after_time']:7.3f}s {row['after_iters']:6d} it {row['after_status'][:4]}"
                f" x{row['ratio_after']:6.1f} kkt {fmt(row['after_max_rel'])} dobj {fmt(row['after_rel_obj_diff'])}",
              flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    ra = [r["ratio_after"] for r in rows]
    print(f"median ratio after {statistics.median(ra):.2f}" +
          (f", before {statistics.median(r['ratio_before'] for r in rows):.2f}" if a.baseline else ""))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
