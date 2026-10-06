"""PARITY.md: one table of coverage numbers from the bench/run.py result files of a run folder.

    python bench/parity_summary.py <run-dir>
"""
import json
import sys
from pathlib import Path

PROVEN = ("optimal", "infeasible", "unbounded")


def summarise(path: Path) -> dict:
    d = json.load(open(path))
    runs = d["runs"]
    n = len(runs)
    proven = [r for r in runs if r.get("qenivo", {}).get("verdict") in PROVEN]
    ver = [r for r in proven if r["qenivo"].get("verified")]
    opt = [r for r in ver if r["qenivo"]["verdict"] == "optimal"]
    inf = [r for r in ver if r["qenivo"]["verdict"] == "infeasible"]
    agree = [r for r in opt if (r.get("rel_obj_diff") is not None and r["rel_obj_diff"] <= 1e-6)
             or (r.get("rel_obj_diff_published") is not None and r["rel_obj_diff_published"] <= 1e-5)]
    wrong = [r["instance"] for r in opt if r.get("rel_obj_diff") is not None and r["rel_obj_diff"] > 1e-4
             and (r.get("rel_obj_diff_published") is None or r["rel_obj_diff_published"] > 1e-4)]
    highs_ok = [r for r in runs if r.get("highs", {}).get("status") in ("Optimal", "Infeasible")]
    engines = {}
    for r in ver:
        e = f"{r['qenivo'].get('engine')}/{r['qenivo'].get('backend')}"
        engines[e] = engines.get(e, 0) + 1
    return {"set": path.stem, "instances": n, "proven": len(proven), "verified": len(ver), "optimal_agree": len(agree),
            "infeasible_farkas": len(inf), "highs_solved": len(highs_ok), "wrong": wrong,
            "missed": sorted(r["instance"] for r in runs if r not in ver), "engines": engines,
            "commit": d["meta"].get("commit"), "gpu": d["meta"].get("gpu")}


def main(run_dir):
    run_dir = Path(run_dir)
    rows = [summarise(p) for p in sorted(run_dir.glob("parity_*.json"))]
    L = ["# Coverage (parity) evidence", "",
         "Every counted answer was proven by the engine AND re-checked by the independent verifier.", "",
         "| set | instances | proven + verified | optimal & agrees with reference | infeasible with Farkas proof | HiGHS solved (same limits) |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['set'].replace('parity_', '')} | {r['instances']} | {r['verified']} | {r['optimal_agree']} | "
                 f"{r['infeasible_farkas']} | {r['highs_solved']} |")
    L.append("")
    for r in rows:
        L.append(f"**{r['set'].replace('parity_', '')}**: engines {r['engines']}; not verified: {' '.join(r['missed']) or 'none'}"
                 + (f"; objective disagrees with reference (>1e-4): {' '.join(r['wrong'])}" if r["wrong"] else ""))
    if rows:
        L += ["", f"commit {str(rows[0]['commit'])[:10]} · GPU {rows[0]['gpu']}"]
    (run_dir / "PARITY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
