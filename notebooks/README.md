# Round-2 A100 notebook — how to run

One-hour Colab campaign that measures only what the first evidence run did not cover.

## Files

| File | Role |
|---|---|
| `notebooks/a100_round2.ipynb` | Colab notebook (A100) |
| `notebooks/build_round2.py` | regenerates the notebook |
| `bench/round2_campaign.py` | the parts (`stamp`, `tight`, `l5`, `refinery`, `headline`) |

Regenerate the notebook after editing the builder:

```bash
python notebooks/build_round2.py
```

## Colab (full A100 run, ≤ 60 min)

1. Runtime → Change runtime type → **A100 GPU** (High-RAM if offered).
2. Open `a100_round2.ipynb`.
3. In the settings cell, pick one code source:
   - `CODE_SOURCE = "upload"` — upload `qenivo.zip` built from branch `main`, or
   - `CODE_SOURCE = "github"` — clones `https://github.com/dhruva137/Q.E.N.I.V.O.git` at `main` and copies the `qenivo/` package to `/content/qenivo`.
4. Runtime → Run all. Allow multiple downloads; a results zip is saved after every part.

Parts and budgets: stamp 2 min · tight 25 min · l5 12 min · refinery 20 min · headline 1 min.

Resume: re-run with the same `RUN_NAME`; finished JSONL keys are skipped.

## Local CPU smoke (`--quick`)

Tiny substitutes only (afiro, L1 batches, short homotopy). Do **not** run the full A100 parts locally.

```powershell
$env:PYTHONPATH = "<worktree>/qenivo/src"
python -u qenivo/bench/round2_campaign.py --run-dir qenivo/runs/r2_smoke --quick
```

Expect JSONL under the run dir plus `HEADLINE_ROUND2.md`. cuPDLPx numbers in the headline are read from `bench/results/a100_evidence_run1/e2_large_lp.jsonl` and are never re-run.
