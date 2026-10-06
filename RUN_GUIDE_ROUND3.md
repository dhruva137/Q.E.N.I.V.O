# RUN GUIDE — Round 3 A100 notebook

One page for the owner. Campaign budget ≈ **5–8 GPU hours** on Colab A100 High-RAM.

## W10 predictions

Eligible IDs (laptop already beats comparator): **C-BATCH-L4**, **C-PERSIST-S1**, **C-STACK-CARGO**, **C-STACK-FPRICE**.

Ship `research_prototypes/transfer/PREDICTIONS.md` and `transfer_model_2026-10-01.json` in the zip (or upload to `/content/…/transfer/`). Each part prints its prediction block before running. Blocked parts (`crude`, `diet`, `simplex`, `qp_milp`) print **PENDING** with the W10 gate id.


## Colab steps

1. **Runtime → Change runtime type → A100 GPU**, enable **High-RAM** if offered → Save.
2. Upload `notebooks/a100_round3.ipynb` (or open from the repo).
3. In the **settings** cell:
   - `RUN_NAME = "round3_run1"` (change only if starting a fresh campaign).
   - `CODE_SOURCE = "upload"` **or** `"github"`.
   - Leave `SELF_TEST = True` (runs `--quick` once before the long parts).
   - Set `W05_PASSED` only if Stack Simplex laptop gate passed (`os.environ["W05_PASSED"]="1"`).
4. If using upload: put **`qenivo.zip`** in the session (Files pane or the upload prompt).
5. Optionally upload **`PREDICTIONS.md`** to `/content/PREDICTIONS.md`.
6. **Runtime → Run all.** Allow multiple downloads.

## Cells (in order)

| # | Cell | Action |
|---|---|---|
| 1 | Title (markdown) | Read budgets / parts |
| 2 | Settings | Edit `RUN_NAME`, `CODE_SOURCE`, `PARTS`, `W05_PASSED` |
| 3 | Setup | GPU assert, install qenivo, cupy, highspy |
| 4 | Run | `--quick` self-test, then each part under its wall-time cap; zip after every part |
| 5 | Headline | Displays `HEADLINE_ROUND3.md` |
| 6 | Final zip | Downloads `{RUN_NAME}_results.zip` again |

## What to download

After each part (and at the end): **`{RUN_NAME}_results.zip`** under `/content/`.

Inside the unzipped run dir expect:

- `stamp.jsonl`, `crude.jsonl`, `stack.jsonl`, `diet.jsonl`, `simplex.jsonl`, `large_lp.jsonl`, `qp_milp.jsonl`
- `HEADLINE_ROUND3.md` (prediction vs measured; numbers from JSONL only)

Keep every zip — resume is by re-running with the **same** `RUN_NAME` (finished keys are skipped).

## Local CPU smoke (before handing to owner)

No GPU. Uses **L2 + tiny S**.

```powershell
$env:PYTHONPATH = "<worktree>\src"
python -u bench/round3_campaign.py --run-dir runs/r3_smoke --quick
```

Require free RAM ≥ ~1.5 GB. Expect JSONL + `HEADLINE_ROUND3.md` with many `PENDING` prediction lines until W10 lands.

## Skip / gate notes

| Part | Skip when |
|---|---|
| `simplex` | W05 not passed (`W05_PASSED` unset/0) or module missing → records PENDING |
| `diet` diet mode | W04 `byte_diet` missing → PENDING row (baseline still runs) |
| `crude` / `stack` full APIs | W02/W01/W03 missing → skeleton PDHG smoke still records rows |
| `qp_milp` | W07/W08 missing → tiny simplex smoke row |

## Do not

- Do not invent numbers in the headline.
- Do not re-run cuPDLPx; compare against `bench/results/a100_evidence_run1/e2_large_lp.jsonl`.
- Do not run full L4/L5 / S=4096 locally.
