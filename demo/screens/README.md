# Demo screenshots checklist

Capture at **1600×900** (browser zoom 100%, OS display scaling noted in the shot name if not 100%). Save as `demo/screens/<id>.png`. Do **not** invent numbers in overlays; crop live UI or paste figures only from cited results files.

Owner records video separately (OBS). These stills feed the deck and portal.

| ID | File | Shot | UI path / command | Must show | Number source |
|---|---|---|---|---|---|
| S01 | `S01_problem_chrome.png` | Console chrome / brand | `qenivo serve` → `/` | QENIVO mark, local-only foot | — |
| S02 | `S02_model_loaded.png` | Model loaded | *Load MRPL-style sample* | `diesel_week` meta, local only | live |
| S03 | `S03_solve_certified.png` | Solve + seal | *Solve & certify* | verdict seal, residuals, top_values, marginals | live; optional Netlib cite `bench/results/netlib_release_20260930.json` |
| S04 | `S04_shadow_prices.png` | Marginals close-up | same result, *Marginal values* panel | row names + duals | live |
| S05 | `S05_crude_valuation.png` | Valuation entry | *Crude valuation* → *Value this cargo* | Until W02 merge: badge **needs w02 merge**; after merge: curve + breakpoints | stub now; later `bench/results/crude_value_*.jsonl` |
| S06 | `S06_case_stack_badges.png` | What-if stack | *What-if cases* → *Run case stack* | per-case **CERTIFIED** lamps | live small stack |
| S07 | `S07_stack_speed_slide.png` | Speed evidence | results slide or terminal `type`/`Get-Content` of HEADLINE | L4×512 **13.66 s** vs HiGHS **767.5 s** | `bench/results/a100_evidence_run1/HEADLINE.md` |
| S08 | `S08_laptop_stack.png` | Laptop GPU | optional | L4×64 vs HiGHS 16-core | `bench/results/persistent_batch_rtx5050.jsonl` |
| S09 | `S09_verify.png` | Independent verify | *Verify certificate* | checks pass/fail lamps | live |
| S10 | `S10_provenance.png` | Sovereignty | `qenivo info` + UI provenance chip | `provenance.clean: true` | live |
| S11 | `S11_audit_log.png` | Audit | serve with `--audit-dir` · console *Run history* or open `audit-YYYY-MM.jsonl` | utc / path / verdict rows | live |
| S12 | `S12_netstat_loopback.png` | Air-gap | `netstat` / Resource Monitor filtered to python | listen `127.0.0.1:8765`, no solve egress | live |

## Capture tips

* Dark theme default in package UI; keep one light-theme alternate only if the deck needs it.
* Hide bearer-token fields if empty clutter; do not show real shared-server tokens.
* For S07/S08 prefer a clean crop of the results markdown/JSONL rather than a busy IDE.
* Leave PNGs uncommitted if huge; checklist stays in git. Suggested max ~400 KB each.

## Status

Captured 2 Oct 2026 by `python packaging/screenshots.py` (headless Edge, 1600x900, dark theme, live
solves on a 4-core laptop with the native cores; every number on screen is computed, none typed).

| ID | File | Captured? |
|---|---|---|
| S01 | `S01_problem_chrome.png` | yes |
| S02 | `S02_model_loaded.png` | yes |
| S03 | `S03_solve_certified.png` | yes (refinery L3, simplex, 1e-8) |
| S04 | `S04_shadow_prices.png` | yes (ranked limits, planner units) |
| S05 | `S05_crude_valuation.png` | yes (L3, Basrah Light S1 P1, +/-20%, every segment certified) |
| S06 | `S06_case_stack_badges.png` | yes (whole-grade cases, net margin) |
| S07, S08 | results files | not UI views; crop the cited results files |
| S09 | `S09_verify.png` | yes |
| S10 | `S10_provenance.png` | yes |
| S11, S12 | audit file / netstat | operating-system views; owner |
| S13 | `S13_sensitivity.png` | yes (Williams ranging) |
