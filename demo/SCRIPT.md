# QENIVO demo script (≤ 3:10)

Owner records with OBS at 1080p (no lyrics in music). Captions optional. This file is narration + click path only — **do not invent numbers**; every on-screen figure must come from a results file below or be spoken as TBD until measured.

**Prep (before record):** `qenivo serve --host 127.0.0.1 --port 8765 --audit-dir audit` · browser `http://127.0.0.1:8765/` · optional second terminal for `qenivo info` and `netstat`. Offline: no CDN, no VPN required.

**Total target:** 3:00–3:10.

---

## 0:00 – 0:30 · Problem (MRPL margin, Red Sea / Hormuz)

**Show:** Title card or console chrome only (QENIVO mark visible). Optional one static slide behind.

**Say:**
> Mangalore Refinery plans on imported crude under thin GRM. In July 2026 MRPL became the first Indian state refiner to avoid Red Sea and Hormuz routing — more cargo choices, more what-if cases, and the solver underneath PIMS still sits on a foreign licence. QENIVO is an indigenous LP/MILP/QP engine that certifies every answer on-prem, air-gapped.

**Numbers (context only; cite if shown):** GRM / PAT narrative is from research notes, not a `bench/results` row — **do not put unverified $ figures on screen**. Problem framing only.

---

## 0:30 – 0:50 · Load the MRPL-style model

**Click:** *Load MRPL-style sample* (or drop a generated `crude_blending` / `refinery_pims` MPS written earlier with `qenivo model … -o …`).

**Say:**
> Here is a small diesel-week LP in the shape a planner exports: crude buys, unit capacity, product contract, sulphur. Nothing leaves this machine — the model stays in the browser and the local API.

**On-screen:** model name, byte/line count. Live objective after solve is **TBD at record time** (not pre-filled).

---

## 0:50 – 1:20 · Solve + proof

**Click:** *Solve & certify* (engine `auto`, tol `1e-6` for the sample; for GPU stack shots use `1e-4` and cite the matching results file).

**Say:**
> Solve. The seal is the certificate: primal/dual residuals and duality gap against the original matrix. If the verdict is optimal, the plan and shadow prices are proven — not a screenshot of a foreign solver.

**Fill at record time:**
| Field | Placeholder | Source |
|---|---|---|
| Objective | `TBD_LIVE_OBJ` | live solve response |
| Engine / time | `TBD_LIVE_ENGINE` / `TBD_LIVE_SEC` | live `engine` block |
| Verdict | optimal / … | live |

**Bench parity (if you flash Netlib):** 97/98 verified optimal — `bench/results/netlib_release_20260930.json`. Infeasible Netlib Farkas 25/29 — `bench/results/netlib_infeas_20260930.json`.

---

## 1:20 – 2:00 · Crude valuation curve and break-even

**Click:** rail *Crude valuation* → column `crude_light` (or cargo name) → *Value this cargo*.

**Say (honest):**
> This is the flagship: the exact break-even price curve for a spot cargo, with a proof on every segment and plain-language “why the plan changed.” The UI entry point is here. The engine lives on branch `w02-crude-value` and is **not yet merged to main** — the API returns `needs w02 merge` until that lands. After merge, replace this beat with the live curve and the measured breakpoints.

**Placeholders (post W02 merge / L3–L4 gate):**
| Field | Placeholder | Expected results path |
|---|---|---|
| Wall time (L4, one crude, ±20%) | `TBD_CV_SEC` | `bench/results/crude_value_*.jsonl` (W02; **not measured on this branch**) |
| Breakpoints found | `TBD_CV_BPS` | same |
| HiGHS grid time (200 / 1000 warm) | `TBD_HIGHS_GRID_SEC` | same |
| Breakpoints missed by grid | `TBD_GRID_MISS` | same |

Until those rows exist, **do not invent break-even dollars**. Show the stub badge and move on.

---

## 2:00 – 2:40 · Case stack (CPU / GPU) vs HiGHS

**Click:** *What-if cases* → *Run case stack* on the built-in five-case JSON (demo flow). For the **speed claim**, cut to a terminal or results slide — do not invent a 500-case live run on the laptop under time pressure.

**Say:**
> Planners do not solve one LP; they solve a stack. On an A100, an L4 refinery matrix times 512 what-if cases at tolerance 1e-4 finished in **13.66 s**, all 512 certified, against HiGHS on all CPU cores at **767.5 s** — about **56×**. That row is in our evidence pack.

**Cite on screen / voice:**
| Claim | Number | File |
|---|---|---|
| A100 L4 × 512 @ 1e-4 | GPU **13.66 s**, 512/512 certified; HiGHS all-cores **767.50 s**; **56.19×** | `bench/results/a100_evidence_run1/HEADLINE.md` (and `e3_batch.jsonl` key `L4/S512/tol0.0001`) |
| A100 L5 single @ 1e-4 | GPU **2.00 s** vs HiGHS **180.41 s**; **90.32×** | same HEADLINE |
| Laptop RTX 5050 L4 × 64 @ 1e-4 | GPU batch **7.27 s** (non-persistent path) vs HiGHS 16-core **207.32 s** ≈ **28.5×** | `bench/results/persistent_batch_rtx5050.jsonl` (L4 / S=64 row) |

**Still TBD this week (W01 fair harness):**
| Field | Placeholder | Path when ready |
|---|---|---|
| Fair 500-case CPU warm chain | `TBD_W01_500_WARM_SEC` | `bench/results/` stack JSONL from W01 |
| Same stack HiGHS cold / warm | `TBD_HIGHS_COLD` / `TBD_HIGHS_WARM` | same (both columns required by R4) |
| Router choice CPU vs GPU | `TBD_ROUTER` | W03 when present |

Script beat says “500-case”; measured headline today is **512** on A100 — say “five hundred–plus” or “512” and point at the file. Do not round 512 → 500 on a results slide.

---

## 2:40 – 3:00 · Sovereignty

**Show:** `qenivo info` in a terminal (provenance clean) and/or console chip *provenance clean*. Optional: Task Manager / `netstat` with serve bound to `127.0.0.1` only — **no outbound solve traffic**.

**Say:**
> Provenance guard: no HiGHS, no CPLEX, no cuSolver in the solve path. The console is local-only. Certificates and the audit JSONL stay on disk for CERT-In retention when `--audit-dir` is set.

---

## 3:00 – 3:10 · Close

**Say:**
> QENIVO: indigenous optimisation with a proof on every plan — built for Indian refining under real crude and sovereignty pressure. Epoch Zero, SIH 2026.

**End card:** team / PS 26119 / “numbers from `bench/results/`”.

---

## Timing checklist (owner)

| Beat | Max | Hard stop |
|---|---|---|
| Problem | 0:30 | 0:30 |
| Load | 0:20 | 0:50 |
| Solve + proof | 0:30 | 1:20 |
| Crude valuation | 0:40 | 2:00 |
| Case stack / speed | 0:40 | 2:40 |
| Sovereignty | 0:20 | 3:00 |
| Close | 0:10 | **3:10** |

If over time: cut valuation explanation (keep stub one-liner) and sovereignty netstat; keep A100 13.66 / 767 numbers.

## Recording notes (owner / OBS)

* 1080p, captions file beside the MP4; no music with lyrics.
* Put the MP4 in `demo/video/` (git-ignored if large); upload as a release asset later.
* Screenshots for the deck: see `demo/screens/README.md`.
