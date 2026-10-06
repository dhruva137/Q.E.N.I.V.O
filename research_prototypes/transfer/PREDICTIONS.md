# W10 Predictions — laptop → A100 (2026-10-01)

Analysis only. No GPU runs. Free RAM ~2 GB at write time.

**Rule:** W11 gets an experiment only if the laptop already beats the named comparator.
Everything else is **blocked** or marked **pending laptop gate**.

---

## Sources (read-only)

| Tag | Path |
|---|---|
| W01 | `bench/results/stack_bench_2026-10-01.{jsonl,md}` (L3×64 only; S=256 incomplete) |
| W02 | worklog not in this tree — code done; L3/L4 HiGHS midpoint gate **open** |
| W04–W06 | no worktrees / no results; `STATUS.md` = **hold** |
| CPU05 | cpu case-structure notes, not in this tree |
| GPU04 | gpu lab notes, not in this tree |
| A100 | `nirnay/bench/results/a100_evidence_run1/` (+ `gpu_results/evidence_run1/`) |
| T4 | `gpu_results/kaggle_t4_run1/.../T4.md` |
| RTX | `nirnay/bench/results/persistent_batch_rtx5050.jsonl` |
| Alpha | internal note, not in this tree |

Calibration JSON: `bench/results/transfer_model_2026-10-01.json`.

---

## Transfer regimes (calibrated)

| Regime | Spec ratio (laptop→A100) | **Calibrated** wall factor | Cal. error vs A100 rows |
|---|---|---|---|
| BW-bound batch PDHG, S≥64 | 246 GB/s → ~1.5–2.0 TB/s ≈ **6–8×** | **3.0–3.5×** (use **3.5×** median) | L4×512: 48.24/13.66=**3.53×**; L4×128: 14.17/4.99=**2.84×** → naive BW **~46% high**; calibrated factor within **±20%** of those two rows |
| Latency / small-S persistent | 20→108 SM | **~4–8×** iter time (not SM×) | A100 run1 launch-path ~266 µs/iter (HIDDEN_ALPHA); laptop persistent qap15 29 / L4 69 / L5 103 µs — no A100 persistent row yet |
| fp64 dense (LU / IR) | 0.107 → 9.7 TFLOP/s ≈ **90×** | **30–90×** (memory + host share) | **unvalidated** — W05/W06 pending |
| CPU warm chains | 16 laptop threads vs 12 Colab vCPU | **0.7–1.2×** wall (same order) | algorithmic ratios hardware-stable |
| L2 tiling | 32→40 MB | **~1.0–1.2×** extra vs laptop tiling gain | micro only |

**Branch rule (pre-flight):** run `e6_overhead_profile.py` L4×512 on A100 first. If iteration kernels **&lt; 70%** of wall, shift engineering to host-side (setup / KKT / Python), not SpMM. Laptop E6: kernels **74–80%** of wall.

---

## Ranked candidates

Score ≈ (predicted advantage vs comparator on A100 VM) × confidence × (1 / GPU-hours).
Only **W11** rows are eligible to spend A100 GPU time under the gate.

### W11 — laptop already beats comparator

| Rank | ID | Experiment | Laptop win (measured) | Comparator | Pred. A100 wall | Interval | Regime | Formula / source | Conf. | GPU-h | W11 part |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **C-BATCH-L4** | Batch PDHG L4/L5 × {512,1024,4096}, iid/independent stack, 1e-4 (+ optional 1e-6) | L4×64 RTX: **7.27 s** vs HiGHS-16c **207 s** (**28×**); L4×512 E6 wall **48.2 s** | HiGHS cold **and** all-core on same VM; also CPU warm on iid (laptop CPU warm L4×64 iid **499 s**) | L4×512 @1e-4: **already 13.66 s** (56× vs HiGHS 767 s). L4×1024: **~22–30 s**; L4×4096: **~70–110 s** @1e-4 | ±25% | BW + fixed overhead | `T_A100 ≈ T_laptop/3.5 + 0.3·T_fixed`; fixed ~20–35% on A100 (E6 note) | **0.90** | 0.5–1.0 | §2 subset (independent) + scale |
| 2 | **C-PERSIST-S1** | Persistent (occupancy-sized) vs launch on L4/L5/qap15, S∈{1,4,16} | Persistent **5–20×** launch at S=1 (14–103 vs 350–454 µs/iter) | Launch path on same A100; HiGHS for L4/L5 S=1 (A100 already **9.9× / 90×** @1e-4) | qap15 ~**10–20 µs/iter**; L4 f64 ~**15–35 µs**; L5 f64 ~**20–45 µs** | ±40% | latency / SM | scale µs by ~3–5× from laptop persistent (BW floor for L5) | **0.70** | 0.25 | §5 preflight + small-S |
| 3 | **C-STACK-CARGO** | CPU warm stack `cargo_menu` / one-factor L4–L5 × S∈{256,1024} | L3×64: q-warm **4.47 s** vs h-warm **5.57** vs h-cold **38.4** (zp **57.8%**) | HiGHS warm (fair) + cold | Wall ~ laptop × **0.9–1.2** (CPU); advantage ratio **stable** | ±30% | CPU algorithmic | pivot counts unchanged; time ∝ single-core | **0.85** | 0 (CPU) | §2 one_crude/cargo kinds |
| 4 | **C-STACK-FPRICE** | CPU warm `factor_price` L3→L4 | L3×64: q-warm **6.03** vs h-warm **6.37** vs h-cold **37.4** (marginal) | HiGHS warm + cold | Same order as laptop; may flip on L4 | ±40% | CPU | W01 row; L4 **pending** S=256 | **0.55** | 0 (CPU) | §2 |
| 5 | **C-F32-CERT** | f32 iterate + f64 host cert @1e-4 (ablation on shipping kernels) | L3 persistent: **2.3×**; L4 launch: **1.3×** (E4); 64/64 certified | Same stack f64 baseline on A100 | Ratio **1.2–1.8×** retained (bytes half; A100 fp64 strong so gain **smaller** than laptop) | ±30% | BW / byte diet | `T_f32 ≈ T_f64 / r` with r∈[1.2,1.8] | **0.75** | 0.3 | §3 **only if** shipped in notebook from existing kernels — else treat as blocked pending W04 gate |
| 6 | **C-L2-TILE** | L2-sized tiles S≥256 | L4 f64 S=1024: **2.38→1.57 µs/case (1.51×)** | Untiled same S | **1.3–1.6×** vs untiled | ±25% | L2 | St≈L2/(8n) | **0.80** | fold into §2/§3 | micro → confirm in batch wall |

**Already confirmed on A100 (do not re-spend for discovery; optional regression only):** L3/L4 batch vs HiGHS cold (HEADLINE), L5×1 @1e-4 **90×**. Re-run only if kernels change.

### Blocked — laptop does **not** beat fair comparator (or loses)

| ID | Why blocked | Evidence |
|---|---|---|
| B-ONE-CRUDE-WARM | q-warm **behind** HiGHS warm | W01 L3×64: 5.46 vs **5.21**; CPU05 S=1000: 8.8 vs **8.3** |
| B-ONE-UNIT | behind HiGHS warm | 5.71 vs **4.95** |
| B-FACTOR-MIXED | behind HiGHS warm | 13.83 vs **7.55** |
| B-INDEPENDENT-CPU | behind cold **and** warm | 33.25 vs 20.25 / 20.11 — use **GPU** path (C-BATCH) |
| B-LARGE-LP-CUPDLPX | many A100 rows already lose to cuPDLPx @1e-6 | rmine15/Linf/cont1; only qap15/nug08 competitive @1e-4 |
| B-W07-MILP | loses HiGHS on small | 7.4 s vs **0.53 s** |
| B-ATLAS / fp16-A / cuSPARSE | measured failures | HIDDEN_ALPHA “not alpha” |

### Pending laptop gate — predictions provisional; **not** on W11 list

| ID | Workstream | Status | What would unlock W11 |
|---|---|---|---|
| P-W02-CRUDE | W02 Crude Valuation | code done; L3/L4 vs HiGHS midpoints **not run** (RAM) | Exact curve wall &lt; HiGHS dense grid (cold **and** warm-chained) on L3/L4 |
| P-W04-DIET | W04 byte diet | **hold** (no results) | L4×512 @1e-4 ≥**2×** baseline, all f64-certified |
| P-W05-SSX | W05 Stack Simplex | **hold** | Beat HiGHS warm 16-core @ L3 `factor_price` S=256 |
| P-W06-MP | W06 mixed precision | **hold** | ≥**3×** fp32+IR vs fp64 dense batch; &lt;20% IR fail |
| P-W01-L4 | W01 S=256 / L4 | incomplete | Finish fair table; re-rank kinds |
| P-W03-ROUTER | W03 | not launched | Needed to auto-pick C-BATCH vs C-STACK |

---

## Per–W11-part mapping

| W11 part | Verdict |
|---|---|
| 0 Pre-flight (5 min) | **Always** — triad, fp64 GEMM, launch/grid.sync, `e6_overhead_profile` L4×512 |
| 1 Crude Valuation L4/L5 | **BLOCKED** — pending W02 laptop gate |
| 2 Stack matrix | **Partial:** `cargo_menu` / `factor_price` (CPU) + `independent` via GPU (C-BATCH). **Omit** `factor_mixed`, `one_unit` until they beat HiGHS warm. L4/L5 S=1024+ only after W01 L4 numbers |
| 3 GPU byte diet | **BLOCKED** — pending W04 (≥2× gate). Optional: C-F32-CERT / C-L2-TILE as **confirmations of research E4/E2b only**, not as “product diet” claim |
| 4 Stack Simplex | **BLOCKED** — pending W05 kill test |
| 5 Single large LPs | **Narrow:** C-PERSIST-S1 + regression of run1 winners (qap15 @1e-4). **Do not** spend hours chasing cuPDLPx losses without laptop kernel win |
| 6 QP / MILP CPU | **BLOCKED** for W07/W08 until they beat HiGHS on laptop |
| 7 HEADLINE_ROUND3 | Only from recorded W11 rows + prediction deltas |

---

## Pre-flight probe list (first 5 minutes on A100)

1. **Bandwidth triad / copy** — 256 MB; record GB/s (expect ~1.5–2.0 TB/s usable).
2. **fp64 GEMM** — large square; record TFLOP/s (expect ~9–10).
3. **Empty launch + `grid.sync`** — µs (compare to laptop 23 µs launch / 1.4–1.7 µs sync).
4. **`e6_overhead_profile.py` L4 × 512** — if kernels &lt;70% wall → host-side plan branch.
5. Print this file’s C-* predictions beside each notebook part.

---

## Honest frame

A100 time buys **confirmation of the size of a laptop win**, not discovery.
Confirmed and refuted predictions both stay in `HEADLINE_ROUND3.md`.
W02/W04/W05/W06 are promising on paper; without laptop gates they stay off the W11 burn list.
