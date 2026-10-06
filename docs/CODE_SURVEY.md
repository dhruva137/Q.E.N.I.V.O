# Code survey: rival solver implementation choices

Date: 2026-10-01. Branch: `w13-survey`. Agent B (code reading).

Read-only survey of **data structures, tolerances, restart rules, and crossover / parametric
triggers** in six open solvers. Deepens and verifies claims in the research memo
`research/2026-10-01/03_frontier_and_white_space.md` (sibling worktree / parent research tree)
against public GitHub sources fetched 2026-10-01. **No rival source is copied into QENIVO `src/`.**

Status labels for each technique:

* **already in QENIVO** — present in this tree (independent implementation).
* **candidate** — useful to adopt or harden for W02 / W05 / W06; not yet (fully) in QENIVO.
* **not applicable** — wrong product shape for our what-if / certified stack, or no analogue.

| Project | Licence (verified) | Upstream |
|---|---|---|
| cuPDLPx | Apache-2.0 | https://github.com/MIT-Lu-Lab/cuPDLPx |
| NVIDIA cuOpt | Apache-2.0 | https://github.com/NVIDIA/cuopt |
| HiGHS | MIT | https://github.com/ERGO-Code/HiGHS |
| SoPlex | Apache-2.0 | https://github.com/scipopt/soplex |
| CLP | EPL-2.0 | https://github.com/coin-or/Clp |
| SCIP | Apache-2.0 | https://github.com/scipopt/scip |

---

## Technique status table

| Project | Technique | Status | Path read |
|---|---|---|---|
| cuPDLPx | Restarted Halpern PDHG + reflection; constant η = 0.998/σ_max(A); PID primal weight | **already in QENIVO** | `src/solver.cu`; `docs/algorithm/restart.md`; QENIVO `engines/pdhg.py` |
| cuPDLPx | Fixed-point-error restarts (sufficient 0.2 / necessary 0.5 / artificial 0.36) | **already in QENIVO** | `docs/algorithm/restart.md`; `PDHGOptions` |
| cuPDLPx | Dual CSR (A and Aᵀ); fused primal/dual kernels; CUDA Graphs over termination windows | **already in QENIVO** (fused + persistent cooperative kernel; Graphs are a candidate polish) | `docs/implementation/kernel-fusion.md`; `src/solver.cu` ~170–247 |
| cuPDLPx | Active-set boost (σ of active submatrix; revert to anchor) | **candidate** | `src/active_set_boost.cu` |
| cuPDLPx | Single-LP only; no batch API; no crossover; fp64 | **not applicable** as product limit (we batch) | memo §2.1; solver loop |
| cuOpt | Stable3 PDLP (Halpern/reflection path; major_iteration 200; restart_strategy 3) | **already in QENIVO** (Halpern path; our defaults align with cuPDLPx/Stable3 spirit) | `cpp/src/pdlp/solve.cu` `set_Stable3` |
| cuOpt | Batch PDLP: shared A, per-climber c/bounds, SpMM, swap/compact, tol 1e-4 forced | **already in QENIVO** (persistent batch + compact) | `solve.cu` `apply_batch_settings_overrides`, `run_batch_pdlp_*`; `pdlp.cu` |
| cuOpt | Timed SpMM batch-size search (start 128, ±2, ≤5 steps) | **candidate** | `pdlp/optimal_batch_size_handler/optimal_batch_size_handler.cu` |
| cuOpt | Reject FP32/mixed SpMV in batch mode | **candidate** (gap we can fill with certified MP on the *basis* engine, not batch PDHG) | `solve.cu` ~907–925 |
| cuOpt | Crossover: dual_push → primal_push → dual/primal phase-2 cleanup | **already in QENIVO** (PDHG → native simplex crossover) | `cpp/src/dual_simplex/crossover.cpp` ~334, ~743, ~1379+ |
| HiGHS | Dual simplex: DSE / Devex, cost perturbation, phase control | **already in QENIVO** (native dual simplex) | `highs/simplex/HEkkDual.cpp` |
| HiGHS | Postoptimal ranging (cost / bound / row) from optimal simplex basis | **already in QENIVO** | `highs/lp_data/HighsRanging.{h,cpp}`; `workload/ranging.py` |
| HiGHS | Parametric breakpoint *walk* (Gass–Saaty sweep API) | **candidate** | ranging is local intervals; full sweep not in HiGHS ranging module |
| SoPlex | Bound-flipping dual ratio test | **already in QENIVO** / parity item | `src/soplex/spxboundflippingrt.h`; native simplex |
| SoPlex | Dedicated Highs-style ranging API | **not applicable** (no analogue found; use basis formulas / CLP / HiGHS) | tree search 2026-10-01 |
| SoPlex | Rational solve: iterative refinement + ratrecon + precision boosting (Boost) | **already in QENIVO** (own limbs; Gleixner–Steffy–Wolter IR) | `src/soplex/solverational.hpp`; `engines/exact_lp.py` |
| CLP | `dualRanging` / `primalRanging` | **already in QENIVO** | `src/ClpSimplexOther.{hpp,cpp}` |
| CLP | `parametrics` / `parametricsLoop` (θ-homotopy on bounds/obj) | **candidate** (W02 breakpoint walk; EPL read-only) | `ClpSimplexOther::parametrics` ~2541+ |
| SCIP | Exact solving mode (`SCIPenableExactSolving`, `lpiexact`, exact LP + ExactSol cons) | **already in QENIVO** for LP (`exact-lp`); MILP exact tree **candidate** | `src/scip/scip_exact.h`, `lpexact.h`, `cons_exactsol.h` |

---

## 1. cuPDLPx (Apache-2.0)

**Core algorithm.** Restarted **Halpern** PDHG with **reflection**. Step size is constant
`η = 0.998 / σ_max(A)` from a power / singular-value estimator (`initialize_step_size_and_primal_weight`
in `src/solver.cu`). Primal weight ω is PID-updated on restart from log dual/primal distance error
(same constants QENIVO exposes as `k_p`, `k_i`, `i_smooth`). Verified against memo §2.1 and
`docs/algorithm/restart.md`.

**Restarts.** Fixed-point residual `r = ‖z − T(z)‖_P` (not PDLP's normalised duality gap). Fire when
any of: sufficient reduction (`β=0.2`), necessary reduction (`β=0.5`) plus local increase vs previous
check, or artificial epoch length `k ≥ 0.36 N`. On restart: new anchor = latest PDHG point, reset
local Halpern counter, update primal weight.

**Data structures / kernels.** Both `A` and `Aᵀ` stored CSR for non-transpose SpMV; fused kernels
combine affine update, projection, reflection, and Halpern anchoring (`docs/implementation/kernel-fusion.md`).
`src/solver.cu` captures a **CUDA Graph** for iterations `2 … F−1` of each
`termination_evaluation_frequency` window and relaunches it across restarts while device buffers keep
stable addresses.

**Active-set boost (post-paper).** `src/active_set_boost.cu` tracks variables confidently at bounds
and inactive rows over a sliding window, re-estimates σ on the active submatrix, raises the step
toward `safety/σ_active`, and reverts to the stored anchor (including primal-weight PID state) on
divergence. **candidate** for QENIVO single-LP and mid-batch polish; not required for the stack
story.

**QENIVO map.** Defaults in `engines/pdhg.py` (`sufficient=0.2`, `necessary=0.5`, `artificial=0.36`,
`step_safety=0.998`, PID gains) match this engineering form. Batching and crossover are QENIVO
additions; cuPDLPx has neither.

---

## 2. NVIDIA cuOpt LP / PDLP / batch (Apache-2.0)

**PDLP modes.** `cpp/src/pdlp/solve.cu` defines Stable1/2/3 and Methodical presets. **Stable3**
(default for batch overrides) turns on reflected PDHG, fixed-point error, conditional major
iterations (`major_iteration = 200`), `restart_strategy = 3`, `never_restart_to_average = true`,
Ruiz + Pock–Chambolle scaling, and **disables** adaptive step size in favour of
`initial_step_size_max_singular_value` — explicitly documented as the cuPDLPx(+) mapping.

**Batch PDLP.** `run_batch_pdlp_fixed` / splitting path accept per-climber objectives, constraint
bounds, objective offsets, and `new_bounds` for variable bounds. `pdlp.cu` allocates
**per-climber** `primal_step_size_`, `dual_step_size_`, `primal_weight_`. Converged climbers are
compacted via `swap_and_resize_helper.cuh`. Products use cuSPARSE **SpMM** (`CUSPARSE_SPMM_CSR_ALG2`,
or `ALG3` when deterministic).

**Forced batch settings** (`apply_batch_settings_overrides`): method PDLP, presolve off,
Stable3, **infeasibility detection off**, iteration limit 100k (unless user-set), all primal/dual/gap
tolerances defaulted to **1e-4**, `inside_mip = true`. Mixed and single precision are **rejected**
in batch (`solve.cu` validation messages match memo §2.3).

**Batch-size search.** `optimal_batch_size_handler.cu` times the two SpMMs (A and Aᵀ) five times,
minimises time/batch_size, starts at **128** (floor power-of-two), then halves or doubles for at
most a few steps. Verifies memo claim; minor detail: code comments say “at max we take 5 steps”
with `max_steps = 4` after the initial direction probe.

**Crossover.** `dual_simplex/crossover.cpp`: `dual_push` (~334) then `primal_push` (~743), then
dual phase-2 and optional primal phase-2 cleanup (~1379+). Classic Megiddo/Bixby-style **one LP at
a time** on CPU. QENIVO's `engines/crossover.py` is the same *role* (FO point → vertex) with a
reduce/crash/native-simplex design aimed at **streaming a stack**.

**QENIVO map.** Batch + 1e-4 FO tier + per-case crossover already exist. Timed SpMM batch sizing and
mixed-precision *basis* work remain **candidate**. Concurrent PDLP/barrier/dual race is **not
applicable** as our differentiator (we stream cases, not race engines on one LP).

---

## 3. HiGHS dual simplex + ranging (MIT)

**Dual simplex.** `highs/simplex/HEkkDual.cpp` initialises duals, optionally **perturbs costs** unless
near-optimal, chooses **dual steepest edge** weights when available else **Devex**, and manages
phase-1/phase-2 forcing when the unperturbed basis is dual-infeasible. This is the fair CPU
comparator for warm-started what-if stacks (memo §2.5, §3.2).

**Ranging.** `HighsRanging` records col cost up/down, col bound up/down, row bound up/down
(`HighsRanging.h`). `getRangingData` requires an optimal solution and a live simplex instance, then
builds “delta” and “theta” spaces and a major theta loop over tableau alphas
(`HighsRanging.cpp`) — classical postoptimal intervals from the optimal basis, not a full parametric
homotopy walker.

**QENIVO map.** `workload/ranging.py` already implements cost and RHS ranging from the optimal
basis (**already in QENIVO**). A CLP-style `parametrics` breakpoint walk along a planner price path
remains **candidate** for W02 (Agent A §1.1 actionable).

---

## 4. SoPlex (Apache-2.0)

**What is present.** Dual **bound-flipping** ratio test (`spxboundflippingrt.h`; solver fields
`boundflips`, multi-RHS `solveVector3` reserved for flips). Rational / exact path
`_optimizeRational` in `solverational.hpp`: stores real LP rim data, runs **iterative refinement**
with continued-fraction reconstruction (`ratrecon`), optional **precision boosting** (Boost
multiprecision). This is the Gleixner–Steffy–Wolter lineage used inside SCIP exact mode.

**What is absent.** No `HighsRanging`-style or CLP `dualRanging` module showed up in the SoPlex tree
(search for rang/sensi/paramet hit only bound-flipping and unrelated `fmt/ranges.h`). Postoptimal
*ranging* for planners should be attributed to HiGHS/CLP/QENIVO basis formulas, not SoPlex.

**QENIVO map.** Bound flipping and exact IR are **already in QENIVO**. Copying SoPlex rational LU is
**not applicable** (R2; we have own limbs in `native/exact/`).

---

## 5. CLP `ClpSimplexOther::parametrics` (EPL-2.0 — read only)

**Ranging.** `dualRanging` / `primalRanging` (+ CBC variant) compute one-sided postoptimal ranges
via ratio checks on dual/primal directions (`ClpSimplexOther.hpp` ~27–69; `.cpp` implementations).

**Parametrics.** `parametrics(startingTheta, endingTheta, reportIncrement, lower/upper bound
changes, RHS changes, objective change)` applies `current + θ · Δ`, solves dual guts, then loops
`parametricsLoop` reporting at increments until infeasible/unbounded or `endingTheta`
(`.cpp` ~2541–2700). Separately: `parametricsObj`, file-driven parametrics, and helpers
`nextTheta` / `whileIterating`. This is the open-source reference for a **Gass–Saaty-style walk**
with event reporting — closer to SANKHYA's 1-D sweep than to HiGHS's static ranging dump.

**Licence.** EPL-2.0: **read for ideas only**; do not copy code into Apache-2.0 QENIVO.

**QENIVO map.** Static ranging **already in QENIVO**. Full `parametrics` loop = **candidate** for
W02 (reimplement from papers, not from CLP sources).

---

## 6. SCIP exact mode (Apache-2.0)

**API.** `SCIPenableExactSolving` / `SCIPisExact` in `scip_exact.h` (must be set before problem
creation; incompatible with reoptimization; needs `SCIP_WITH_EXACTSOLVE`). Exact LP layer
`lpexact.h` stores columns/rows with `SCIP_RATIONAL` coefficients alongside floating LP. Constraint
handler `cons_exactsol` ensures the primal solution is exact. Exact branching `SCIPbranchLPExact`
and exact cut store complete the MIP path. LP engine backends live under `src/lpiexact/`
(SoPlex / QSopt_ex style).

**QENIVO map.** Exact *LP* vertex + certificate is **already in QENIVO** (`engines/exact_lp.py`,
`certify/`). Full SCIP-style exact *MILP* tree is **candidate** only if W07 needs proven integer
gaps; not required for the LP what-if stack.

---

## Cross-check vs `03_frontier_and_white_space.md`

| Memo claim | Code verdict |
|---|---|
| cuPDLPx Halpern + reflection, η=0.998/‖A‖₂, PID ω, three FPE restart rules | **Confirmed** (`restart.md`, `solver.cu`) |
| CUDA Graphs over termination windows | **Confirmed** (`solver.cu` graph capture/launch) |
| Active-set boost post-paper | **Confirmed** (`active_set_boost.cu`) |
| cuOpt batch: SpMM, per-LP steps, compact, tol 1e-4, Stable3 | **Confirmed** (`solve.cu`, `pdlp.cu`) |
| Mixed precision forbidden in batch | **Confirmed** (expects in `run_pdlp`) |
| Batch size timed search from 128 | **Confirmed** (`optimal_batch_size_handler.cu`) |
| Crossover dual then primal push | **Confirmed** (`crossover.cpp` 334 / 743) |
| HiGHS ranging module | **Confirmed** (`HighsRanging.*`); not a full parametric walker |
| cuPDLPx licence Apache-2.0 (not MIT) | **Confirmed** (`LICENSE`); QENIVO `NOTICE` correct; `pdhg.py` header still says “MIT licence” (doc bug only) |

---

## Paths read (this survey)

Fetched into a local disposable cache (not committed):

* cuPDLPx: `LICENSE`, `src/solver.cu`, `src/active_set_boost.cu`, `docs/algorithm/restart.md`, `docs/implementation/kernel-fusion.md`
* cuOpt: `LICENSE`, `cpp/src/pdlp/solve.cu`, `pdlp.cu`, `optimal_batch_size_handler/optimal_batch_size_handler.cu`, `swap_and_resize_helper.cuh`, `dual_simplex/crossover.cpp`
* HiGHS: `LICENSE.txt`, `highs/simplex/HEkkDual.cpp`, `highs/lp_data/HighsRanging.{h,cpp}`
* SoPlex: `LICENSE`, `src/soplex/spxsolver.h`, `spxboundflippingrt.h`, `solverational.hpp`, `spxscaler.h`
* CLP: `LICENSE`, `src/ClpSimplexOther.{hpp,cpp}`
* SCIP: `LICENSE`, `src/scip/scip_exact.{h,c}`, `lpexact.h`, `cons_exactsol.h`
* QENIVO (in-tree): `src/qenivo/engines/{pdhg,crossover,exact_lp}.py`, `workload/ranging.py`, `NOTICE`

No local clones of the six upstreams were found under Desktop/SIH 119; all rival reads were via
GitHub raw/API.
