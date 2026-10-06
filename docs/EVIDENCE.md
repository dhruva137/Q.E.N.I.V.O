# Evidence, requirement by requirement (PS 26119)

Status as of 30 Sep 2026. Every number points to a result file or a test; failures stay in the
files. **Done** = in the code with tests and a measured result. **Pending run** = the code is done
and the measurement needs a GPU or a long run (the notebook part that produces it is named).
**Gap** = not yet built; it is on the roadmap.

Machine for the CPU numbers: Windows 11 laptop, 16 logical cores, RTX 5050 Laptop GPU (8 GB).

| ID | Requirement (PS wording) | Status | Evidence |
|---|---|---|---|
| R1 | "shall not be built upon any existing open source solver library … from mathematical foundation" | **Done** | No solver library in the solve path; tripwires on every SciPy sparse direct solver and optimiser (`provenance.py`, checked by the tests and in every benchmark row as `provenance_clean`); offline bundle = qenivo + NumPy + SciPy with SBOM |
| R2 | a solver core, not a modelling environment | **Done** | Seven engines under one `Problem` type; interfaces are thin (`api.py`, `cli.py`) |
| R3 | revised simplex and interior point | **Done** | `engines/simplex.py` (bounded primal/dual, product-form updates, Harris tests, perturbation, warm start), `engines/ipm.py` (Mehrotra); both in the Netlib run below |
| R4 | branch-and-bound, branch-and-cut, cutting planes, presolve, heuristics, node selection | **Done** | `engines/milp.py`, `presolve.py`, `cuts.py`: presolve with postsolve, root GMI and cover cuts, reliability branching, best-bound/dive hybrid, rounding and diving; MIPLIB below |
| R5 | QP | **Done** | `engines/pdqp.py` + native IPM (`engines/native_ipm.py`); Maros-Meszaros on Kaggle T4: 96/135 verified, 94 agree with the reference (HiGHS 100) (`bench/results/kaggle_t4_run1/PARITY.md`) |
| R6 | modular, extensible to MIQP, NLP, MINLP | **Done** | `engines/registry.py` (`register_engine`, entry-point group `qenivo.engines`); MIQP shipped as a plugin (`engines/miqp.py`), matches enumeration on a cardinality portfolio (`tests/test_extensions.py`) |
| R7 | sparse matrix techniques, efficient numerical linear algebra | **Done** | Sparse CSR everywhere; own CUDA SpMV; native C++ sparse LU with threshold/Markowitz pivoting and eta updates (`native/simplex_core.cpp`); own supernodal sparse Cholesky for the IPM (`native/ipm/`); native simplex median 1.7x HiGHS time on 14 hard Netlib LPs (`bench/results/native_simplex_speed_round2.jsonl`) |
| R8 | multi-core parallelization | **Done** | `workload/parallel.py`: case pool 3.2x on 8 cores, 32/32 cases certified (`bench/results/multicore.json`, BLAS pinned to one thread per process); verifier-gated engine race (`--engine race`) |
| R9 | GPU acceleration where it provides measurable benefit | **Done** | Certified what-if batches vs HiGHS on all cores: A100 512 cases 56x (`a100_evidence_run1`), 2x T4 256 cases 43x (`kaggle_t4_run1/T4.md`), RTX 5050 laptop 64 cases 28x (`persistent_batch_rtx5050.jsonl`); persistent kernel 493 -> 43 us/iteration on qap15; GPU is slower on small single LPs and the router does not use it there |
| R10 | numerical stability, reliable convergence | **Done** | Every verdict re-checked in float64 on the original model; robustness ladder (simplex → IPM → PDHG); infeasibility and unboundedness proven by checked rays; float32 answers refined to 1e-8+ and checked |
| R11 | thousands to millions of variables; degenerate, ill-conditioned, difficult MILP | **Done** | Million-scale LPs solved and certified on the A100 (savsched1 1.77M nonzeros, pds-100 1.09M, rmine15); degenerate Netlib family solved; hard MILP exhibit: crude scheduling 1.2% gap at 120 s |
| R12 | API or CLI | **Done** | `pip install`, `qenivo solve model.mps`, Python `qenivo.solve()`, JSON API (`qenivo serve`) |
| R13 | MIPLIB, Netlib or Mittelmann vs an established solver | **Done** | Netlib 97/98 verified (`netlib_release_20260930.json`), MIPLIB 2017 subset, Kennington, Maros-Meszaros, Mittelmann large LPs, all against HiGHS under the same limits |
| R14 | robustness demonstration (degeneracy, weak relaxations, ill-conditioning) | **Done** | Netlib infeasible 25/29 with Farkas proofs, none wrong; PDHG design ablation (`kaggle_t4_run1/ARCHITECTURE_DECISIONS.md`) |
| R15 | transparent, extensible, sovereign foundation | **Done** | Apache-2.0; JSON certificates any third party can re-check (`qenivo verify`, exact rational mode); `docs/ARCHITECTURE.md` "Built to grow" |
| R16 | refinery scheduling, crude blending, process optimisation, production planning, logistics, power dispatch, transportation, supply chain | **Done** | `models/industry.py` + `industry_ext/` + `refinery_full/`: 17 sector models, 7 solved to proven optimality matching HiGHS to ~1e-9 (crude scheduling is the hard MILP above); refinery planning LP, Williams refinery, Haverly pooling, the open refinery benchmark (GAMS reader) |
| R17 | datasets: MIPLIB, Netlib, Mittelmann, QPLIB/Maros, literature case studies | **Done** | All fetched from official sources by `bench/fetch.py` / `bench/miplib_run.py`; the MIPLIB reference values are the official `miplib2017-v33.solu` |

## Netlib LP (98 models), CPU, 120 s per model, 1e-6, independent verification

`bench/results/netlib_20260930.json` (auto routing, CPU only):

| | Result |
|---|---|
| verified optimal | **92 / 98** (previous run, before today's simplex work: 85) |
| objective within 1e-6 relative of HiGHS | 80 (a further 7 differ by 1e-6 to 3e-6, consistent with the 1e-6 tolerance; 5 have no HiGHS reference within the limit) |
| not finished within the 240 s wall | 80bau3b, greenbea, greenbeb, pilot, pilot.ja |

## Netlib infeasible (29 models)

`bench/results/netlib_infeas_20260930.json`: **25 of the 29 models proven infeasible**, each with a Farkas
certificate that the independent verifier accepted (previous run: 22). No model was given a wrong
verdict. Not proven within the limits: cplex2, gosh, pang (verdict withheld), gran (240 s wall).
Note: the run's greenbea row used the *feasible* Netlib model (a name clash in the set list, now
fixed as `greenbea-infeas`); the infeasible greenbea was then solved separately: infeasible, Farkas
certificate valid.

## MIPLIB 2017 subset (15 classic models), 120 s, one thread each, against the official optima

`bench/results/miplib.jsonl`:

| | Ours | HiGHS |
|---|---|---|
| proven optimal | **4 / 15** (p0201 17.7 s, flugpl 19.6 s, dcmulti 33.7 s, khb05250 2.8 s; `miplib_20260930b.jsonl`) | 9 / 15 |
| within 3% of the optimum at the limit | 7 more (gen-ip002 0.8%, noswot 2.4%, mas76 3.0%, …) | |

The gap to HiGHS is the MIR/flow-cover cut families and native speed (roadmap M2 and M1).

## Sector models (R16), CPU

| Model | Class | Result | HiGHS |
|---|---|---|---|
| economic dispatch (6 units x 12 h) | QP | optimal, KKT certified | (QP, not compared) |
| unit commitment (5 x 12) | MILP | optimal 383,342.26 | 383,342.26 |
| transportation (10 x 30) | LP | optimal 77,790.99 | 77,790.99 |
| facility location (8 x 25) | MILP | optimal 2,668.62 | 2,668.62 |
| lot sizing (4 items x 8) | MILP | optimal 12,060.84 | 12,060.84 |
| crude blending | LP | optimal 17,943.38 | 17,943.38 |
| crude scheduling (3 tanks x 3 crudes x 8) | MILP | 543.88, 1.2% gap at 120 s | 537.36 |
| multi-commodity flow (15 nodes x 4) | LP | optimal 904.67 | 904.67 |

## Release build, Netlib (laptop CPU, 120 s, 1e-6)

`bench/results/netlib_release_20260930.json` (all branches merged, round-2 native simplex): **97 of 98**
verified optimal and within 1e-6 of HiGHS in one run; dfl001 is verified optimal by our certificate but its
objective differs from HiGHS by more than 1e-6 relative, so it is not counted. On the 93 models both this
run and the previous one solved, total solve time fell from 1,046 s to 710 s.

## Native C++ simplex speed, round 2 (laptop CPU)

`bench/results/native_simplex_speed_round2.jsonl`, `bench/native_speed.py`, one repetition, HiGHS simplex on the
same machine as the reference. Every answer is KKT-checked on the original model (max 1.5e-8, greenbea).

| model | rows | HiGHS (s) | before (s) | after (s) | after / HiGHS |
|---|---|---|---|---|---|
| 25fv47 | 821 | 0.294 | 1.149 | 0.376 | 1.3 |
| bnl2 | 2,324 | 0.117 | 3.784 | 0.280 | 2.4 |
| scfxm3 | 990 | 0.055 | 0.341 | 0.079 | 1.4 |
| ship08l | 778 | 0.019 | 0.135 | 0.019 | 1.0 |
| degen2 | 444 | 0.025 | 0.106 | 0.029 | 1.1 |
| sctap3 | 1,480 | 0.016 | 0.414 | 0.034 | 2.1 |
| pilot4 | 410 | 0.053 | 0.345 | 0.067 | 1.3 |
| bnl1 | 643 | 0.041 | 0.289 | 0.074 | 1.8 |
| czprob | 929 | 0.097 | 0.417 | 0.130 | 1.3 |
| d6cube | 415 | 0.110 | 0.391 | 0.223 | 2.0 |
| pilot87 | 2,030 | 8.256 | 44.03 | 44.57 | 5.4 |
| greenbea | 2,392 | 1.052 | 12.33 | 1.714 | 1.6 |
| 80bau3b | 2,262 | 0.215 | 4.018 | 0.378 | 1.8 |
| pilot.ja | 940 | 0.158 | 0.605 (numerical error) | 0.546 | 3.5 |

Median time relative to HiGHS: **1.70x** after, 6.36x before (dual simplex with dual steepest edge,
bound-flipping ratio test, dual phase 1, incremental updates, row-wise PRICE).

## Kaggle, 2x Tesla T4, 4 CPU cores (`bench/results/kaggle_t4_run1/`)

Run on 30 Sep 2026 with the code of that morning (the Python simplex path; the native core was not
loaded there). Full tables: `T4.md`, `PARITY.md`, `WORLD.md`, `ARCHITECTURE_DECISIONS.md` in that folder.

* **What-if batches, 1e-4, every case certified**: refinery L4 x 256 cases, 1 GPU 36.8 s, 2 GPUs 22.3 s;
  HiGHS on all 4 cores 974.5 s (**43.7x** with 2 GPUs). L4 x 64: 10.0 s vs 245.3 s (24.5x, 1 GPU).
* **Coverage, independently verified**: Netlib 89/98 (80 agree with the reference; HiGHS 93),
  Netlib infeasible 23/29 with Farkas proofs, Kennington 14/16, Maros-Meszaros QP 96/135 verified,
  94 agree with the reference (HiGHS 100).
* **MIPLIB 2017 subset, 120 s**: 4/10 proven optimal (HiGHS 6/10).
* **Same machine, 1e-6**: exact vertex engines (HiGHS simplex/IPM, GLOP, SCIP) are faster on single
  mid-size Netlib LPs; QENIVO beats the GPU first-order comparators on d2q06c (9.6 s vs HiGHS PDLP 185 s,
  cuPDLPx 112.8 s reduced) and loses on 80bau3b (83 s vs 0.2 s for HiGHS simplex).

## Reproduce

```
python bench/run.py --set netlib --backend numpy --time-limit 120 --jobs 4 --out-file netlib.json
python bench/miplib_run.py --time-limit 120
python bench/multicore.py --level 2 --cases 32
python -m pytest                                 # 90+ tests, GPU tests run when CUDA is present
```
GPU: `notebooks/a100_evidence.ipynb`, `notebooks/a100_architecture.ipynb`,
`notebooks/kaggle_t4.ipynb` (instructions in `EPOCHZERO_KAGGLE_RUN.md`).
