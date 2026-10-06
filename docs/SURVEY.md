# Survey: how the world's LP/MILP engines are built, and where this project goes past them

Snapshot 30 Sep 2026. Sources at the end. Every claim about this project cites a file in `bench/results/`.

## 1. How the field evolved

| Era | Architecture | Who |
|---|---|---|
| 1947-1990s | Revised simplex with LU updates (product form, then Forrest-Tomlin), steepest-edge / Devex pricing, presolve | CPLEX, Xpress, later Gurobi; open: GLPK, CLP |
| 1984-2010s | Interior-point (Mehrotra predictor-corrector) with sparse Cholesky and **crossover** to a vertex | every commercial solver; HiGHS IPX |
| 2000s-2020s | MILP branch and cut: presolve, Gomory / MIR / flow-cover cuts, reliability branching, conflict analysis, heuristics, parallel tree search | CPLEX, Gurobi, Xpress, SCIP, HiGHS; China: COPT, MindOpt, OptVerse |
| 2021-2024 | **First-order LP on GPUs**: PDLP (Applegate, Lubin, Hinder et al., Google), cuPDLP.jl / cuPDLP-C (Lu & Yang, COPT), then restarted Halpern PDHG with reflection (cuPDLPx, MIT group; Apache-2.0) | Google OR-Tools, COPT, HiGHS, MIT |
| 2025 | **Hybrid and concurrent**: GPU PDHG + GPU barrier (cuDSS) + CPU dual simplex raced (NVIDIA cuOpt); Gurobi 13 ships PDHG on CPU and GPU with **concurrent crossover** threads launched from intermediate PDHG iterates (1e-2..1e-5); COPT adds GPU barrier and PDLP with crossover | NVIDIA, Gurobi, COPT |
| 2025-2026 research | Crossover-aware PDHG ("Backing PDHG into a Corner"), spiral-dynamics crossover, presolve for GPU first-order methods (PSLP), GPU primal heuristics (Feasibility Jump, fix-and-propagate), GPU-batched branch and bound (B^3-PWL), batched small LPs (BatchPDLP.jl), multi-GPU / cluster PDLP (D-PDLP, ShardLP) | Gurobi, NVIDIA, MIT, academia |

**The lesson of 2025:** nobody serious drives a first-order method to 1e-8 any more. The GPU gets a
moderate-accuracy point fast; an exact engine (simplex crossover) finishes. Accuracy and speed are
split between two kinds of hardware.

## 2. Our own measured failure patterns (what the logs say)

1. **Iteration explosion at tight tolerance.** At 1e-6 our PDHG needs 473,792 iterations on Linf_520c
   (cuPDLPx 12,800); `bench/results/a100_evidence_run1/e2_large_lp.jsonl`.
2. **Per-iteration overhead, not arithmetic.** 270-3,000 us per iteration against cuPDLPx's 27-114 us on
   the same A100 (same file); the persistent cooperative kernel removes most of it on mid-size models
   (qap15 493 -> 43 us/iteration on an RTX 5050).
3. **Degeneracy in the simplex.** Dantzig pricing stalled on bnl2 (133 s); Devex pricing: 4.2 s.
4. **MILP is node-LP bound.** Python node LPs capped tree throughput; native node LPs and warm starts
   are 12-134x faster (`bench/results/native_simplex_netlib.jsonl`).
5. **Nonlinear refinery models** need a feasible start more than a faster LP (T4 branch findings).

Patterns 1, 2 and 4 have one root: we asked one method to do two jobs, **be fast** and **be exact**.

## 3. The architecture: a certified two-tier engine for scenario stacks

```
 what-if stack (S cases, one matrix)
        |
  Tier 1  GPU  - persistent cooperative PDHG kernel, all S cases in one launch, to 1e-3..1e-4
        |           (fast regime; no iteration explosion; no per-iteration host round trips)
        v   stream each case as soon as it crosses the crossover threshold
  Tier 2  CPU cores - native C++ simplex crossover per case, warm-started from
        |           (a) the basis guessed from the GPU point, and
        |           (b) the exact basis of an already finished NEIGHBOUR case: price cases share
        |               the constraints, so a neighbour's optimal basis is primal feasible for all
        v
  Tier 3  proof - exact vertex, exact duals (shadow prices), exact ranging, certificate,
                  independent verifier
```

What exists elsewhere: concurrent crossover for **one** LP (Gurobi 13), racing engines (cuOpt), GPU
batching of **small** LPs (BatchPDLP.jl), and CPU warm starts of what-if cases from a base-case basis
(common practice in planning tools, and in other PS 26119 entries). What we did not find elsewhere: GPU
batching of refinery-sized what-if stacks whose points are streamed into exact per-case crossover, warm-started
from a neighbouring case's basis, with a checkable certificate per case. It is also the planner's actual need: exact marginal values per case,
not 1e-6 approximations.

## 4. Status

* Tier 1: done (`kernels/persistent.py`, `engines/pdhg.py`); A100 56x vs HiGHS on 12 cores at 1e-4.
* Tier 2: `engines/crossover.py` (basis guess from a point + native simplex); measurements in
  `bench/results/crossover_rtx5050.jsonl`; cross-case basis sharing next.
* Tier 3: done (`certify/`), plus exact ranging (`workload/ranging.py`).

## Sources

* Rothberg, *Concurrent Crossover for PDHG*, arXiv 2510.24429 (Gurobi, 2025)
* *Backing PDHG into a Corner*, arXiv 2511.13894 (2025)
* *A New Crossover Algorithm for LP Inspired by the Spiral Dynamic of PDHG*, arXiv 2409.14715
* Lu, Peng, Yang, *cuPDLPx*, arXiv 2507.14051; Lu & Yang, cuPDLP.jl, arXiv 2311.12180
* *An Overview of GPU-based First-Order Methods for LP and Extensions*, arXiv 2506.02174
* *Presolving for GPU-Accelerated First-Order LP Solvers* (PSLP), arXiv 2604.23951
* Cordduk et al., *GPU-Accelerated Primal Heuristics for MIP*, arXiv 2510.20499 (NeurIPS 2025)
* *B^3-PWL: GPU-Batched Branch-and-Bound*, arXiv 2608.28988
* NVIDIA cuOpt documentation, LP features (concurrent PDLP / barrier / dual simplex, crossover)
* Gurobi blogs: first GPU-accelerated solver (PDHG beta); GPUs for LPs vs MIPs; Gurobi 13 release notes
* COPT user guide, arXiv 2208.14314; GAMS blog on cuOpt (Sep 2025); BatchPDLP.jl (PSORLab)
