# Roadmap

The foundation is in place: seven engines (simplex, IPM, PDHG, float32-refined PDHG, PDQP,
branch and cut, the MIQP plugin), certificates with an independent verifier, the planner workload
layer, a sector model library, multi-core execution, packaging, access control and an audit trail.
What follows is ordered by impact on the problem statement, with a measurable exit test for each
milestone. Items marked **done** are in the code with tests.

## M1: Linear algebra at production scale (the biggest single lever)

| Item | Why | Exit test |
|---|---|---|
| Native sparse LU with Markowitz pivoting and Forrest-Tomlin updates (C++ core, Python bindings) | the simplex engine is exact but refactorises densely; this makes it scale to 10^5 rows | full Netlib + Kennington through `simplex`, all verified |
| Native supernodal sparse Cholesky with AMD ordering (CPU), cuDSS-free GPU path | the IPM factorises densely up to 6,000 rows today | Mittelmann LP set through `ipm` at 1e-8 |
| Crossover (IPM / PDHG point to an optimal basis) | vertex answers and exact ranging for large models | ranging on every Netlib model |
| Iterative refinement + rational re-check of the final basis | "proven optimal", not "optimal to 1e-9" | Netlib optima reproduced exactly (Koch's rational values) |
| **done:** product-form basis updates in the simplex (was one dense LU per pivot) | node LPs of branch and cut | 10-20x more nodes per second on MIPLIB instances |
| **done:** float32 PDHG with float64 iterative refinement (`pdhg-rf`) | FP64-grade answers from FP32 hardware | Netlib published optima to 1e-7 from float32 iterations (tests) |

## M2: MILP that closes gaps

| Item | Exit test |
|---|---|
| **done:** presolve, root Gomory mixed-integer and knapsack cover cuts | root bound moved on gt2 13,460 -> 17,184 and p0201 6,875 -> 7,125 |
| MIR and flow cover cuts, local cuts in the tree | MIPLIB subset: from 4/15 to HiGHS's 9/15 proven at 120 s, one thread |
| **done:** reliability branching; conflict analysis and restarts to do | nodes to optimality vs the current engine |
| **done:** multi-core case pool and verifier-gated engine race; parallel tree search to do | speed-up per core on MIPLIB |
| GPU Feasibility Jump and fix-and-propagate heuristics | time to first feasible on MIPLIB |
| VIPR-format MILP certificates | certificates accepted by the independent VIPR checker |

## M3: The refinery wedge

| Item | Exit test |
|---|---|
| Pattern-shared batched kernels (one sparsity pattern, S coefficient vectors) so distributive recursion for S cases runs in one GPU batch | 128 recursions in one batch, each converged and certified |
| Spatial branch and bound over the McCormick relaxation (global pooling) | Haverly 1/2/3 closed to the global optimum with a proof |
| Industrial bound derivation (unit capacities, tank limits) so the open refinery benchmark gets a certified bound | a certified bound for cases 1-3 of the Du et al. benchmark |
| Shadow mode: import the matrix the incumbent planning tool writes, solve side by side, report plan differences with certificates | a month of MRPL cases solved in shadow with every difference explained |
| Multi-period crude scheduling MILP (Lee et al. 1996 family) as a first-class workload (**model in** `models/industry.py`; closing it fast needs M2) | published example problems solved to optimality |

## M4: Platform

| Item | Exit test |
|---|---|
| Multi-GPU and multi-node first-order LP | a 10^8-nonzero LP solved across GPUs |
| AMPL `.nl`, LP-format and Pyomo / PuLP / JuMP plugins (written in this repository) | the same model solved from each front end |
| **done:** Excel / CSV case tables, IIS / Kerberos sign-in through a trusted proxy, 180-day audit trail | `docs/DEPLOYMENT.md` |
| Signed releases, reproducible builds, SBOM per release | third-party rebuild gives identical wheels |
| Planner console: case library, run history, role-based access (audit log **done**) | acceptance test with MRPL planners |
