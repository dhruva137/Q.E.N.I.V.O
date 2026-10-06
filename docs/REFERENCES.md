# References

QENIVO is an independent implementation of the methods below. No source code from any of the cited
projects is included. Each module's header names the papers it follows.

## First-order methods (GPU tier)

| Component | Reference |
|---|---|
| PDHG for LP (PDLP) | D. Applegate, M. Díaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy. *Practical large-scale linear programming using primal-dual hybrid gradient.* NeurIPS 2021. [arXiv:2106.04756](https://arxiv.org/abs/2106.04756) |
| Infeasibility detection with PDHG | D. Applegate, M. Díaz, H. Lu, M. Lubin. *Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming.* SIAM J. Optim. 34(1), 2024. [arXiv:2102.04592](https://arxiv.org/abs/2102.04592) |
| GPU PDHG | H. Lu, J. Yang. *cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid gradient for linear programming in Julia.* 2023. [arXiv:2311.12180](https://arxiv.org/abs/2311.12180) |
| Restarted Halpern PDHG with reflection | H. Lu, J. Yang. *Restarted Halpern PDHG for linear programming.* 2024. [arXiv:2407.16144](https://arxiv.org/abs/2407.16144) |
| Engineering of the GPU solver | H. Lu, Z. Peng, J. Yang. *cuPDLPx: A further enhanced GPU-based first-order solver for linear programming.* 2025. [arXiv:2507.14051](https://arxiv.org/abs/2507.14051) |
| First-order convex QP (PDQP) | H. Lu, J. Yang. *A practical and optimal first-order method for large-scale convex quadratic programming.* 2023. [arXiv:2311.07710](https://arxiv.org/abs/2311.07710) |
| Crossover from a first-order point | E. Rothberg. *Concurrent crossover for PDHG.* 2025. [arXiv:2510.24429](https://arxiv.org/abs/2510.24429) |
| Survey | *An overview of GPU-based first-order methods for linear programming and extensions.* 2025. [arXiv:2506.02174](https://arxiv.org/abs/2506.02174) |

## Simplex and linear algebra (CPU tier)

| Component | Reference |
|---|---|
| Bounded revised simplex | I. Maros. *Computational Techniques of the Simplex Method.* Kluwer, 2003. [doi:10.1007/978-1-4615-0257-9](https://doi.org/10.1007/978-1-4615-0257-9) |
| Harris ratio test, Devex pricing | P. M. J. Harris. *Pivot selection methods of the Devex LP code.* Math. Programming 5, 1973. [doi:10.1007/BF01580108](https://doi.org/10.1007/BF01580108) |
| Steepest-edge pricing | J. J. Forrest, D. Goldfarb. *Steepest-edge simplex algorithms for linear programming.* Math. Programming 57, 1992. [doi:10.1007/BF01581089](https://doi.org/10.1007/BF01581089) |
| Dual simplex, bound flipping | A. Koberstein. *The dual simplex method, techniques for a fast and stable implementation.* PhD thesis, Paderborn, 2005. |
| Parallel dual simplex (the comparator HiGHS) | Q. Huangfu, J. A. J. Hall. *Parallelizing the dual revised simplex method.* Math. Prog. Comp. 10, 2018. [doi:10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5) |
| Interior point | S. Mehrotra. *On the implementation of a primal-dual interior point method.* SIAM J. Optim. 2(4), 1992. [doi:10.1137/0802028](https://doi.org/10.1137/0802028); S. J. Wright. *Primal-Dual Interior-Point Methods.* SIAM, 1997. |
| Minimum-degree ordering | P. R. Amestoy, T. A. Davis, I. S. Duff. *An approximate minimum degree ordering algorithm.* SIAM J. Matrix Anal. Appl. 17(4), 1996. [doi:10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952) |
| Exact LP by iterative refinement | A. M. Gleixner, D. E. Steffy, K. Wolter. *Iterative refinement for linear programming.* INFORMS J. Comput. 28(3), 2016. [doi:10.1287/ijoc.2016.0692](https://doi.org/10.1287/ijoc.2016.0692) |
| Parametric cost / RHS analysis | S. I. Gass, T. L. Saaty. *The computational algorithm for the parametric objective function.* Naval Research Logistics Quarterly 2(1–2), 1955. [doi:10.1002/nav.3800020106](https://doi.org/10.1002/nav.3800020106); T. Gal. *Postoptimal Analyses, Parametric Programming, and Related Topics.* 2nd ed., de Gruyter, 1995; R. J. Vanderbei. *Linear Programming: Foundations and Extensions*, ch. 7. |

## Mixed-integer programming

| Component | Reference |
|---|---|
| Reliability branching | T. Achterberg, T. Koch, A. Martin. *Branching rules revisited.* Oper. Res. Lett. 33(1), 2005. [doi:10.1016/j.orl.2004.04.002](https://doi.org/10.1016/j.orl.2004.04.002) |
| Gomory mixed-integer cuts | R. E. Gomory. *An algorithm for the mixed integer problem.* RAND RM-2597, 1960. |
| c-MIR cuts | H. Marchand, L. A. Wolsey. *Aggregation and mixed integer rounding to solve MIPs.* Oper. Res. 49(3), 2001. [doi:10.1287/opre.49.3.363.11211](https://doi.org/10.1287/opre.49.3.363.11211) |
| Feasibility Jump | B. Luteberget, G. Sartor. *Feasibility Jump: an LP-free Lagrangian MIP heuristic.* Math. Prog. Comp. 15, 2023. [doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8) |
| GPU primal heuristics | *GPU-accelerated primal heuristics for mixed integer programming.* NeurIPS 2025. [arXiv:2510.20499](https://arxiv.org/abs/2510.20499) |

## Nonlinear and pooling

| Component | Reference |
|---|---|
| Filter line-search IPM | A. Wächter, L. T. Biegler. *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming.* Math. Programming 106, 2006. [doi:10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y) |
| McCormick envelopes | G. P. McCormick. *Computability of global solutions to factorable nonconvex programs.* Math. Programming 10, 1976. [doi:10.1007/BF01580665](https://doi.org/10.1007/BF01580665) |
| Pooling and distributive recursion | C. A. Haverly. *Studies of the behavior of recursion for the pooling problem.* ACM SIGMAP Bulletin 25, 1978. [doi:10.1145/1111237.1111238](https://doi.org/10.1145/1111237.1111238) |

## Benchmarks

| Set | Reference |
|---|---|
| Netlib LP | D. M. Gay. *Electronic mail distribution of linear programming test problems.* Mathematical Programming Society COAL Newsletter 13, 1985. [netlib.org/lp](https://www.netlib.org/lp/) |
| MIPLIB 2017 | A. Gleixner et al. *MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library.* Math. Prog. Comp. 13, 2021. [doi:10.1007/s12532-020-00194-3](https://doi.org/10.1007/s12532-020-00194-3) |
| Maros-Mészáros QP | I. Maros, C. Mészáros. *A repository of convex quadratic programming problems.* Optim. Methods Softw. 11, 1999. [doi:10.1080/10556789908805768](https://doi.org/10.1080/10556789908805768) |
| Mittelmann LP | H. Mittelmann. *Benchmarks for optimization software.* [plato.asu.edu/bench.html](https://plato.asu.edu/bench.html) |
| Refinery planning benchmark | W. Du et al. *A production planning benchmark for real-world refinery-petrochemical complexes.* 2025. [arXiv:2503.22057](https://arxiv.org/abs/2503.22057) |

## Parametric LP, ranging, and sensitivity (W02 / W13)

| Component | Reference |
|---|---|
| Parametric objective simplex | S. Gass, T. Saaty. *The computational algorithm for the parametric objective function.* Naval Research Logistics Quarterly 2(1–2), 1955. [doi:10.1002/nav.3800020106](https://doi.org/10.1002/nav.3800020106) |
| Postoptimal / parametric monograph | T. Gal. *Postoptimal Analyses, Parametric Programming, and Related Topics.* De Gruyter, 1994/1995. [doi:10.1515/9783110871203](https://doi.org/10.1515/9783110871203) |
| Multiparametric LP | T. Gal, J. Nedoma. *Multiparametric linear programming.* Management Science 18(7), 1972. [doi:10.1287/mnsc.18.7.406](https://doi.org/10.1287/mnsc.18.7.406) |
| Geometric parametric LP | I. Adler, R. D. C. Monteiro. *A geometric view of parametric linear programming.* Algorithmica 8, 1992. [doi:10.1007/BF01758841](https://doi.org/10.1007/BF01758841) |
| Degenerate sensitivity warning | B. Jansen, J. J. de Jong, C. Roos, T. Terlaky. *Sensitivity analysis in linear programming: just be careful!* EJOR 101(1), 1997. [doi:10.1016/S0377-2217(96)00172-5](https://doi.org/10.1016/S0377-2217(96)00172-5) |
| Optimal partition postoptimal analysis | H. J. Greenberg. *The use of the optimal partition in a linear programming solution for postoptimal analysis.* Operations Research Letters 15(4), 1994. [doi:10.1016/0167-6377(94)90075-2](https://doi.org/10.1016/0167-6377(94)90075-2) |

## Refinery planning and crude selection (W13)

| Component | Reference |
|---|---|
| Planning under uncertainty | W. Li, C.-W. Hui, A. Li, P. Li. *Refinery planning under uncertainty.* Ind. Eng. Chem. Res. 43(21), 2004. [doi:10.1021/ie049737d](https://doi.org/10.1021/ie049737d) |
| Integrated CDU–FCC–blend planning | W. Li, C.-W. Hui, A. Li. *Integrating CDU, FCC and product blending models into refinery planning.* Computers & Chemical Engineering 29(9), 2005. [doi:10.1016/j.compchemeng.2005.05.010](https://doi.org/10.1016/j.compchemeng.2005.05.010) |
| Model reduction for refinery-wide opt. | T. Gueddar, V. Dua. *Disaggregation–aggregation based model reduction for refinery-wide optimization.* Computers & Chemical Engineering 35(12), 2011. [doi:10.1016/j.compchemeng.2011.04.016](https://doi.org/10.1016/j.compchemeng.2011.04.016) |
| Energy-oriented reduced models | T. Gueddar, V. Dua. *Novel model reduction techniques for refinery-wide energy optimisation.* Applied Energy 89(1), 2012. [doi:10.1016/j.apenergy.2011.05.056](https://doi.org/10.1016/j.apenergy.2011.05.056) |

## Crude scheduling MILP (W07 / W13)

| Component | Reference |
|---|---|
| Discrete-time unloading / inventory MILP | H. Lee, J. M. Pinto, I. E. Grossmann, S. Park. *Mixed-integer linear programming model for refinery short-term scheduling of crude oil unloading with inventory management.* Ind. Eng. Chem. Res. 35(5), 1996. [doi:10.1021/ie950519h](https://doi.org/10.1021/ie950519h) |
| Continuous-time crude operations | Z. Jia, M. Ierapetritou, J. D. Kelly. *Refinery short-term scheduling using continuous time formulation: crude-oil operations.* Ind. Eng. Chem. Res. 42(13), 2003. [doi:10.1021/ie020124f](https://doi.org/10.1021/ie020124f) |
| Industrial blend-scheduling benefits | J. D. Kelly, J. L. Mann. *Crude oil blend scheduling optimization: an application with multimillion dollar benefits.* Hydrocarbon Processing 82(6), 2003. (no Crossref DOI located; URL in `docs/LITERATURE.md`) |
| SBM/VLCC multi-CDU operations | P. C. P. Reddy, I. A. Karimi, R. Srinivasan. *Novel solution approach for optimizing crude oil operations.* AIChE Journal 50(6), 2004. [doi:10.1002/aic.10112](https://doi.org/10.1002/aic.10112) |
| RTN continuous-time crude blending | P. M. Castro, I. E. Grossmann. *Global optimal scheduling of crude oil blending operations with RTN continuous-time and multiparametric disaggregation.* Ind. Eng. Chem. Res. 53(39), 2014. [doi:10.1021/ie503002k](https://doi.org/10.1021/ie503002k) |
| Priority-slot continuous-time crude | S. Mouret, I. E. Grossmann, P. Pestiaux. *A novel priority-slot based continuous-time formulation for crude-oil scheduling problems.* Ind. Eng. Chem. Res. 48(18), 2009. [doi:10.1021/ie8019592](https://doi.org/10.1021/ie8019592) |
| Continuous-time crude model comparison | X. Chen, I. E. Grossmann, L. Zheng. *A comparative study of continuous-time models for scheduling of crude oil operations in inland refineries.* Computers & Chemical Engineering 44, 2012. [doi:10.1016/j.compchemeng.2012.05.009](https://doi.org/10.1016/j.compchemeng.2012.05.009) |

## Batched LP and stochastic scenario methods (W05 / W13)

| Component | Reference |
|---|---|
| GPU-batched PDLP (software) | PSORLab. *BatchPDLP.jl* (MIT). [github.com/PSORLab/BatchPDLP.jl](https://github.com/PSORLab/BatchPDLP.jl) |
| Progressive hedging | R. T. Rockafellar, R. J.-B. Wets. *Scenarios and policy aggregation in optimization under uncertainty.* Math. Oper. Res. 16(1), 1991. [doi:10.1287/moor.16.1.119](https://doi.org/10.1287/moor.16.1.119) |
| Adaptive sampling PH | *An adaptive sampling-based progressive hedging algorithm for stochastic programming.* 2024. [arXiv:2407.20944](https://arxiv.org/abs/2407.20944) |

## Mixed precision, iterative refinement, exact arithmetic (W06 / W13)

| Component | Reference |
|---|---|
| Three-precision GMRES-IR | E. Carson, N. J. Higham. *Accelerating the solution of linear systems by iterative refinement in three precisions.* SIAM J. Sci. Comput. 40(2), 2018. [doi:10.1137/17M1140819](https://doi.org/10.1137/17M1140819) |
| Tensor-core mixed-precision IR (SC18) | A. Haidar, S. Tomov, J. Dongarra, N. J. Higham. *Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers.* SC18. [doi:10.1109/SC.2018.00050](https://doi.org/10.1109/SC.2018.00050) |
| Tensor-core IR (Proc. R. Soc. A) | A. Haidar et al. *Mixed-precision iterative refinement using tensor cores on GPUs to accelerate solution of linear systems.* Proc. R. Soc. A 476:20200110, 2020. [doi:10.1098/rspa.2020.0110](https://doi.org/10.1098/rspa.2020.0110) |
| Ozaki Scheme II (GEMM emulation) | *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique.* 2025. [arXiv:2504.08009](https://arxiv.org/abs/2504.08009) |
| FP8 tensor-core DGEMM via Ozaki | *DGEMM without FP64 arithmetic — using FP64 emulation and FP8 tensor cores with Ozaki scheme.* 2025. [arXiv:2508.00441](https://arxiv.org/abs/2508.00441) |

## Certification and safe bounds (W06 / W07 / W13)

| Component | Reference |
|---|---|
| Verifying IP results (VIPR paper) | K. K. F. Cheung, A. Gleixner, D. E. Steffy. *Verifying integer programming results.* CPAIOR 2017, LNCS 10335. [doi:10.1007/978-3-319-59250-3_13](https://doi.org/10.1007/978-3-319-59250-3_13) |
| VIPR software | [github.com/ambros-gleixner/VIPR](https://github.com/ambros-gleixner/VIPR) |
| Safe LP/MILP bounds | A. Neumaier, O. Shcherbina. *Safe bounds in linear and mixed-integer linear programming.* Math. Programming 99, 2004. [doi:10.1007/s10107-003-0433-3](https://doi.org/10.1007/s10107-003-0433-3) |

Literature narrative: `docs/LITERATURE.md` (2026-10-01).
Code survey: `docs/CODE_SURVEY.md` (2026-10-01).

## Open-source solver software (W13 code survey)

Licences verified from upstream `LICENSE` / GitHub licence API on 2026-10-01. QENIVO does not
redistribute these codebases; citations are for implementation comparison only.

| Software | Licence | URL |
|---|---|---|
| cuPDLPx | Apache-2.0 | https://github.com/MIT-Lu-Lab/cuPDLPx |
| NVIDIA cuOpt | Apache-2.0 | https://github.com/NVIDIA/cuopt |
| HiGHS | MIT | https://github.com/ERGO-Code/HiGHS |
| SoPlex | Apache-2.0 | https://github.com/scipopt/soplex |
| CLP (COIN-OR) | EPL-2.0 | https://github.com/coin-or/Clp |
| SCIP | Apache-2.0 | https://github.com/scipopt/scip |
| PSLP (presolve for FO LP) | Apache-2.0 | https://github.com/dance858/PSLP |
| MPAX | MIT | https://github.com/MIT-Lu-Lab/MPAX |
| D-PDLP | Apache-2.0 | https://github.com/Lhongpei/D-PDLP |
