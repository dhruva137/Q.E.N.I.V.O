# Literature survey: parametric LP, crude operations, batch solvers, mixed precision, certificates

Date: 2026-10-01. Branch: `w13-survey`. Agent A (papers).

This note is a **paper-level** survey for workstreams W02 (parametric / ranging), W05 (GPU batch
stack), W06 (mixed precision / exactness), and W07 (crude scheduling MILP). It deliberately does
**not** restate the customer practice synthesis, complexity/roofline limits, or frontier white-space
tables already in `research/2026-10-01/01_customer…`, `02_limits…`, and `03_frontier…`. Those documents
establish *why* planners care and *where* the product wedge is; this one extracts *what the papers
prove*, with numbers, and maps each result onto QENIVO's existing modules (`workload/ranging.py`,
`engines/pdhg` / `pdhg-rf`, `certify/`, `engines/milp`, case stacks).

**Citation policy.** Every primary DOI or arXiv id below was checked against Crossref metadata and/or
an HTTP 200 from arXiv or the publisher landing page on 2026-10-01. Entries that could not be
DOI-verified are marked **[UNVERIFIED DOI]** (URL-only verification where noted). No rival solver
source code is reproduced.

---

## 1. Parametric and post-optimal LP (feeds W02)

Refinery what-if stacks are one-parameter or few-parameter families of LPs that share a matrix and
differ in costs or RHS. Classical parametric programming and modern sensitivity warnings are the
theory behind QENIVO's exact ranging and the degeneracy risk called out in the customer note.

### 1.1 Gass & Saaty (1955) — parametric objective simplex

**Problem.** Given a linear program whose objective is \(c(\theta) = c^0 + \theta\,d\), recover the
optimal basis as a function of the scalar \(\theta\) without restarting from scratch at every
sample.

**Method.** Revised simplex with an extra reduced-cost row for the direction \(d\). As \(\theta\)
moves, the entering variable is the first whose reduced cost crosses zero; a ratio test on the
tableau (or its LU form) yields the next critical value \(\theta^*\). Between breakpoints the basis
is constant, so duals and activity are affine in \(\theta\).

**Key result.** The real line partitions into finitely many intervals on which a single optimal
basis is valid; breakpoints are the values where an alternative optimal basis appears. The
algorithm enumerates those intervals in order.

**Numbers.** On the small hand-worked examples of the era the method finishes in a handful of
pivots per breakpoint; the modern relevance is not the flop count but the combinatorial claim:
**breakpoints, not dense \(\theta\)-grids, are the exact description of a one-parameter cost
sweep.**

**QENIVO.** Cost ranging in `workload/ranging.py` is the local (fixed-basis) special case: given an
optimal basis \(B\), the interval of \(c_j\) for which \(B\) remains dual feasible is exactly one
Gass–Saaty piece. Full parametric sweep / breakpoint walk along a price path is the W02 extension
(homotopy scheduler in the frontier note); do not claim critical-region lookup as the core product
when measured crude ranging half-widths are tiny.

**Citation (verified).** S. Gass, T. Saaty. *The computational algorithm for the parametric
objective function.* Naval Research Logistics Quarterly 2(1–2):39–45, 1955.
[doi:10.1002/nav.3800020106](https://doi.org/10.1002/nav.3800020106)

### 1.2 Gal (1994/1995) — postoptimal and multiparametric monograph

**Problem.** Organise postoptimal analysis (ranging, shadow prices under degeneracy), parametric
programming in one and several parameters, and related multiparametric LP (mpLP) into a single
reference usable by practitioners and theorists.

**Method.** Textbook development of critical regions, rim and matrix parametric forms, and the
connection to Gal & Nedoma's earlier multiparametric algorithm. Emphasis on when dual information
is unique and when only a set of optimal duals exists.

**Key result.** Multiparametric LP admits a finite polyhedral subdivision of the parameter space
into critical regions, each with a fixed optimal basis (or optimal partition). Degeneracy merges
regions and can make naive "shadow price = dual component" reports wrong.

**Numbers.** Gal & Nedoma's original mpLP paper already shows that the number of critical regions
can grow combinatorially with the number of parameters; Gal's monograph is the place that
consequence is spelled out for postoptimal practice.

**QENIVO.** Use Gal as the authority for *language* (critical region, optimal partition, ranging
interval) in W02 docs and planner-facing "why this dual" copy. Do **not** pursue full mpLP region
enumeration as the stack engine: prior measurement (frontier note) found zero cache hits on
128 nearby price cases and median crude ranging half-width ~0.04%. Keep 1-D ranging + breakpoints.

**Citations (verified).** T. Gal. *Postoptimal Analyses, Parametric Programming, and Related
Topics.* De Gruyter, 2nd ed., 1994/1995.
[doi:10.1515/9783110871203](https://doi.org/10.1515/9783110871203).
T. Gal, J. Nedoma. *Multiparametric linear programming.* Management Science 18(7):406–422, 1972.
[doi:10.1287/mnsc.18.7.406](https://doi.org/10.1287/mnsc.18.7.406)

### 1.3 Adler & Monteiro (1992) — geometric / path view of parametric LP

**Problem.** Relate the parametric LP path (as a cost or RHS parameter varies) to the geometry of
optimal faces and to trajectories that interior-point methods track, so that parametric questions
are not tied exclusively to simplex pivots.

**Method.** Analyse the optimal set as a function of the parameter; characterise how the optimal
face jumps and how central-path-like trajectories approach those faces. This is the bridge from
parametric simplex to parametric interior-point behaviour.

**Key result.** The parametric solution map is piecewise smooth away from a finite set of
critical values; at those values the optimal face changes dimension. Interior trajectories see the
same combinatorial events through limiting faces rather than through bases.

**Numbers.** Complexity is governed by the number of optimal-face transitions along the path, not
by the floating-point iteration count of any one IPM solve — the same breakpoint counting as in
Gass–Saaty, restated for faces.

**QENIVO.** Relevant if W02 ever warm-starts an IPM or PDHG along a price homotopy: the "next
interesting \(\theta\)" is still a face/basis event. For the current two-tier design (GPU PDHG to
1e-4, then simplex crossover), Adler–Monteiro justifies treating the **vertex/face change** as the
object of interest, not denser sampling of first-order iterates.

**Citation (verified).** I. Adler, R. D. C. Monteiro. *A geometric view of parametric linear
programming.* Algorithmica 8:161–176, 1992.
[doi:10.1007/BF01758841](https://doi.org/10.1007/BF01758841)

### 1.4 Jansen, de Jong, Roos & Terlaky (1997) — "just be careful!"

**Problem.** Commercial and textbook sensitivity reports often quote a single shadow price and a
single allowable increase/decrease even when the LP is degenerate. Those numbers can be
**arbitrary selections** among many optimal duals, and the reported ranges can be wrong for the
intended managerial question.

**Method.** Construct small, fully worked degenerate LPs where (i) two solvers return different
duals at the same optimal objective, (ii) classical ranging intervals fail to describe the true
break-even of a RHS change, and (iii) the **optimal partition** (index sets of positive primals /
positive dual slacks) is the object that remains stable.

**Key result.** Under primal or dual degeneracy, shadow prices are set-valued. Correct
managerial sensitivity needs either the full optimal dual set, a conventional selection rule
(lexicographic, central), or explicit intervals — not a single float from the basis on hand.
Optimal-partition methods avoid some basis-dependent artefacts.

**Numbers.** Their counter-examples are tiny (a few variables/constraints) but decisive: duals
that differ by O(1) absolute while the objective agrees to machine precision. That is exactly the
failure mode planners hit when two LP packages disagree on crude breakeven.

**QENIVO (critical for W02).** Degeneracy is the main product risk for "explainable marginal
values." Required behaviours: detect primal/dual degeneracy; report ranging as an interval;
label duals as "one optimal dual" unless uniqueness is proven; optionally offer a lexicographic or
central dual to match incumbent tools. Never ship a single shadow price as *the* crude value
without the degeneracy flag.

**Citation (verified).** B. Jansen, J. J. de Jong, C. Roos, T. Terlaky. *Sensitivity analysis in
linear programming: just be careful!* European Journal of Operational Research 101(1):15–28, 1997.
[doi:10.1016/S0377-2217(96)00172-5](https://doi.org/10.1016/S0377-2217(96)00172-5)

### 1.5 Greenberg (1994) — optimal partition for postoptimal analysis

**Problem.** After an LP solve, support what-if questions (rim changes, coefficient probes) in a
way that is consistent when bases are non-unique.

**Method.** Prefer the **optimal partition** induced by a strictly complementary solution (or an
interior solution of the optimal face) over a particular optimal basis. Postoptimal probes are
then phrased in terms of that partition.

**Key result.** Many postoptimal queries that look basis-dependent become well-defined once the
partition is fixed; this is the constructive answer to the Jansen et al. warning.

**Numbers.** Greenberg's note is methodological rather than a large computational study; its
influence is through solver design (strict complementarity / optimal face) rather than a
benchmark table.

**QENIVO.** Pair with Jansen: ranging and certificates should record whether the dual is unique,
and when an IPM or PDHG point is used only as a crossover seed, the **simplex vertex** (plus
partition if available) is what planners should see. Candidate W02 work: expose partition /
degeneracy bits in the JSON certificate.

**Citation (verified).** H. J. Greenberg. *The use of the optimal partition in a linear programming
solution for postoptimal analysis.* Operations Research Letters 15(4):157–163, 1994.
[doi:10.1016/0167-6377(94)90075-2](https://doi.org/10.1016/0167-6377(94)90075-2)

---

## 2. Refinery planning and crude selection (feeds W02 / planner models)

Planning LPs decide monthly crude baskets and unit throughputs; duals on crude purchase rows are
the economic object QENIVO must certify. This section stays on formulation and valuation papers,
not the industrial-process narrative already covered in the customer synthesis.

### 2.1 Li, Hui & Li — planning under uncertainty and integrated unit models

**Problem.** Refinery planning LPs are wrong in two ways at once: yields and demands are uncertain,
and crude distillation / FCC / blending submodels are often over-simplified, so the dual on a crude
can be an artefact of the yield table.

**Method.** (i) Stochastic / scenario formulations of the planning LP with uncertain prices and
demands; (ii) higher-fidelity CDU–FCC–blending blocks embedded in the planning model so that
marginal values reflect process structure, not only aggregate yields.

**Key result.** Planning quality and crude rankings are sensitive to both uncertainty modelling and
unit detail; a deterministic LP with coarse yields can mis-order crudes even when the solver is
exact.

**Numbers.** The 2004 IECR uncertainty paper works with industrial-scale planning structures
(dozens of crudes/products, multiperiod); the 2005 Computers & Chemical Engineering integration
paper shows material-balance consistency across CDU, FCC and blend headers — the regime where
shadow prices become operationally meaningful.

**QENIVO.** Scenario stacks (`workload` case API) already match the multi-data pattern; dual
certification and ranging answer the "is this crude breakeven real?" question once the matrix is
trusted. Integrated CDU–FCC–blend models belong in `models/industry` / GAMS import paths (W07
adjacent), not in the LP engine.

**Citations (verified).** W. Li, C.-W. Hui, A. Li, P. Li. *Refinery planning under uncertainty.*
Industrial & Engineering Chemistry Research 43(21):6742–6755, 2004.
[doi:10.1021/ie049737d](https://doi.org/10.1021/ie049737d).
W. Li, C.-W. Hui, A. Li. *Integrating CDU, FCC and product blending models into refinery
planning.* Computers & Chemical Engineering 29(9):2010–2028, 2005.
[doi:10.1016/j.compchemeng.2005.05.010](https://doi.org/10.1016/j.compchemeng.2005.05.010)

### 2.2 Gueddar & Dua — reduced models for refinery-wide optimisation / selection

**Problem.** Full refinery-wide NLP/MINLP models are too heavy for repeated crude-selection or
energy-optimisation loops; planners need reduced models that preserve the ranking of decisions.

**Method.** Disaggregation–aggregation and data-based model reduction: build smaller surrogate
planning models from detailed simulations or from structured aggregation of process blocks, then
optimise the surrogate for crude/energy decisions.

**Key result.** Carefully reduced models can recover near-optimal refinery-wide decisions at a
fraction of the detailed-model cost; the economic ranking of crudes is the quantity to preserve,
not every intermediate flow.

**Numbers.** Reported case studies cut model size enough for repeated optimisation passes while
keeping energy / profit metrics close to the detailed reference (paper-dependent; treat percentages
as study-specific, not universal constants).

**QENIVO.** Supports the product story that the **solver** must be exact on whatever matrix the
planner trusts — reduction is a modelling choice upstream of QENIVO. Stack throughput matters
because reduced models are still re-solved for many price/availability cases.

**Citations (verified).** T. Gueddar, V. Dua. *Disaggregation–aggregation based model reduction for
refinery-wide optimization.* Computers & Chemical Engineering 35(12):2874–2886, 2011.
[doi:10.1016/j.compchemeng.2011.04.016](https://doi.org/10.1016/j.compchemeng.2011.04.016).
T. Gueddar, V. Dua. *Novel model reduction techniques for refinery-wide energy optimisation.*
Applied Energy 89(1):117–126, 2012.
[doi:10.1016/j.apenergy.2011.05.056](https://doi.org/10.1016/j.apenergy.2011.05.056)

### 2.3 Du et al. (2025) — refinery–petrochemical planning benchmark

**Problem.** Public, realistic planning benchmarks for integrated refinery–petrochemical complexes
have been scarce, which blocks reproducible solver and formulation comparisons.

**Method.** Release a production-planning benchmark with industrial structure (process units,
streams, blending, purchase/sale decisions) suitable for LP/MILP/MINLP experiments and for
cross-solver timing.

**Key result.** A citable open instance family that future QENIVO planning demos can use without
proprietary PIMS decks.

**Numbers.** See the paper for instance sizes (variables/constraints per scenario); treat the
arXiv abstract figures as authoritative until a journal version freezes them.

**QENIVO.** Primary public planning corpus for W02/W05 demos and for dual/ranging reconciliation
reports against HiGHS/CPLEX-style baselines. Prefer this over ad-hoc toy blend LPs when claiming
planner relevance.

**Citation (verified).** W. Du et al. *A production planning benchmark for real-world
refinery-petrochemical complexes.* 2025. [arXiv:2503.22057](https://arxiv.org/abs/2503.22057)

---

## 3. Crude scheduling MILP (feeds W07)

Scheduling sits below planning: vessels, tanks, charging tanks, CDUs, and blend recipes over days
to weeks. These papers define the MILP shapes QENIVO's branch-and-cut and industry models must
carry.

### 3.1 Lee, Pinto, Grossmann & Park (1996) — discrete-time unloading / inventory MILP

**Problem.** Short-term crude unloading and tank inventory management with vessel arrivals, tank
capacities, and CDU charging limits — decisions that are discrete in time and logistics.

**Method.** Mixed-integer linear programming on a discrete time grid: binary variables for vessel–
tank and tank–CDU connections; continuous inventories and flows; material balances and capacity
inequalities.

**Key result.** A solvable MILP that captures the logistics skeleton of crude scheduling without
yet closing all blending nonlinearities — the template for later continuous-time and hybrid
models.

**Numbers.** Industrial motivation at refinery scale; computational sections of the era report
solve times in minutes on then-current MIP solvers for the presented instances (exact timings are
hardware-dated; the lasting number is "hundreds of binaries" growing with the horizon and tank
count).

**QENIVO.** Canonical discrete-time formulation for `models/industry` crude-scheduling demos and
for W07 gap plots versus HiGHS. Prefer this as the **baseline** before continuous-time variants.

**Citation (verified).** H. Lee, J. M. Pinto, I. E. Grossmann, S. Park. *Mixed-integer linear
programming model for refinery short-term scheduling of crude oil unloading with inventory
management.* Industrial & Engineering Chemistry Research 35(5):1630–1641, 1996.
[doi:10.1021/ie950519h](https://doi.org/10.1021/ie950519h)

### 3.2 Jia, Ierapetritou & Kelly (2003) — continuous-time crude operations

**Problem.** Discrete grids either explode in binaries or mis-time events; crude operations need
event- or slot-based continuous-time models that still remain MILP.

**Method.** Continuous-time MILP with event points for vessel unloading, tank transfers, and CDU
charging; big-M timing constraints; inventory tracking between events.

**Key result.** Continuous-time formulations can cut binary counts relative to fine discrete grids
while representing the same logistics, at the cost of weaker LP relaxations and trickier big-M
constants.

**Numbers.** Case studies show substantial reductions in binary variables versus uniform
discretisation for the same scheduling horizon (paper tables); solve success still depends on
horizon length and number of tanks.

**QENIVO.** Second formulation family for W07: compare discrete (Lee) vs continuous-time (Jia) on
the same physical pattern; document which branch-and-cut features (cuts, diving) help each.

**Citation (verified).** Z. Jia, M. Ierapetritou, J. D. Kelly. *Refinery short-term scheduling using
continuous time formulation: crude-oil operations.* Industrial & Engineering Chemistry Research
42(13):3085–3097, 2003. [doi:10.1021/ie020124f](https://doi.org/10.1021/ie020124f)

### 3.3 Kelly & Mann (2003) — industrial blend-scheduling value

**Problem.** Argue, from industrial practice, that crude blend scheduling optimisation returns
large absolute dollar benefits and is not a toy OR exercise.

**Method.** Application narrative and optimisation workflow for crude blend scheduling in an
operating refinery context (Hydrocarbon Processing special report), emphasising blend quality,
tank logistics, and economic upside.

**Key result.** Scheduling quality moves millions of dollars per year in the cited application
setting — the business case for investing in MILP scheduling, not only monthly planning LPs.

**Numbers.** The article's "multimillion dollar benefits" claim is the headline figure; treat it
as industry-reported, not as a controlled academic benchmark.

**QENIVO.** Use for pitch / W07 motivation only. Pair with a **measurable** MILP gap on a public
or synthetic SBM/VLCC instance rather than quoting the magazine figure as a solver result.

**Citation (URL verified; no Crossref DOI found).** J. D. Kelly, J. L. Mann. *Crude oil blend
scheduling optimization: an application with multimillion dollar benefits.* Hydrocarbon
Processing 82(6), 2003.
[URL](https://www.hydrocarbonprocessing.com/magazine/2003/june-2003/special-report-processplant-optimization/crude-oil-blend-scheduling-optimization-an-application-with-multimillion-dollar-benefits-part-1/)
**[UNVERIFIED DOI]**

### 3.4 Reddy, Karimi & Srinivasan (2004) — SBM / VLCC multi-CDU operations

**Problem.** Coastal refineries (MRPL-like pattern) receive VLCCs via SBM and jetties into storage
and charging tanks feeding multiple CDUs; composition and logistics couple into a hard scheduling
MINLP/MILP.

**Method.** Detailed scheduling model of unloading, tank inventory, and charging with a novel
solution approach (decomposition / heuristics layered on the mathematical model) aimed at
industrial feasibility.

**Key result.** The physical pattern — SBM/VLCC → storage → charging → multi-CDU — is modellable
and solvable with specialised strategies even when a black-box MIP struggle on the full horizon.

**Numbers.** Instances involve large binary sets and nonlinear blending constraints; the paper's
contribution is solvability via structure, not a single open MIPLIB-style gap number.

**QENIVO.** Preferred **geography-faithful** W07 demo model. Build a 7-day horizon MILP with
thousands of binaries, report feasible schedule + bound vs HiGHS within a fixed time — matching
the action item already named in the customer synthesis, with this paper as the formulation
source.

**Citation (verified).** P. C. P. Reddy, I. A. Karimi, R. Srinivasan. *Novel solution approach for
optimizing crude oil operations.* AIChE Journal 50(6):1177–1197, 2004.
[doi:10.1002/aic.10112](https://doi.org/10.1002/aic.10112)

### 3.5 Castro & Grossmann (and Grossmann-line continuous-time crude models)

**Problem.** Need tight continuous-time MILP representations for crude blending/scheduling that
can aim at global optimality when bilinear blending appears (often via reformulation or
multiparametric disaggregation).

**Method.** Resource-Task Network (RTN) continuous-time models; priority-slot formulations
(Mouret–Grossmann–Pestiaux); multiparametric disaggregation to tighten or globalise blending
products; comparative studies of continuous-time crude models.

**Key result.** Priority-slot and RTN forms are competitive MILP vehicles for crude scheduling;
global approaches exist for blending substructures but scale carefully with the number of slots
and quality specs.

**Numbers.** Mouret et al. and Castro & Grossmann report instance libraries with tens to low
hundreds of slots/events; comparative studies (Chen–Grossmann–Zheng) show that model choice
changes LP gap and time-to-feasibility by large factors on the same physical data.

**QENIVO.** Use Castro–Grossmann RTN / multiparametric disaggregation as the **advanced** W07
track after Lee and Reddy baselines. Do not start product demos here; graduate to it when discrete
baselines are certified.

**Citations (verified).** P. M. Castro, I. E. Grossmann. *Global optimal scheduling of crude oil
blending operations with RTN continuous-time and multiparametric disaggregation.* Industrial &
Engineering Chemistry Research 53(39):15127–15145, 2014.
[doi:10.1021/ie503002k](https://doi.org/10.1021/ie503002k).
S. Mouret, I. E. Grossmann, P. Pestiaux. *A novel priority-slot based continuous-time formulation
for crude-oil scheduling problems.* Industrial & Engineering Chemistry Research 48(18):8515–8528,
2009. [doi:10.1021/ie8019592](https://doi.org/10.1021/ie8019592).
X. Chen, I. E. Grossmann, L. Zheng. *A comparative study of continuous-time models for scheduling
of crude oil operations in inland refineries.* Computers & Chemical Engineering 44:141–167, 2012.
[doi:10.1016/j.compchemeng.2012.05.009](https://doi.org/10.1016/j.compchemeng.2012.05.009)

---

## 4. Batched / multi-scenario LP on GPUs and stochastic LP (feeds W05 / post-deadline)

QENIVO's Tier-1 story is batched first-order LP on a shared matrix. This section covers the
closest published batch solvers and the stochastic-programming use of a scenario cloud — without
replaying the cuOpt / cuPDLPx white-space table.

### 4.1 BatchPDLP.jl — GPU-batched PDLP for many small LPs

**Problem.** Solve many independent (or lightly related) LPs concurrently on a GPU when each LP is
small-to-medium and host launch overhead would dominate a sequential loop.

**Method.** Julia implementation of PDLP with a batch axis, GPU kernels over the stack, MIT
licence. Designed for throughput across instances rather than a single giant LP.

**Key result.** Demonstrates that PDHG/PDLP maps naturally onto batched SpMV/SpMM; engineering
focus is occupancy, batch packing, and per-instance termination.

**Numbers.** See repository benchmarks for batch sizes and per-instance iteration rates; treat
GitHub HEAD numbers as moving. Qualitatively: batching wins when S is large and each \(A\) is not
huge — complementary to QENIVO's "one big shared \(A\), many \(c/b\)" refinery stack.

**QENIVO.** Already noted as a related system in `docs/SURVEY.md`. Difference to emphasise: QENIVO
targets **shared-matrix** refinery stacks with streaming crossover and certificates; BatchPDLP.jl
is the small-LP batch reference. Candidate: adopt packing/compaction ideas, not code.

**Citation (software verified).** PSORLab. *BatchPDLP.jl* (MIT).
[github.com/PSORLab/BatchPDLP.jl](https://github.com/PSORLab/BatchPDLP.jl)

### 4.2 Progressive hedging and scenario decomposition

**Problem.** Turn a cloud of what-if LPs into one implementable first-stage decision (robust /
stochastic plan), not only a spreadsheet of scenario optima.

**Method.** Rockafellar–Wets progressive hedging (PH): decompose by scenario, penalise disagreement
on non-anticipative variables, iterate. Modern variants add adaptive sampling of scenarios.
GPU opportunity: each PH subproblem is a QP/LP on a shared structure — a batched PDQP/PDLP loop.

**Key result.** PH converges under convexity for the classical two-stage stochastic LP; it is a
practical heuristic/algorithm for large scenario sets when extensive-form MIP is impossible.

**Numbers.** Classical theory is asymptotic; adaptive PH (2024) reports wall-clock reductions on
standard stochastic LP test sets by sampling fewer scenarios early (see arXiv for tables). No
published system yet matches "GPU-batched PH on shared refinery \(A\)" as a product feature.

**QENIVO.** Post-deadline product candidate (frontier runner-up): run PH outer loop with inner
batched PDQP on the existing case stack. Near-term W05 stays on **evaluate all scenarios**; PH is
"decide given all scenarios."

**Citations (verified).** R. T. Rockafellar, R. J.-B. Wets. *Scenarios and policy aggregation in
optimization under uncertainty.* Mathematics of Operations Research 16(1):119–147, 1991.
[doi:10.1287/moor.16.1.119](https://doi.org/10.1287/moor.16.1.119).
*An adaptive sampling-based progressive hedging algorithm for stochastic programming.* 2024.
[arXiv:2407.20944](https://arxiv.org/abs/2407.20944)

### 4.3 Scenario decomposition vs QENIVO case stacks

**Problem.** Distinguish Benders / dual decomposition / PH (stochastic programming) from planner
what-if stacks (many related LPs, each answer kept).

**Method.** Decomposition methods coordinate subproblems toward one policy; QENIVO case stacks
emit one certified optimum per case. Both use batch linear algebra; only the outer loop differs.

**Key result.** Same GPU SpMM substrate serves both products; certification requirements differ
(policy feasibility vs per-case KKT).

**QENIVO.** Keep APIs separate: `cases` for what-ifs; a future `hedge`/`ph` for stochastic
aggregation. Do not blur certificates.

---

## 5. Mixed-precision iterative refinement and exact LP (feeds W06)

Consumer GPUs are weak at fp64. Mixed precision plus refinement is how QENIVO's `pdhg-rf` and a
future certified basis engine stay exact.

### 5.1 Carson & Higham — GMRES-IR in three precisions

**Problem.** Solve \(Ax=b\) faster and to high accuracy by factoring in low precision and refining
in higher precision, with GMRES applied to the preconditioned residual system.

**Method.** LU (or other factorisation) in precision \(u_f\); residual and correction solves
orchestrated in precisions \(u\) and \(u_r\); GMRES-IR to restore backward stability when classical
iterative refinement stalls.

**Key result.** Under stated conditions on the condition number relative to the precisions, the
method achieves residuals on the order of the working precision despite a cheap factorisation.

**Numbers.** Analyses give explicit \(\kappa(A)\) thresholds (functions of \(u_f,u,u_r\)) beyond
which refinement may fail without higher precision or stronger preconditioning — the checklist for
when QENIVO must fall back to fp64 factorisation.

**QENIVO.** Theory spine for W06 basis work: factor \(B\) cheaply, refine FTRAN/BTRAN, prove the
vertex. Also legitimises `pdhg-rf` (fp32 iterations, fp64 KKT) as IR in the first-order setting.

**Citation (verified).** E. Carson, N. J. Higham. *Accelerating the solution of linear systems by
iterative refinement in three precisions.* SIAM Journal on Scientific Computing 40(2):A817–A847,
2018. [doi:10.1137/17M1140819](https://doi.org/10.1137/17M1140819)

### 5.2 Haidar et al. (SC18; Proc. R. Soc. A 2020) — tensor-core IR on GPUs

**Problem.** Use fp16/bf16 tensor cores for the heavy factorisation/GEMM work while delivering
fp64-quality answers for dense linear systems on GPUs.

**Method.** Mixed-precision factorisation on tensor cores; iterative refinement (and related
precisions) to recover accuracy; carefully engineered GPU kernels.

**Key result.** Large speedups on dense LU/GEMM-bound solves versus fp64-only paths, with final
residuals matching fp64 targets when IR converges.

**Numbers.** SC18 and the 2020 Royal Society follow-on report multi-× speedups on GPU dense
systems (hardware- and size-dependent; cite the paper's tables for the card you compare to). The
qualitative constant for QENIVO: **tensor-core math is usable inside a certified loop**, not only
for ML.

**QENIVO.** Direction for certified mixed-precision **basis** engines on T4/L4/RTX (frontier
Direction 2). `pdhg-rf` already applies the idea to PDHG; W06 extends it to simplex LU.

**Citations (verified).** A. Haidar, S. Tomov, J. Dongarra, N. J. Higham. *Harnessing GPU tensor
cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers.* SC18.
[doi:10.1109/SC.2018.00050](https://doi.org/10.1109/SC.2018.00050).
A. Haidar et al. *Mixed-precision iterative refinement using tensor cores on GPUs to accelerate
solution of linear systems.* Proc. R. Soc. A 476:20200110, 2020.
[doi:10.1098/rspa.2020.0110](https://doi.org/10.1098/rspa.2020.0110)

### 5.3 Gleixner, Steffy & Wolter — iterative refinement for LP

**Problem.** Compute highly accurate (or exact rational) LP solutions using floating-point LP
solvers plus refinement, without paying full exact-arithmetic cost on every pivot.

**Method.** Solve in floating point; construct refined problems from residual information; iterate
until feasibility/optimality certificates pass at tight tolerances; optionally hand off to exact
methods.

**Key result.** Iterative refinement is practical for LP: many instances reach very tight
tolerances or exact answers with few refinement rounds on top of a standard engine.

**Numbers.** INFORMS JOC computational study: large fractions of Netlib/MIPLIB-relaxation-style LPs
reach the requested accuracy with small refinement iteration counts (see paper tables).

**QENIVO.** Already listed in `docs/REFERENCES.md` and aligned with `certify` + exact verify.
W06 should treat Gleixner–Steffy–Wolter as the **LP-specific** IR playbook, with Carson–Higham /
Haidar as the linear-algebra substrate.

**Citation (verified; already in REFERENCES).** A. M. Gleixner, D. E. Steffy, K. Wolter.
*Iterative refinement for linear programming.* INFORMS Journal on Computing 28(3):449–464, 2016.
[doi:10.1287/ijoc.2016.0692](https://doi.org/10.1287/ijoc.2016.0692)

### 5.4 Ozaki schemes — exact-ish GEMM via low-precision hardware

**Problem.** Emulate accurate fp64 matrix multiplication using integer modular techniques and/or
fp8/fp16 tensor cores, for workloads where native fp64 throughput is poor.

**Method.** Ozaki Scheme II and follow-on FP8 tensor-core DGEMM emulation: split/scale multiplications
so that low-precision hardware accumulates an correctly rounded or high-accuracy product.

**Key result.** DGEMM-class work can be offloaded to tensor-core paths with predictable accuracy
trade-offs, opening exact or high-accuracy verification GEMMs on consumer GPUs.

**Numbers.** arXiv reports detail error bounds and throughput versus cuBLAS fp64 on target GPUs;
use those tables when claiming T4-vs-A100 ratios.

**QENIVO.** Candidate for stack-wide **integer/Ozaki verification GEMMs** of \(Ax\) and dual
residuals (frontier Direction 2 certification layer). Complements, does not replace, rational
verify on CPU.

**Citations (verified).** *Ozaki Scheme II…* [arXiv:2504.08009](https://arxiv.org/abs/2504.08009).
*DGEMM without FP64 arithmetic…* [arXiv:2508.00441](https://arxiv.org/abs/2508.00441)

---

## 6. Certification of LP / MILP answers (feeds W06 / Tier 3)

Planners and auditors need checkable evidence, not log lines from the engine.

### 6.1 VIPR and Cheung–Gleixner–Steffy — verifying integer programming results

**Problem.** Branch-and-cut emits a tree of dual bounds, cuts, and pruning decisions; reproducing
"MIP gap proven" requires a portable proof format independent of the solver binary.

**Method.** VIPR (Verifying Integer Programming Results): a proof certificate format and checker.
Cheung, Gleixner & Steffy formalise how to verify IP results using such certificates (cuts,
branching, bound changes) so a third party can re-check optimality or infeasibility claims.

**Key result.** MIP optimality claims can be made auditable: the checker replays reasoning without
trusting the originating solver's arithmetic beyond what the certificate states.

**Numbers.** Tooling papers report successful verification on suites of MIP proofs; size overhead
of certificates is instance-dependent (can be large — budget disk/IO in W07 MILP demos).

**QENIVO.** LP path already has JSON KKT certificates + independent verifier. MILP path should
grow toward VIPR-class proofs for node bounds and final gap, especially when claiming scheduling
feasibility/gap. Near-term: keep LP certificates rock-solid; add VIPR export as W07 stretch.

**Citations (verified).** K. K. F. Cheung, A. Gleixner, D. E. Steffy. *Verifying integer programming
results.* In: CPAIOR 2017, LNCS 10335, Springer.
[doi:10.1007/978-3-319-59250-3_13](https://doi.org/10.1007/978-3-319-59250-3_13).
VIPR software: [github.com/ambros-gleixner/VIPR](https://github.com/ambros-gleixner/VIPR)

### 6.2 Neumaier & Shcherbina — safe bounds for LP / MILP

**Problem.** Floating-point LP relaxations can be optimistic: a computed dual "bound" may not be a
true valid bound because of rounding, which is fatal in branch-and-bound.

**Method.** Construct rigorously valid dual bounds using directed rounding / interval methods so
that a reported lower (upper) bound cannot cut off feasible integer solutions by accident.

**Key result.** Safe bounds are slightly weaker but **sound**; they are the correct substrate for
MIP certificates and for any "proven gap" claim.

**Numbers.** Overhead is typically modest relative to unsafe floating-point bounds on standard
MIP test sets (see paper); the important constant is zero false fathoming.

**QENIVO.** Adopt safe-bound language in MILP node LPs and in any answer-cache brackets (frontier
Direction 3). LP JSON certificates should remain KKT-based; MILP bound claims should cite
Neumaier–Shcherbina soundness.

**Citation (verified).** A. Neumaier, O. Shcherbina. *Safe bounds in linear and mixed-integer
linear programming.* Mathematical Programming 99:339–356, 2004.
[doi:10.1007/s10107-003-0433-3](https://doi.org/10.1007/s10107-003-0433-3)

---

## Actionables for W02 / W05 / W06 / W07

Pointers are to sections of this file. Parent / Agent B should fold the top ten into
`WORKLOG_W13.md`.

1. **W02 — Degeneracy-first ranging UX (§1.4, §1.5).** Ship interval duals + degeneracy flags;
   treat Jansen/Greenberg as acceptance tests for crude breakeven reports.
2. **W02 — Keep 1-D Gass–Saaty ranging; do not bet on mpLP regions (§1.1–§1.2).** Breakpoint walk
   along a price path is enough; Gal regions are documentation, not the engine.
3. **W02 — Parametric events = face/basis changes (§1.3).** Homotopy schedulers should advance to
   the next critical \(\theta\), not to a fixed \(\Delta\)-grid.
4. **W02/W05 — Du et al. benchmark as public planning corpus (§2.3).** Prefer arXiv:2503.22057
   instances for demos and dual reconciliation.
5. **W05 — Shared-\(A\) batch ≠ BatchPDLP small-LP batch (§4.1).** Steal packing ideas; keep
   streaming crossover + certificate as the differentiator.
6. **W05 (post-deadline) — GPU progressive hedging on the stack (§4.2).** Rockafellar–Wets outer
   loop + batched PDQP inner loop as the "decide given what-ifs" product.
7. **W06 — Carson–Higham / Haidar IR for basis LU; Gleixner for LP refinement (§5.1–§5.3).**
   Formalise `pdhg-rf` against this literature; plan tensor-core factorisation of \(B\).
8. **W06 — Ozaki GEMM for stack verification (§5.4).** Candidate exact-ish \(Ax\) checks on T4
   before CPU rational verify.
9. **W06/W07 — Safe bounds + VIPR path (§6.1–§6.2).** Neumaier–Shcherbina for MILP node bounds;
   VIPR export as stretch for scheduling gap claims.
10. **W07 — Formulation ladder Lee → Reddy → Jia/Castro (§3.1–§3.5).** Discrete baseline, then
    MRPL-like SBM/VLCC, then continuous-time/RTN; Kelly–Mann for motivation only (**no DOI**).

---

## Verification log (2026-10-01)

| Status | Items |
|---|---|
| Crossref DOI OK | Gass–Saaty; Jansen et al.; Adler–Monteiro; Gal book; Gal–Nedoma; Greenberg 1994; Lee et al.; Jia et al.; Reddy et al.; Castro–Grossmann 2014; Mouret et al.; Chen–Grossmann–Zheng; Li–Hui–Li (2004 & 2005); Gueddar–Dua (2011 & 2012); Carson–Higham; Haidar SC18; Haidar 2020; Gleixner–Steffy–Wolter; Cheung–Gleixner–Steffy; Neumaier–Shcherbina; Rockafellar–Wets |
| arXiv HTTP / API OK | 2503.22057; 2504.08009; 2508.00441; 2407.20944 |
| Software URL OK | BatchPDLP.jl; VIPR |
| **[UNVERIFIED DOI]** | Kelly & Mann 2003 (Hydrocarbon Processing) — landing page HTTP 200, no Crossref DOI located |

---

## Code survey

Rival **implementation** choices (data structures, tolerances, restart rules, crossover triggers,
ranging/parametrics, exact mode), with licences and already-in-QENIVO / candidate / not-applicable
labels, are in [`docs/CODE_SURVEY.md`](CODE_SURVEY.md) (Agent B, 2026-10-01). That note verifies and
deepens `research/2026-10-01/03_frontier_and_white_space.md` against public GitHub sources; no rival
code is copied into QENIVO.
