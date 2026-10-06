"""Engine routing: single-LP engine pick, plus stack-structure routing for case stacks.

The rule is a measurement-derived decision table, not a guess, and every decision is written
into the certificate ("routing.reason"), so a reviewer can see why a model went where it went.

Single-LP rules (defaults; tunable through `RoutingPolicy`); case stacks use `route_stack` below:
  QP (default)                         -> ipm-native      (Mehrotra + supernodal LDL', KKT cert)
  QP, very large (A+Q nnz) or GPU batch-> pdqp            (first-order; GPU when available)
  MILP                                 -> milp (branch and bound over the LP engines)
  LP, batch S >= batch_gpu_min, GPU    -> pdhg on the GPU (one pass serves every scenario)
  LP, nnz >= gpu_nnz_min, GPU          -> pdhg on the GPU
  LP, rows <= vertex_rows_max          -> simplex          (exact vertex, basis, clean duals)
  LP, rows <= ipm_rows_max             -> ipm              (few, accurate iterations)
  otherwise                            -> pdhg on the CPU  (memory-light, matrix-free)
The thresholds come from the team's GPU decision-map runs (L4 and A100, 2026-09-27/28): the
GPU lost on single small LPs and won from batch 8 upwards or from ~1e5 nonzeros.

Stack routing (`classify_stack` / `route_stack`) picks CPU warm chain vs GPU batch vs mixed
clusters from the delta structure. Thresholds cite W01 L3/S=64 aggregates in
`bench/results/stack_bench_2026-10-01.jsonl` (see `StackRoutingPolicy` comments).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..kernels.backend import gpu_available


@dataclass
class RoutingPolicy:
    batch_gpu_min: int = 8
    gpu_nnz_min: int = 100_000
    vertex_rows_max: int = 600
    vertex_rows_max_native: int = 5000   # with the native C++ simplex (Netlib: all verified, see bench/results)
    ipm_rows_max: int = 3000
    # Convex QP: native IPM below this combined A+Q nnz; PDQP above (and for GPU batches).
    qp_ipm_nnz_max: int = 250_000
    prefer_gpu: bool = True


def route(prob, batch: int = 1, engine: str = "auto", backend: str = "auto",
          policy: RoutingPolicy | None = None) -> dict:
    policy = policy or RoutingPolicy()
    has_gpu = gpu_available() if backend in ("auto", "cupy", "gpu", "cuda") else False
    if backend in ("cupy", "gpu", "cuda") and not has_gpu:
        raise RuntimeError("GPU backend requested but no CUDA device / CuPy found")
    want_gpu = has_gpu and policy.prefer_gpu and backend != "numpy"

    def pick(eng, dev, why):
        return {"engine": eng, "backend": dev, "reason": why, "gpu_available": has_gpu,
                "batch": batch, "rows": prob.m, "cols": prob.n, "nnz": prob.nnz}

    if engine != "auto":
        dev = "cupy" if (want_gpu and engine in ("pdhg", "pdhg-rf", "pdqp") and backend != "numpy") else "numpy"
        if backend in ("numpy", "cpu"):
            dev = "numpy"
        return pick(engine, dev, "engine chosen by the caller")
    if prob.is_mip and prob.is_qp:
        return pick("miqp", "numpy", "integer variables and a quadratic objective (MIQP plugin)")
    if prob.is_mip:
        return pick("milp", "numpy", "integer variables present")
    if prob.is_qp:
        qnnz = int(prob.Q.nnz) if prob.Q is not None else 0
        total = int(prob.nnz) + qnnz
        if batch >= policy.batch_gpu_min and want_gpu:
            return pick("pdqp", "cupy",
                        f"QP batch of {batch} >= {policy.batch_gpu_min}: PDQP on GPU")
        # Wide-but-short QPs (few rows, many cols) still favour IPM: the normal-equations
        # factor is m×m. Only send truly large row-counts above the nnz threshold to PDQP.
        large = total >= policy.qp_ipm_nnz_max and prob.m > policy.ipm_rows_max
        if large:
            dev = "cupy" if want_gpu else "numpy"
            return pick("pdqp", dev,
                        f"large QP ({total} A+Q nnz, {prob.m} rows > {policy.ipm_rows_max}): PDQP")
        return pick("ipm-native", "numpy",
                    "convex QP: native Mehrotra IPM with supernodal LDL' (KKT-certified)")
    if batch >= policy.batch_gpu_min and want_gpu:
        return pick("pdhg", "cupy", f"batch of {batch} scenarios >= {policy.batch_gpu_min}: one GPU pass serves all")
    if prob.nnz >= policy.gpu_nnz_min and want_gpu:
        return pick("pdhg", "cupy", f"{prob.nnz} nonzeros >= {policy.gpu_nnz_min}: GPU first-order wins at this size")
    if batch > 1:
        single = route(prob, 1, "auto", "numpy", policy)
        return pick(single["engine"], "numpy",
                    f"{batch} cases without a GPU: case by case on {single['engine']}, warm-started ({single['reason']})")
    vmax = policy.vertex_rows_max
    try:                                          # the native C++ simplex handles much larger models
        from ..engines.native_simplex import library
        if library() is not None:
            vmax = max(vmax, policy.vertex_rows_max_native)
    except Exception:  # noqa: BLE001
        pass
    if batch == 1 and prob.m <= vmax:
        return pick("simplex", "numpy", f"{prob.m} rows <= {vmax}: exact vertex solution is cheap")
    if batch == 1 and prob.m <= policy.ipm_rows_max:
        return pick("ipm", "numpy", f"{prob.m} rows <= {policy.ipm_rows_max}: interior point, few accurate iterations")
    return pick("pdhg", "numpy", "large model without a GPU: matrix-free first-order method")


# -------------------------------------------------------------------------------- stack router
# Policy constants calibrated from the L3/S=64 stack_aggregate rows
# (bench/results/stack_bench_2026-10-01.jsonl, summarised in the matching notes):
#
#   key L3/one_crude/S64/seed0:
#     qenivo_warm wall=5.46 sum=4.58; highs_warm wall=5.21 sum=3.99; zp_qwarm=57.8%
#     -> CPU warm chain (one-factor cost ray)
#   key L3/cargo_menu/S64/seed0:
#     qenivo_warm wall=4.47 sum=3.96; highs_warm wall=5.57 sum=4.02; zp_qwarm=57.8%
#     -> CPU warm (same one-crude shape)
#   key L3/one_unit/S64/seed0:
#     qenivo_warm wall=5.71 sum=5.22; highs_warm wall=4.95 sum=3.83; zp_qwarm=26.6%
#     -> CPU warm (one-factor bound ray); HiGHS warm slightly ahead but same lane
#   key L3/factor_price/S64/seed0:
#     qenivo_warm wall=6.03 sum=6.06; highs_warm wall=6.37 sum=5.97; zp_qwarm=1.6%
#     -> CPU warm (low-rank cost factors; rank ~3 still warm-competitive)
#   key L3/factor_mixed/S64/seed0:
#     qenivo_warm wall=13.83 sum=18.93; highs_warm wall=7.55 sum=8.95; zp_qwarm=0.0%
#     -> mixed / cluster (costs+bounds); HiGHS warm faster on CPU alone
#   key L3/independent/S64/seed0:
#     qenivo_warm wall=33.25 sum=60.86; highs_warm wall=20.11 sum=33.23; zp_qwarm=0.0%
#     -> GPU batch when a GPU is present (CPU warm loses; research: GPU dominates iid stacks)

RANK_CPU_MAX = 3
# Brief "~20%"; one_crude/cargo_menu grids use lo/hi=±0.20 (stacks.py defaults).
MAX_REL_MOVE_CPU = 0.20
# Numerical rank: singular values below this fraction of σ_max are treated as zero.
SVD_RANK_TOL = 1e-6
# Cap k-means clusters for mixed stacks (W03 brief: k <= 8).
MIXED_CLUSTER_MAX = 8


@dataclass
class StackRoutingPolicy:
    """Stack-structure thresholds. Defaults cite W01 JSONL rows above."""
    rank_cpu_max: int = RANK_CPU_MAX
    max_rel_move_cpu: float = MAX_REL_MOVE_CPU
    svd_rank_tol: float = SVD_RANK_TOL
    mixed_cluster_max: int = MIXED_CLUSTER_MAX
    prefer_gpu: bool = True
    # No-GPU fallback: W03 brief max_workers = min(cores-2, 6); RAM lane uses threads=1.
    max_workers_cap: int = 6
    leave_cores: int = 2


@dataclass
class StackProfile:
    """Structural fingerprint of a what-if stack (deltas vs a fixed base matrix)."""
    n_cases: int
    n_changed_coords: int
    delta_rank: int
    max_rel_move: float
    only_costs: bool
    only_bounds: bool
    costs_and_bounds: bool
    is_one_factor_ray: bool          # numerical rank == 1 (pure 1-D parametric)
    estimated_zero_pivot_frac: float | None = None
    changed_coord_kinds: dict = field(default_factory=dict)  # {"c": n, "lx": n, ...}
    singular_values: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _rel(new: float, old: float) -> float:
    den = max(abs(old), 1.0)
    return abs(new - old) / den


def _case_sparse_changes(base, case) -> dict[str, float]:
    """Map 'kind:index' -> relative move for a name-keyed Case or index CaseDelta."""
    out: dict[str, float] = {}
    # CaseDelta (index-based)
    if hasattr(case, "c_idx") and hasattr(case, "c_val"):
        for j, v in zip(case.c_idx, case.c_val):
            out[f"c:{int(j)}"] = _rel(float(v), float(base.c[int(j)]))
        for j, v in zip(getattr(case, "lx_idx", ()), getattr(case, "lx_val", ())):
            out[f"lx:{int(j)}"] = _rel(float(v), float(base.lx[int(j)]))
        for j, v in zip(getattr(case, "ux_idx", ()), getattr(case, "ux_val", ())):
            out[f"ux:{int(j)}"] = _rel(float(v), float(base.ux[int(j)]))
        for i, v in zip(getattr(case, "lc_idx", ()), getattr(case, "lc_val", ())):
            out[f"lc:{int(i)}"] = _rel(float(v), float(base.lc[int(i)]))
        for i, v in zip(getattr(case, "uc_idx", ()), getattr(case, "uc_val", ())):
            out[f"uc:{int(i)}"] = _rel(float(v), float(base.uc[int(i)]))
        return out
    # Case (name-keyed)
    rn, cn = base.names()
    ci = {c: j for j, c in enumerate(cn)}
    ri = {r: i for i, r in enumerate(rn)}
    for k, v in (getattr(case, "cost", None) or {}).items():
        j = ci[k]
        # Case.cost is user's sense; base.c is internal minimise sense
        new_c = base.obj_sign * float(v)
        out[f"c:{j}"] = _rel(new_c, float(base.c[j]))
    for k, v in (getattr(case, "col_lo", None) or {}).items():
        j = ci[k]
        out[f"lx:{j}"] = _rel(float(v), float(base.lx[j]))
    for k, v in (getattr(case, "col_hi", None) or {}).items():
        j = ci[k]
        out[f"ux:{j}"] = _rel(float(v), float(base.ux[j]))
    for k, v in (getattr(case, "row_lo", None) or {}).items():
        i = ri[k]
        out[f"lc:{i}"] = _rel(float(v), float(base.lc[i]))
    for k, v in (getattr(case, "row_hi", None) or {}).items():
        i = ri[k]
        out[f"uc:{i}"] = _rel(float(v), float(base.uc[i]))
    return out


def _estimate_zero_pivot_frac(base, deltas, base_sol) -> float | None:
    """Fraction of cases whose cost moves stay inside base ranging half-widths (cost-only)."""
    if base_sol is None:
        return None
    try:
        from .ranging import cost_ranging
        ranges = {d["column"]: d for d in cost_ranging(base, base_sol)}
    except Exception:  # noqa: BLE001
        return None
    _, cn = base.names()
    n_ok = 0
    for case in deltas:
        ch = _case_sparse_changes(base, case)
        cost_keys = [k for k in ch if k.startswith("c:")]
        if not cost_keys:
            # bound-only: no cost-ranging estimate
            continue
        ok = True
        for key in cost_keys:
            j = int(key.split(":")[1])
            name = cn[j]
            rd = ranges.get(name)
            if rd is None:
                ok = False
                break
            # rebuild absolute new cost
            if hasattr(case, "c_idx"):
                # CaseDelta stores internal c
                idx = list(case.c_idx)
                if j not in idx:
                    continue
                new_c = float(case.c_val[idx.index(j)])
            else:
                new_c = base.obj_sign * float(case.cost[name])
            user_c = base.obj_sign * new_c
            lo, hi = rd["cost_min"], rd["cost_max"]
            if not (lo - 1e-12 <= user_c <= hi + 1e-12):
                ok = False
                break
        if ok and cost_keys:
            n_ok += 1
    n_cost_cases = sum(1 for d in deltas if any(k.startswith("c:") for k in _case_sparse_changes(base, d)))
    if n_cost_cases == 0:
        return None
    return float(n_ok) / float(n_cost_cases)


def classify_stack(base, deltas, base_sol=None,
                   policy: StackRoutingPolicy | None = None) -> StackProfile:
    """Fingerprint a stack: changed coords, delta-matrix rank, move size, cost/bound mix.

    `deltas` may be `CaseDelta` list (W01 generators) or name-keyed `Case` list (`solve_cases`).
    Optional `base_sol` (simplex vertex) enables ranging-based zero-pivot fraction estimate.
    """
    policy = policy or StackRoutingPolicy()
    S = len(deltas)
    if S == 0:
        return StackProfile(n_cases=0, n_changed_coords=0, delta_rank=0, max_rel_move=0.0,
                            only_costs=True, only_bounds=True, costs_and_bounds=False,
                            is_one_factor_ray=False)

    per_case = [_case_sparse_changes(base, d) for d in deltas]
    all_keys: list[str] = sorted({k for ch in per_case for k in ch})
    kinds = {"c": 0, "lx": 0, "ux": 0, "lc": 0, "uc": 0}
    for k in all_keys:
        kinds[k.split(":")[0]] = kinds.get(k.split(":")[0], 0) + 1

    n_coords = len(all_keys)
    max_rel = 0.0
    if n_coords == 0:
        D = np.zeros((0, S))
        rank = 0
        sv: list[float] = []
    else:
        key_index = {k: i for i, k in enumerate(all_keys)}
        D = np.zeros((n_coords, S), dtype=np.float64)
        for s, ch in enumerate(per_case):
            for k, rel in ch.items():
                D[key_index[k], s] = rel
                if rel > max_rel:
                    max_rel = rel
        # SVD of the k x S block (changed coords × cases)
        if min(D.shape) == 0:
            rank, sv = 0, []
        else:
            try:
                svals = np.linalg.svd(D, compute_uv=False)
            except np.linalg.LinAlgError:
                svals = np.array([])
            if svals.size == 0:
                rank, sv = 0, []
            else:
                cutoff = policy.svd_rank_tol * float(svals[0])
                rank = int(np.sum(svals > cutoff))
                sv = [float(x) for x in svals[: min(8, svals.size)]]

    has_c = kinds.get("c", 0) > 0
    has_b = any(kinds.get(t, 0) > 0 for t in ("lx", "ux", "lc", "uc"))
    only_c = has_c and not has_b
    only_b = has_b and not has_c
    both = has_c and has_b
    zp = _estimate_zero_pivot_frac(base, deltas, base_sol)

    return StackProfile(
        n_cases=S,
        n_changed_coords=n_coords,
        delta_rank=rank,
        max_rel_move=float(max_rel),
        only_costs=only_c,
        only_bounds=only_b,
        costs_and_bounds=both,
        is_one_factor_ray=(rank == 1),
        estimated_zero_pivot_frac=zp,
        changed_coord_kinds=kinds,
        singular_values=sv,
    )


def _cluster_case_indices(base, deltas, k: int, seed: int = 0) -> list[list[int]]:
    """Cheap Lloyd k-means on per-case relative-delta vectors; k capped by mixed_cluster_max."""
    S = len(deltas)
    if S == 0:
        return []
    per_case = [_case_sparse_changes(base, d) for d in deltas]
    keys = sorted({kk for ch in per_case for kk in ch})
    if not keys:
        return [list(range(S))]
    X = np.zeros((S, len(keys)), dtype=np.float64)
    ki = {kk: i for i, kk in enumerate(keys)}
    for s, ch in enumerate(per_case):
        for kk, rel in ch.items():
            X[s, ki[kk]] = rel
    k = max(1, min(k, S, MIXED_CLUSTER_MAX))
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(S, size=k, replace=False)].copy()
    labels = np.zeros(S, dtype=np.int64)
    for _ in range(16):
        # assign
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        labels = np.argmin(d2, axis=1)
        # update
        new_c = centres.copy()
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_c[j] = X[rng.integers(0, S)]
            else:
                new_c[j] = X[mask].mean(0)
        if np.allclose(new_c, centres):
            centres = new_c
            break
        centres = new_c
    return [[i for i in range(S) if labels[i] == j] for j in range(k) if np.any(labels == j)]


def _cpu_workers(policy: StackRoutingPolicy, threads: int = 1) -> int:
    # RAM / W03 lane: threads=1 means a single warm chain (no process fan-out).
    if threads <= 1:
        return 1
    cores = os.cpu_count() or 2
    return max(1, min(policy.max_workers_cap, max(1, cores - policy.leave_cores)))


def route_stack(profile: StackProfile, *, base=None, deltas=None,
                engine: str = "auto", backend: str = "auto",
                exact_duals: bool = False, threads: int = 1,
                policy: StackRoutingPolicy | None = None,
                lp_policy: RoutingPolicy | None = None) -> dict:
    """Pick cpu_warm / cpu_warm_ray / gpu_batch / mixed from a StackProfile.

    Returns a decision dict with `route`, `reason`, `profile`, plus execution fields
    (`engine`, `backend`, `order`, `clusters`, `max_workers`) for `solve_cases`.
    Forced `engine != auto` still wins (same contract as single-LP `route`).
    """
    policy = policy or StackRoutingPolicy()
    lp_policy = lp_policy or RoutingPolicy(prefer_gpu=policy.prefer_gpu)
    S = profile.n_cases
    has_gpu = gpu_available() if backend in ("auto", "cupy", "gpu", "cuda") else False
    if backend in ("cupy", "gpu", "cuda") and not has_gpu:
        raise RuntimeError("GPU backend requested but no CUDA device / CuPy found")
    want_gpu = has_gpu and policy.prefer_gpu and backend not in ("numpy", "cpu")

    def pack(route_name: str, eng: str, dev: str, why: str, **extra) -> dict:
        out = {
            "route": route_name,
            "reason": why,
            "profile": profile.to_dict(),
            "engine": eng,
            "backend": dev,
            "gpu_available": has_gpu,
            "batch": S,
            "order": extra.pop("order", "nn"),
            "clusters": extra.pop("clusters", None),
            "max_workers": extra.pop("max_workers", _cpu_workers(policy, threads)),
            "exact_duals": exact_duals,
        }
        out.update(extra)
        return out

    if engine != "auto":
        # Forced path: reuse single-LP device pick; record as forced route.
        r = route(base, S, engine, backend, lp_policy) if base is not None else {
            "engine": engine,
            "backend": "cupy" if (want_gpu and engine in ("pdhg", "pdhg-rf", "pdqp")) else "numpy",
            "reason": "engine chosen by the caller",
            "gpu_available": has_gpu,
        }
        return pack("forced", r["engine"], r["backend"],
                    f"forced engine={engine} ({r.get('reason', 'caller')})", order=None)

    # --- structure policy ---
    low_rank = profile.delta_rank <= policy.rank_cpu_max
    small_move = profile.max_rel_move <= policy.max_rel_move_cpu + 1e-12
    workers = _cpu_workers(policy, threads)

    # Pure 1-D ray (one_crude / cargo_menu / one_unit grids): CPU warm, ordered on the factor.
    # W01 L3/one_crude/S64 & L3/cargo_menu/S64: qenivo_warm ≈ highs_warm, zp≈58%.
    if profile.is_one_factor_ray and (profile.only_costs or profile.only_bounds) and small_move:
        single = route(base, 1, "auto", "numpy", lp_policy) if base is not None else {
            "engine": "simplex", "reason": "default vertex engine"}
        return pack(
            "cpu_warm_ray", single["engine"], "numpy",
            f"1-D ray rank=1, max_rel={profile.max_rel_move:.3f}<={policy.max_rel_move_cpu}: "
            f"CPU warm chain ordered on the factor "
            f"(W01 L3/one_crude|cargo_menu|one_unit S64; {single.get('reason', '')})",
            order="factor", max_workers=workers,
        )

    # Mixed costs+bounds (factor_mixed): cluster then route each cluster.
    # W01 L3/factor_mixed/S64: highs_warm 7.55 << qenivo_warm 13.83.
    if profile.costs_and_bounds:
        k = min(policy.mixed_cluster_max, max(2, profile.delta_rank), max(1, S))
        clusters = None
        if base is not None and deltas is not None:
            clusters = _cluster_case_indices(base, deltas, k)
        why = (f"costs+bounds (rank={profile.delta_rank}): mixed clusters k<={k}; "
               f"W01 L3/factor_mixed/S64 highs_warm faster than qenivo_warm on CPU")
        if want_gpu:
            # GPU for the diverse remainder; clusters recorded for later per-cluster routing.
            return pack("mixed", "pdhg", "cupy", why + "; GPU present → batch + optional crossover",
                        order="cluster", clusters=clusters, max_workers=workers)
        return pack("mixed", "simplex" if base is None else route(base, 1, "auto", "numpy", lp_policy)["engine"],
                    "numpy", why + "; no GPU → CPU warm per cluster",
                    order="cluster", clusters=clusters, max_workers=workers)

    # Low-rank cost (or bound) stack within ~20%: CPU warm.
    # Covers factor_price (rank~3) — W01 L3/factor_price/S64 qenivo_warm 6.03 vs highs_warm 6.37.
    if low_rank and small_move and (profile.only_costs or profile.only_bounds):
        single = route(base, 1, "auto", "numpy", lp_policy) if base is not None else {
            "engine": "simplex", "reason": "default vertex engine"}
        return pack(
            "cpu_warm", single["engine"], "numpy",
            f"rank={profile.delta_rank}<={policy.rank_cpu_max}, max_rel={profile.max_rel_move:.3f}: "
            f"CPU warm chain (W01 L3/factor_price/S64 warm-competitive; {single.get('reason', '')})",
            order="nn", max_workers=workers,
        )

    # High-rank / large moves / independent: GPU batch when available.
    # W01 L3/independent/S64: qenivo_warm 33.25 vs highs_warm 20.11 — GPU lane.
    if want_gpu:
        return pack(
            "gpu_batch", "pdhg", "cupy",
            f"high-rank/large-move stack (rank={profile.delta_rank}, coords={profile.n_changed_coords}, "
            f"max_rel={profile.max_rel_move:.3f}): GPU batch to 1e-4"
            + ("; exact-duals crossover per case" if exact_duals else "")
            + " (W01 L3/independent/S64 CPU warm loses)",
            order=None, max_workers=1,
        )

    # No GPU: CPU warm everywhere (W03 brief).
    single = route(base, 1, "auto", "numpy", lp_policy) if base is not None else {
        "engine": "simplex", "reason": "default vertex engine"}
    return pack(
        "cpu_warm", single["engine"], "numpy",
        f"no GPU: CPU warm chain (rank={profile.delta_rank}, coords={profile.n_changed_coords}); "
        f"max_workers={workers} (W01 independent would prefer GPU)",
        order="nn", max_workers=workers,
    )
