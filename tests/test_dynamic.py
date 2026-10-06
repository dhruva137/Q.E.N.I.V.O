"""Dynamic engine: plan ticker, Greeks (finite-difference checked), stalled-lane hand-off, speculation."""
import numpy as np
import pytest

from qenivo.engines.simplex import solve_simplex
from qenivo.models.refinery import refinery_level
from qenivo.workload import dynamic as D
from qenivo.workload.ranging import cost_ranging, rhs_ranging
from qenivo.workload.stacks import groups, make_stack

# Finite differences: phi comes from cold vertex solves, exact to ~1e-12 relative, so a difference
# quotient with step h carries an error of about 1e-12 |phi| / h. With |phi| ~ 5e5 and h = 1e-5 that
# is ~1e-2 in the worst case; measured ~3e-6. The tests demand
#     |FD - delta| <= 1e-7 (1 + |delta|) + 1e-11 (1 + |phi|) / h.
H = 1e-5


@pytest.fixture(scope="module")
def p1():
    return refinery_level(1)


@pytest.fixture(scope="module")
def crude_dir(p1):
    g = groups(p1)
    sel = g["crude_cols"][g["crude_k"] == 0]
    dc = np.zeros(p1.n)
    dc[sel] = p1.c[sel]
    return sel, dc


def phi(p) -> float:
    r = solve_simplex(p, tol=1e-9)
    assert r["status"] == "optimal"
    return p.obj_sign * float(p.c @ r["x"] + p.c0)


def fd_tol(delta, value, h=H):
    return 1e-7 * (1 + abs(delta)) + 1e-11 * (1 + abs(value)) / h


def close(a, b, rel=1e-8):
    return abs(a - b) <= rel * (1 + abs(a) + abs(b))


# ------------------------------------------------------------------------------ basis factor
@pytest.mark.parametrize("level", [1, 2])
def test_basis_factor_solves(level):
    p = refinery_level(level)
    tk = D.PlanTicker(p)
    f = tk.held.factor
    B = tk.Az[:, tk.held.basis].toarray()
    rng = np.random.default_rng(0)
    b = rng.standard_normal(p.m)
    assert np.allclose(B @ f.ftran(b), b, atol=1e-9)
    assert np.allclose(B.T @ f.btran(b), b, atol=1e-9)
    assert f.nucleus < p.m                     # peeling did reduce the dense part


# ------------------------------------------------------------------------------ ticker
def test_ticker_walk_matches_cold(p1, crude_dir):
    sel, _ = crude_dir
    g = groups(p1)
    rows = g["cap_rows"]["distillation_cap"]
    tk = D.PlanTicker(p1)
    rng = np.random.default_rng(7)
    u, v = 0.0, 0.0
    for k in range(40):
        u = float(np.clip(u + rng.normal(0, 0.03), -0.3, 0.3))
        if k % 4 == 3:
            v = float(np.clip(v + rng.normal(0, 0.05), -0.3, 0.3))
        s = tk.tick(c=(sel, p1.c[sel] * (1 + u)), uc=(rows, p1.uc[rows] * (1 + v)))
        assert s.verdict == "optimal"
        assert s.residuals["max_rel"] <= 1e-9
        assert close(s.objective, phi(tk.current()))
    st = tk.stats()
    assert st["instant"] > 0 and st["warm"] > 0
    assert st["instant"] + st["warm"] + st["cold"] == 40
    assert st["instant_rejected"] == 0


def test_cost_and_rhs_ranges_match_ranging(p1):
    sol = D.PlanTicker(p1).first
    tk = D.PlanTicker(p1, start=sol.extra["basis"])
    sol.extra["basis"] = tk.held.statuses()
    cr = cost_ranging(p1, sol)
    for j in range(p1.n):
        a, b = tk.cost_range(j)
        assert np.isclose(a, cr[j]["cost_min"], rtol=1e-7, atol=1e-7), j
        assert np.isclose(b, cr[j]["cost_max"], rtol=1e-7, atol=1e-7), j
    rr = rhs_ranging(p1, sol)
    checked = 0
    for i, r in enumerate(rr):
        st = tk.held.statuses()["row_statuses"][i]
        if not r["binding"] or p1.lc[i] == p1.uc[i]:
            continue
        a, b = tk.row_limit_range(i, "upper" if st == "at_upper" else "lower")
        assert np.isclose(a, r["limit_min"], rtol=1e-7, atol=1e-7), i
        assert np.isclose(b, r["limit_max"], rtol=1e-7, atol=1e-7), i
        checked += 1
    assert checked > 5


def _interior_column(tk, p):
    """A basic column with a finite, non-degenerate cost range."""
    for j in tk.held.basis[tk.held.basis < p.n]:
        a, b = tk.cost_range(j)
        if np.isfinite(a) and np.isfinite(b) and b - a > 1e-3 * (1 + abs(p.c[j])):
            return int(j), a, b
    raise AssertionError("no interior column")


def test_tick_inside_range_is_instant_and_exact(p1):
    tk = D.PlanTicker(p1)
    j, a, b = _interior_column(tk, p1)
    x0 = tk.first.x.copy()
    s = tk.tick(c={j: a + 0.7 * (b - a)})
    assert s.engine["path"] == "instant" and s.verdict == "optimal"
    assert np.allclose(s.x, x0, atol=1e-9)                # cost tick inside the range: vertex unchanged
    assert close(s.objective, phi(tk.current()))


def test_tick_outside_range_warm_resolves(p1):
    tk = D.PlanTicker(p1)
    j, a, b = _interior_column(tk, p1)
    s = tk.tick(c={j: b + 0.5 * (b - a) + 1.0})
    assert s.engine["path"] == "warm" and s.verdict == "optimal"
    assert s.engine["region"].startswith("dual")
    assert tk.counts["instant_rejected"] == 0              # the region test itself saw it, not the safety net
    assert close(s.objective, phi(tk.current()))
    # a right-hand-side tick past its range is caught by the primal side of the test
    tk2 = D.PlanTicker(p1)
    rows = [i for i, st in enumerate(tk2.held.statuses()["row_statuses"]) if st == "at_upper"]
    for i in rows:
        lo, hi = tk2.row_limit_range(i, "upper")
        if np.isfinite(hi) and hi - lo > 1e-6:
            break
    s = tk2.tick(uc={i: hi + 0.5 * (hi - lo) + 1.0})
    assert s.engine["path"] in ("warm", "cold") and s.verdict == "optimal"
    assert s.engine["region"].startswith("primal")
    assert close(s.objective, phi(tk2.current()))


def test_forced_out_of_range_instant_is_caught(p1):
    """If the region test is wrong (forced to say 'inside'), the KKT check rejects the answer."""
    tk = D.PlanTicker(p1)
    j, a, b = _interior_column(tk, p1)
    tk._region_hook = lambda inside: True
    s = tk.tick(c={j: b + 0.5 * (b - a) + 1.0})
    assert tk.counts["instant_rejected"] == 1
    assert s.engine["path"] == "warm" and s.verdict == "optimal"
    assert close(s.objective, phi(tk.current()))


def test_tampered_instant_answer_is_not_optimal(p1, monkeypatch):
    tk = D.PlanTicker(p1)
    j, a, b = _interior_column(tk, p1)
    real = tk.held.factor.btran
    monkeypatch.setattr(tk.held.factor, "btran", lambda c: real(c) * 1.01)   # corrupt the duals
    q = D.with_data(tk.current(), c=np.where(np.arange(p1.n) == j, a + 0.5 * (b - a), p1.c))
    z, y, why = tk._instant(q, True, False)
    from qenivo.api import _finish
    if z is not None:
        cand = _finish(q, "optimal", z[:p1.n], y, tk.tol, {"engine": "plan_ticker"})
        assert cand.verdict == "not_proven"
    s = tk.tick_problem(q)
    assert s.verdict == "optimal" and s.engine["path"] != "instant"
    assert close(s.objective, phi(q))


# ------------------------------------------------------------------------------ Greeks
@pytest.fixture(scope="module")
def cost_greeks(p1, crude_dir):
    _, dc = crude_dir
    return D.plan_greeks(p1, np.linspace(-0.5, 0.5, 11), dc=dc)


def _phi_t(p, t, **ray):
    return phi(D._at(p, t, ray.get("dc"), ray.get("dlc"), ray.get("duc"), None, None))


def test_cost_greeks_central_differences(p1, crude_dir, cost_greeks):
    _, dc = crude_dir
    G = cost_greeks
    assert G["breakpoints"] and not G["notes"]
    for c in G["cases"]:
        assert c["verdict"] == "optimal"
        if min(abs(c["t"] - b["t"]) for b in G["breakpoints"]) <= 2 * H:
            continue
        fd = (_phi_t(p1, c["t"] + H, dc=dc) - _phi_t(p1, c["t"] - H, dc=dc)) / (2 * H)
        assert c["delta"] is not None
        assert abs(fd - c["delta"]) <= fd_tol(c["delta"], c["value"]), (c["t"], fd, c["delta"])
        assert abs(c["delta"] - float(dc @ c["x"]) * p1.obj_sign) <= 1e-9 * (1 + abs(c["delta"]))


def test_cost_greeks_one_sided_at_breakpoints(p1, crude_dir, cost_greeks):
    _, dc = crude_dir
    for b in cost_greeks["breakpoints"]:
        t = b["t"]
        f0 = _phi_t(p1, t, dc=dc)
        right = (_phi_t(p1, t + H, dc=dc) - f0) / H
        left = (f0 - _phi_t(p1, t - H, dc=dc)) / H
        assert abs(right - b["slope_right"]) <= fd_tol(b["slope_right"], f0)
        assert abs(left - b["slope_left"]) <= fd_tol(b["slope_left"], f0)
        assert abs(b["gamma"]) > 1e3 * fd_tol(b["slope_left"], f0)       # a real kink, not noise
        assert b["gamma"] * p1.obj_sign < 0                                # cost ray: phi concave (min form)
    # a case placed exactly on a breakpoint reports both sides and no derivative
    b = cost_greeks["breakpoints"][0]
    G = D.plan_greeks(p1, [b["t"] - 0.01, b["t"], b["t"] + 0.01], dc=dc)
    mid = G["cases"][1]
    assert mid["at_breakpoint"] and mid["delta"] is None
    assert close(mid["delta_minus"], b["slope_left"]) and close(mid["delta_plus"], b["slope_right"])


def test_cost_gamma_matches_second_difference(p1, crude_dir, cost_greeks):
    _, dc = crude_dir
    G = cost_greeks
    h = 0.05
    for t in [G["breakpoints"][0]["t"] + 0.01, G["breakpoints"][-1]["t"] - 0.02, 0.45]:
        f = [_phi_t(p1, t + s * h, dc=dc) for s in (-1, 0, 1)]
        fd2 = (f[2] - 2 * f[1] + f[0]) / h ** 2
        pred = D.second_difference(G, t, h)
        assert abs(fd2 - pred) <= 1e-6 * (1 + abs(pred)) + 1e-11 * (1 + abs(f[1])) / h ** 2, (t, fd2, pred)


def test_rhs_greeks_finite_differences(p1):
    g = groups(p1)
    rows = g["cap_rows"]["distillation_cap"]
    duc = np.zeros(p1.m)
    duc[rows] = p1.uc[rows]
    G = D.plan_greeks(p1, np.linspace(-0.05, 0.05, 6), duc=duc)
    assert not G["notes"]
    assert all(b["gamma"] * p1.obj_sign > 0 for b in G["breakpoints"])   # rhs ray: phi convex
    n_checked = 0
    for c in G["cases"]:
        if G["breakpoints"] and min(abs(c["t"] - b["t"]) for b in G["breakpoints"]) <= 2 * H:
            continue
        fd = (_phi_t(p1, c["t"] + H, duc=duc) - _phi_t(p1, c["t"] - H, duc=duc)) / (2 * H)
        assert abs(fd - c["delta"]) <= fd_tol(c["delta"], c["value"]), (c["t"], fd, c["delta"])
        n_checked += 1
    assert n_checked >= 4
    for t in (-0.03, 0.01, 0.04):
        h = 0.01
        f = [_phi_t(p1, t + s * h, duc=duc) for s in (-1, 0, 1)]
        pred = D.second_difference(G, t, h)
        assert abs((f[2] - 2 * f[1] + f[0]) / h ** 2 - pred) <= 1e-6 * (1 + abs(pred)) + 1e-11 * (1 + abs(f[1])) / h ** 2


def test_coordinate_deltas_finite_differences(p1):
    """dphi/dc_j = x_j, dphi/d(row limit) = y_i, dphi/d(column bound) = reduced cost, where the
    coordinate is interior to its range (elsewhere only one-sided derivatives exist)."""
    tk = D.PlanTicker(p1)
    s = tk.first
    v0 = s.objective
    rng = np.random.default_rng(3)
    done = 0
    for j in rng.permutation(p1.n):
        a, b = tk.cost_range(j)
        c = p1.c[j]
        if not (c - a > 2 * H and b - c > 2 * H):
            continue
        e = np.zeros(p1.n)
        e[j] = 1.0
        fd = (_phi_t(p1, H, dc=e) - _phi_t(p1, -H, dc=e)) / (2 * H)
        assert abs(fd - s.x[j]) <= fd_tol(s.x[j], v0), j
        done += 1
        if done >= 8:
            break
    assert done >= 8
    done = 0
    rs = tk.held.statuses()["row_statuses"]
    for i in range(p1.m):
        if rs[i] not in ("at_upper", "at_lower") or p1.lc[i] == p1.uc[i]:
            continue
        side = "upper" if rs[i] == "at_upper" else "lower"
        lo, hi = tk.row_limit_range(i, side)
        lim = (p1.uc if side == "upper" else p1.lc)[i]
        if not (lim - lo > 2 * H and hi - lim > 2 * H):
            continue
        e = np.zeros(p1.m)
        e[i] = 1.0
        ray = {"duc": e} if side == "upper" else {"dlc": e}
        fd = (_phi_t(p1, H, **ray) - _phi_t(p1, -H, **ray)) / (2 * H)
        assert abs(fd - s.y[i]) <= fd_tol(s.y[i], v0), i
        done += 1
    assert done >= 3
    # column bound: a nonbasic column whose bound can move both ways inside the basis's range
    rc = s.reduced_costs
    cs = tk.held.statuses()["col_statuses"]
    done = 0
    for j in range(p1.n):
        if cs[j] not in ("at_lower", "at_upper") or p1.lx[j] == p1.ux[j] or abs(rc[j]) <= 1e-6:
            continue
        key = "ux" if cs[j] == "at_upper" else "lx"
        e = np.zeros(p1.n)
        e[j] = 1.0
        a, b = tk.ray_interval(**{"d" + key: e})
        if not (a < -2 * H and b > 2 * H):
            continue
        vec = getattr(p1, key)
        fd = (phi(D.with_data(p1, **{key: vec + H * e})) - phi(D.with_data(p1, **{key: vec - H * e}))) / (2 * H)
        assert abs(fd - rc[j]) <= fd_tol(rc[j], v0), j
        done += 1
    assert done >= 3


def test_greeks_reject_mixed_direction(p1, crude_dir):
    _, dc = crude_dir
    with pytest.raises(ValueError):
        D.plan_greeks(p1, [0.0, 0.1], dc=dc, duc=np.ones(p1.m))


# ------------------------------------------------------------------------------ hand-off
def test_stalled_lanes_handed_to_crossover(p1):
    from qenivo.engines import pdhg
    S = 5
    probs = [d.apply(p1) for d in make_stack(p1, "factor_price", S, seed=3)]
    st = lambda k: np.stack([getattr(q, k) for q in probs], 1)  # noqa: E731
    br = pdhg.solve_batch(p1.A, st("c"), st("lc"), st("uc"), st("lx"), st("ux"), p1.c0,
                          pdhg.PDHGOptions(tol=1e-3, time_limit=60), "numpy")
    out = D.handoff_stalled(probs, br.x, br.y, range(S), tol=1e-9)
    assert sorted(out) == list(range(S))
    srcs = [s.engine["warm_from"] for s in out.values()]
    assert srcs.count(None) == 1                       # only the first case lacks a neighbour basis
    for i, s in out.items():
        assert s.verdict == "optimal" and s.residuals["max_rel"] <= 1e-9
        assert s.engine["engine"] == "crossover" and 1e-6 < s.engine["first_order_max_rel"] <= 1e-2
        assert close(s.objective, phi(probs[i]))
    # with a finished case supplied, every lane starts from a basis, the nearest one first
    base = solve_simplex(probs[0], tol=1e-9)
    fin = {0: {"col_statuses": base["col_statuses"], "row_statuses": base["row_statuses"]}}
    out2 = D.handoff_stalled(probs, br.x, br.y, range(1, S), finished=fin, tol=1e-9)
    assert all(s.engine["warm_from"] is not None for s in out2.values())
    assert all(s.verdict == "optimal" for s in out2.values())


def test_handoff_passes_the_neighbour_basis(p1, monkeypatch):
    import importlib
    X = importlib.import_module("qenivo.engines.crossover")
    seen = []
    real = X.crossover

    def spy(prob, x, y, **kw):
        seen.append(kw.get("start"))
        return real(prob, x, y, **kw)

    monkeypatch.setattr(X, "crossover", spy)
    probs = [d.apply(p1) for d in make_stack(p1, "one_crude", 4)]
    sols = [solve_simplex(q, tol=1e-9) for q in probs]
    X_ = np.stack([s["x"] for s in sols], 1) * (1 + 1e-4)          # near-optimal points, not exact
    Y_ = np.stack([s["y"] for s in sols], 1)
    fin = {0: {"col_statuses": sols[0]["col_statuses"], "row_statuses": sols[0]["row_statuses"]}}
    out = D.handoff_stalled(probs, X_, Y_, [1, 2, 3], finished=fin, tol=1e-9)
    assert len(seen) == 3 and all(s is not None for s in seen)
    first = min(out, key=lambda i: np.linalg.norm(probs[i].c - probs[0].c))
    assert out[first].engine["warm_from"] == 0
    assert seen[0]["col_statuses"] == fin[0]["col_statuses"]
    for i, s in out.items():
        assert s.verdict == "optimal" and close(s.objective, phi(probs[i]))


def test_far_point_goes_to_warm_simplex(p1):
    probs = [d.apply(p1) for d in make_stack(p1, "one_crude", 3)]
    X = np.zeros((p1.n, 3))
    Y = np.zeros((p1.m, 3))
    out = D.handoff_stalled(probs, X, Y, [0, 1, 2], tol=1e-9)
    for i, s in out.items():
        assert s.engine["engine"] == "simplex" and s.verdict == "optimal"
        assert close(s.objective, phi(probs[i]))


# ------------------------------------------------------------------------------ speculation
def _lattice(p, sel, h=0.02):
    return lambda k: D.with_data(p, c=np.where(np.isin(np.arange(p.n), sel), p.c * (1 + k * h), p.c))


@pytest.mark.parametrize("background", [False, True])
def test_speculative_presolve_reuses_exact_matches(p1, crude_dir, background):
    sel, _ = crude_dir
    lat = _lattice(p1, sel)
    pos = {"k": 0}

    def predict(tk, p):
        k = pos["k"]
        return [lat(k - 1), lat(k + 1)]

    spec = D.Speculator(predict=predict, background=background)
    tk = D.PlanTicker(lat(0), speculator=spec)
    rng = np.random.default_rng(5)
    for _ in range(30):
        pos["k"] = int(np.clip(pos["k"] + rng.choice([-1, 1]), -25, 25))
        if background:
            assert spec.wait_idle(timeout=30)
        else:
            spec.run_pending()
        s = tk.tick_problem(lat(pos["k"]))
        assert s.verdict == "optimal"
        assert close(s.objective, phi(lat(pos["k"])))
    spec.close()
    assert tk.counts["speculative"] > 0
    assert spec.stats["hits"] == tk.counts["speculative"]
    assert spec.stats["rejected"] == 0


def test_speculation_needs_bit_identical_data(p1, crude_dir):
    sel, _ = crude_dir
    lat = _lattice(p1, sel, h=0.2)
    spec = D.Speculator(predict=lambda tk, p: [lat(3)])
    tk = D.PlanTicker(lat(0), speculator=spec)
    tk.tick_problem(lat(0))
    spec.run_pending()
    assert spec.stats["solved"] == 1
    q = lat(3)
    q.c = np.nextafter(q.c, np.inf, where=np.isin(np.arange(p1.n), sel), out=q.c.copy())
    assert D.fingerprint(q) != D.fingerprint(lat(3))
    s = tk.tick_problem(q)
    assert s.engine["path"] != "speculative" and s.verdict == "optimal"
    s = tk.tick_problem(lat(3))
    assert s.engine["path"] in ("speculative", "instant")


def test_fingerprint_collision_is_not_reused(p1, crude_dir, monkeypatch):
    """Reuse needs equal data, not only an equal digest: a colliding fingerprint must miss."""
    sel, _ = crude_dir
    lat = _lattice(p1, sel, h=0.2)
    monkeypatch.setattr(D, "fingerprint", lambda p: "same")
    spec = D.Speculator(predict=lambda tk, p: [lat(3)])
    tk = D.PlanTicker(lat(0), speculator=spec)
    tk.tick_problem(lat(0))
    spec.run_pending()
    s = tk.tick_problem(lat(-3))
    assert spec.stats["misses"] == 1                      # the store was consulted ...
    assert s.engine["path"] != "speculative" and spec.stats["hits"] == 0   # ... and refused
    assert close(s.objective, phi(lat(-3)))


# ------------------------------------------------------------------ adaptive policy
def test_adaptive_policy_follows_measured_latency(p1):
    """The cheaper mode by mean measured tick latency wins; the other is retried every PROBE_EVERY ticks."""
    tk = D.PlanTicker(p1)
    assert tk.mode == "instant"
    for _ in range(5):
        tk._learn("instant", 0.20)      # instant ticks slow (mostly leaving the region)
        tk._learn("warm", 0.02)
    assert tk.mode == "warm" and tk.counts["mode_switches"] == 1
    picks = [tk._pick_mode() for _ in range(2 * tk.PROBE_EVERY + 2)]
    assert picks.count("instant") == 2 and picks.count("warm") == len(picks) - 2   # periodic probes only
    for _ in range(40):
        tk._learn("instant", 0.001)     # ticks now stay inside the region
    assert tk.mode == "instant"


def test_warm_only_mode_answers_are_certified_and_equal(p1, crude_dir):
    """Forcing warm-only mode changes the path, never the answer or its certificate."""
    _, dc = crude_dir
    a, b = D.PlanTicker(p1, adaptive=False), D.PlanTicker(p1)
    b.mode, b._lat = "warm", {"instant": 1.0, "warm": 1e-9}          # stay in warm-only mode
    b._since_other = -10 ** 9                                          # and never probe
    for t in np.linspace(-0.4, 0.4, 9):
        c = p1.c + t * dc
        sa, sb = a.tick(c=c), b.tick(c=c)
        assert sa.verdict == sb.verdict == "optimal"
        assert sb.engine["mode"] == "warm" and sb.engine["path"] in ("warm", "cold")
        assert sb.objective == pytest.approx(sa.objective, rel=1e-9, abs=1e-9)


def test_factorisation_is_lazy_after_a_warm_solve(p1, crude_dir):
    """A basis reached by a warm solve is factorised only when an instant question needs it."""
    _, dc = crude_dir
    tk = D.PlanTicker(p1, adaptive=False)
    tk.tick(c=p1.c + 0.45 * dc)                                        # far enough to need a warm solve
    assert tk.held is not None and tk.held._factor is None
    tk.tick(c=p1.c + 0.45 * dc + 1e-9 * dc)                            # an instant question
    assert tk.held._factor is not None
