"""GPU checks: the CUDA path gives the CPU path's answers, and QENIVO's kernels match the vendor's."""
import numpy as np
import pytest

from qenivo.kernels.backend import gpu_available

pytestmark = pytest.mark.skipif(not gpu_available(), reason="no CUDA device")


def test_own_kernels_match_cpu_and_are_deterministic():
    import cupy as cp

    from qenivo.kernels.cuda_kernels import GPUMatrix
    from qenivo.models.refinery import refinery_lp
    A = refinery_lp(2, 4, 4, 6, seed=0).A
    G = GPUMatrix(A)
    for S in (1, 3, 64):
        X = np.random.default_rng(S).standard_normal((A.shape[1], S))
        Y1, Y2 = G.mv(cp.asarray(X)), G.mv(cp.asarray(X))
        assert bool(cp.all(Y1 == Y2))                           # bit-for-bit repeatable
        assert np.allclose(cp.asnumpy(Y1), A @ X, rtol=1e-12, atol=1e-10)


def test_gpu_pdhg_equals_cpu_pdhg():
    from qenivo import solve
    from qenivo.models.williams import PUBLISHED_OPTIMUM, williams_lp
    g = solve(williams_lp(), engine="pdhg", backend="cupy", tol=1e-8)
    c = solve(williams_lp(), engine="pdhg", backend="numpy", tol=1e-8)
    assert g.verdict == c.verdict == "optimal"
    assert abs(g.objective - PUBLISHED_OPTIMUM) <= 0.01
    assert abs(g.objective - c.objective) <= 1e-8 * (1 + abs(c.objective))   # both at tol 1e-8 (relative)


def test_gpu_batch_certifies_every_case():
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import pdhg
    from qenivo.models.refinery import refinery_lp
    p = refinery_lp(1, 4, 4, 4, seed=2)
    C = p.c[:, None] * np.random.default_rng(0).uniform(0.9, 1.1, (p.n, 16))
    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, pdhg.PDHGOptions(tol=1e-6), "cupy")
    for j in range(16):
        q = p.copy(); q.c = C[:, j].copy()
        assert br.status[j] == "optimal"
        assert kkt_residuals(q, br.x[:, j], br.y[:, j])["max_rel"] <= 1e-6 * 1.0001


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_float32_kernels_match_float64_and_are_deterministic():
    import cupy as cp
    import scipy.sparse as sps

    from qenivo.kernels.cuda_kernels import GPUMatrix
    A = sps.random(300, 200, density=0.05, random_state=1, format="csr")
    X = np.random.default_rng(2).standard_normal((200, 7))
    M64, M32 = GPUMatrix(A), GPUMatrix(A, dtype=np.float32)
    y64 = M64.mv(cp.asarray(X)).get()
    y32a = M32.mv(cp.asarray(X, dtype=cp.float32)).get()
    y32b = M32.mv(cp.asarray(X, dtype=cp.float32)).get()
    assert y32a.dtype == np.float32
    assert np.array_equal(y32a, y32b)
    assert np.max(np.abs(y32a - y64)) <= 1e-5 * (1 + np.max(np.abs(y64)))


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_refined_float32_on_gpu_reaches_1e8():
    from conftest import DATA
    from qenivo import read
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import refine
    p = read(DATA / "blend.mps")
    r = refine.solve(p, tol=1e-8, backend="cupy", time_limit=120)
    assert r.status[0] == "optimal"
    assert kkt_residuals(p, r.x[:, 0], r.y[:, 0])["max_rel"] <= 1e-8


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_batch_on_all_devices_certifies():
    from qenivo.engines.multigpu import solve_batch_devices
    from qenivo.models.refinery import LEVELS, refinery_lp
    p = refinery_lp(*LEVELS[1], seed=0)
    C = p.c[:, None] * np.random.default_rng(0).uniform(0.95, 1.05, (p.n, 6))
    r = solve_batch_devices(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, tol=1e-6)
    assert not r["errors"] and all(s == "optimal" for s in r["status"])


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_persistent_kernel_agrees_and_is_repeatable():
    from conftest import DATA
    from qenivo import read
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import pdhg
    p = read(DATA / "sc105.mps")
    a = pdhg.solve(p, tol=1e-7, backend="cupy", persistent="off", check_every=64)
    b = pdhg.solve(p, tol=1e-7, backend="cupy", persistent="on", persistent_check_every=64)
    c = pdhg.solve(p, tol=1e-7, backend="cupy", persistent="on", persistent_check_every=64)
    assert a.status[0] == b.status[0] == "optimal"
    assert a.iterations[0] == b.iterations[0]
    assert np.max(np.abs(a.x - b.x)) <= 1e-9 * (1 + np.max(np.abs(a.x)))
    assert np.array_equal(b.x, c.x) and np.array_equal(b.y, c.y)
    assert kkt_residuals(p, b.x[:, 0], b.y[:, 0])["max_rel"] <= 1e-7


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_persistent_kernel_batch_and_float32():
    from qenivo.engines import pdhg
    from qenivo.models.refinery import LEVELS, refinery_lp
    p = refinery_lp(*LEVELS[1], seed=0)
    C = p.c[:, None] * np.random.default_rng(0).uniform(0.95, 1.05, (p.n, 8))
    for dt in ("float64", "float32"):
        o = pdhg.PDHGOptions(tol=1e-4, persistent="on", dtype=dt, detect_infeasibility=False)
        br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, o, "cupy")
        assert all(s == "optimal" for s in br.status), (dt, br.status)


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_l2_tiles_match_untiled_within_tol():
    """A3: tiled and untiled batch answers agree at 1e-4 (no long hangs: tiny L1, S=8, tile=3)."""
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import pdhg
    from qenivo.models.refinery import LEVELS, refinery_lp
    p = refinery_lp(*LEVELS[1], seed=0)
    C = p.c[:, None] * np.random.default_rng(3).uniform(0.95, 1.05, (p.n, 8))
    base = pdhg.PDHGOptions(tol=1e-4, persistent="off", detect_infeasibility=False,
                            time_limit=120.0, diet_l2_tile=False)
    tiled = pdhg.PDHGOptions(**{**base.__dict__, "diet_l2_tile": True, "diet_tile_size": 3})
    a = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, base, "cupy")
    b = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, tiled, "cupy")
    assert a.status == b.status
    assert all(s == "optimal" for s in a.status)
    assert np.allclose(a.x, b.x, rtol=1e-3, atol=1e-3)
    assert np.allclose(a.y, b.y, rtol=1e-3, atol=1e-3)
    for j in range(8):
        q = p.copy(); q.c = C[:, j].copy()
        assert kkt_residuals(q, b.x[:, j], b.y[:, j])["max_rel"] <= 1e-4 * 1.0001


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_f32_host_cert_and_f64_fallback():
    """A1: f32 iterate + host f64 certify; forced cert failure triggers warm f64 re-solve."""
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import pdhg
    from qenivo.models.refinery import LEVELS, refinery_lp
    p = refinery_lp(*LEVELS[1], seed=1)
    C = p.c[:, None] * np.random.default_rng(4).uniform(0.95, 1.05, (p.n, 4))
    o = pdhg.PDHGOptions(tol=1e-4, persistent="off", detect_infeasibility=False,
                         time_limit=120.0, diet_f32_cert=True, dtype="float32")
    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, o, "cupy")
    assert all(s == "optimal" for s in br.status), br.status
    for j in range(4):
        q = p.copy(); q.c = C[:, j].copy()
        assert kkt_residuals(q, br.x[:, j], br.y[:, j])["max_rel"] <= 1e-4 * 1.0001

    # Force the fallback path: corrupt one column so host certify fails, then re-solve.
    bad = pdhg.BatchResult(x=br.x.copy(), y=br.y.copy(), status=list(br.status),
                           iterations=br.iterations.copy(), restarts=br.restarts.copy(),
                           kkt={k: v.copy() for k, v in br.kkt.items()},
                           solve_time=0.0, setup_time=0.0, backend=br.backend, eta=br.eta,
                           primal_weight=br.primal_weight.copy())
    bad.x[:, 0] = 0.0
    fixed = pdhg._certify_f32_with_f64_fallback(
        p.A, C, p.lc[:, None], p.uc[:, None], p.lx[:, None], p.ux[:, None], p.c0,
        bad, o, "cupy")
    assert fixed.status[0] == "optimal"
    q0 = p.copy(); q0.c = C[:, 0].copy()
    assert kkt_residuals(q0, fixed.x[:, 0], fixed.y[:, 0])["max_rel"] <= 1e-4 * 1.0001


@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA device")
def test_shared_bounds_and_retire_still_certify():
    """A2+A4 smoke: shared-bounds layout + 25% retire threshold still host-certifies."""
    from qenivo.certify.kkt import kkt_residuals
    from qenivo.engines import pdhg
    from qenivo.models.refinery import LEVELS, refinery_lp
    p = refinery_lp(*LEVELS[1], seed=2)
    C = p.c[:, None] * np.random.default_rng(5).uniform(0.95, 1.05, (p.n, 6))
    o = pdhg.PDHGOptions(tol=1e-4, persistent="off", detect_infeasibility=False,
                         time_limit=120.0, diet_shared_bounds=True, diet_retire_frac=0.25,
                         diet_f32_cert=True, dtype="float32")
    br = pdhg.solve_batch(p.A, C, p.lc, p.uc, p.lx, p.ux, p.c0, o, "cupy")
    assert all(s == "optimal" for s in br.status), br.status
    for j in range(6):
        q = p.copy(); q.c = C[:, j].copy()
        assert kkt_residuals(q, br.x[:, j], br.y[:, j])["max_rel"] <= 1e-4 * 1.0001
