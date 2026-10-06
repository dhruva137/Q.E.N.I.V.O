"""Every sector model of PS 26119 builds and is solved to a certified answer."""
import pytest

from qenivo import solve
from qenivo.certify.kkt import kkt_residuals
from qenivo.engines import native_simplex as ns
from qenivo.engines.simplex import solve_simplex
from qenivo.models.industry import LIBRARY

SMALL = {"economic_dispatch": dict(units=4, periods=6), "unit_commitment": dict(units=3, periods=6),
         "transportation": dict(sources=6, sinks=15), "facility_location": dict(facilities=5, customers=12),
         "lot_sizing": dict(items=3, periods=5), "crude_blending": {},
         "crude_scheduling": dict(tanks=2, crudes=2, periods=4),
         "crude_unloading": dict(size="small", seed=0),
         "multi_commodity_flow": dict(nodes=10, commodities=3)}

# crude_unloading is a MILP whose small instance took 188 s on the pure-Python simplex (4-core
# laptop, 1 Oct 2026; 7.3 s natively in the W07 audit). Same proven optimum, but not inside 120 s.
_NEEDS_NATIVE = pytest.mark.skipif(
    ns.library() is None,
    reason="needs the native simplex to prove optimality inside the time limit: " + ns.status())


@pytest.mark.parametrize("name", [pytest.param(n, marks=_NEEDS_NATIVE) if n == "crude_unloading" else n
                                  for n in sorted(LIBRARY)])
def test_sector_model_solves_and_certifies(name):
    p = LIBRARY[name](**SMALL.get(name, {}))
    s = solve(p, time_limit=120)
    assert s.verdict == "optimal", (name, s.engine)
    if p.integer is None:
        k = kkt_residuals(p, s.x, p.obj_sign * s.y)
        assert k["max_rel"] <= 1e-6


def test_lp_sector_models_match_the_simplex():
    for name in ("transportation", "crude_blending", "multi_commodity_flow"):
        p = LIBRARY[name](**SMALL.get(name, {}))
        ref = solve_simplex(p)
        s = solve(p, engine="ipm", backend="numpy")
        assert abs(p.obj_sign * (p.c @ ref["x"] + p.c0) - s.objective) <= 1e-6 * (1 + abs(s.objective))


def test_qp_model_roundtrips_through_mps(tmp_path):
    from qenivo import read
    from qenivo.io.mps import write_mps
    p = LIBRARY["economic_dispatch"](**SMALL["economic_dispatch"])
    f = tmp_path / "ed.mps"
    write_mps(p, f, name="ed")
    q = read(f)
    assert q.Q is not None and abs(q.Q - p.Q).max() < 1e-12
    a, b = solve(p, time_limit=60), solve(q, time_limit=60)
    assert a.verdict == b.verdict == "optimal"
    assert abs(a.objective - b.objective) <= 1e-6 * (1 + abs(a.objective))
