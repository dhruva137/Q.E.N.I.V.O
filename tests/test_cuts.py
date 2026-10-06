"""Cutting planes are valid: no cut removes a proven optimal integer point."""
import numpy as np
import pytest

from qenivo.engines.cuts import cover_cuts, gmi_cuts, mir_cuts
from qenivo.engines.milp import INT_TOL, _lp, solve_milp
from qenivo.models.industry import LIBRARY

CASES = [("facility_location", dict(facilities=6, customers=15)), ("lot_sizing", dict(items=3, periods=6)),
         ("unit_commitment", dict(units=4, periods=8)), ("crude_scheduling", dict(tanks=2, crudes=2, periods=4))]


@pytest.mark.parametrize("name,kw", CASES)
def test_root_cuts_keep_the_optimal_point(name, kw):
    p = LIBRARY[name](**kw)
    intmask = p.integer
    opt = solve_milp(p, time_limit=120, cuts=False, presolve=False)
    assert opt.verdict == "optimal"
    xo = opt.x
    lx, ux = p.lx.copy(), p.ux.copy()
    lx[intmask], ux[intmask] = np.ceil(lx[intmask] - INT_TOL), np.floor(ux[intmask] + INT_TOL)
    root = _lp(p, lx, ux, 60)
    assert root["status"] == "optimal"
    produced = 0
    for sep in (lambda: mir_cuts(p, root["x"], intmask, lx, ux), lambda: cover_cuts(p, root["x"], intmask, lx, ux),
                lambda: gmi_cuts(p, root, intmask, lx, ux)):
        r = sep()
        if r is None:
            continue
        P, rhs = r
        produced += P.shape[0]
        lhs = P @ xo
        assert np.all(lhs >= rhs - 1e-6 * (1 + np.abs(rhs))), (name, float(np.min(lhs - rhs)))
    # and the cut loop reaches the same optimum
    withcuts = solve_milp(p, time_limit=120)
    assert withcuts.verdict == "optimal"
    assert abs(withcuts.objective - opt.objective) <= 1e-6 * (1 + abs(opt.objective))
