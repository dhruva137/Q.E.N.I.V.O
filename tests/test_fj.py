"""Feasibility Jump finds points that satisfy every row, bound and integrality."""
import numpy as np

from qenivo.engines.fj import feasibility_jump
from qenivo.engines.milp import _feasible
from qenivo.models.industry import LIBRARY


def test_feasibility_jump_on_sector_milps():
    for name, kw in (("facility_location", dict(facilities=6, customers=15)), ("lot_sizing", dict(items=3, periods=6)),
                     ("unit_commitment", dict(units=4, periods=8))):
        p = LIBRARY[name](**kw)
        x = feasibility_jump(p, p.integer, p.lx, p.ux, time_limit=20)
        assert x is not None, name
        assert _feasible(p, x)
        assert np.allclose(x[p.integer], np.round(x[p.integer]))
