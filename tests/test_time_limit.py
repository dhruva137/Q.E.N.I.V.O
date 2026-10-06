"""time_limit bounds the whole solve, robustness ladder included.

The ladder used to give each fallback engine the full limit again, so a solve whose first engine ran
out of time could take three times as long as the user asked (45 minutes for a 900 s limit).
"""
import time

import numpy as np

import qenivo
from qenivo.models.williams import williams_lp


def _stall(time_limit=0.0, **_):
    time.sleep(max(0.0, min(time_limit, 5.0)))      # spends what it is given, never finishes


def test_ladder_keeps_to_the_time_limit(monkeypatch):
    from qenivo.engines import ipm, pdhg, simplex
    p = williams_lp()

    def fake_simplex(prob, tol=1e-9, time_limit=3600.0, **kw):
        _stall(time_limit)
        return {"status": "time_limit", "x": None, "y": None, "iterations": 0,
                "col_statuses": [], "row_statuses": [], "time": time_limit}

    def fake_ipm(prob, tol=1e-6, time_limit=3600.0, **kw):
        _stall(time_limit)
        return {"status": "time_limit", "x": None, "y": None, "iterations": 0, "time": time_limit}

    class _Batch:
        def column(self, k):
            return {"status": "time_limit", "x": np.zeros(p.n), "y": np.zeros(p.m), "iterations": 0,
                    "restarts": 0, "primal_weight": 1.0, "ray": None}

    def fake_pdhg(prob, tol=1e-4, time_limit=3600.0, **kw):
        _stall(time_limit)
        return _Batch()

    monkeypatch.setattr(simplex, "solve_simplex", fake_simplex)
    monkeypatch.setattr(ipm, "solve_ipm", fake_ipm)
    monkeypatch.setattr(pdhg, "solve", fake_pdhg)
    t0 = time.perf_counter()
    sol = qenivo.solve(p, engine="simplex", time_limit=1.0)
    wall = time.perf_counter() - t0
    assert sol.verdict == "not_proven"
    assert wall < 1.8, f"a 1 s limit took {wall:.2f} s"
    assert "time limit reached" in sol.engine.get("reason", ""), sol.engine.get("reason")
