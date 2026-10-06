"""Scalar-affine mapping used by the JuMP / MathOptInterface plugin.

The Julia optimizer in QenivoMOI.jl walks MOI ScalarAffineFunction constraints
and builds the same CSR the function `translate` builds from a FakeModel. Tests
call `translate` so they do not need Julia or JuMP. Lubin et al., JuMP 1.0,
Mathematical Programming Computation, 2023.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _affine():
    name = "qenivo_bindings_affine"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[1] / "affine.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def translate(model):
    affine = _affine()
    if isinstance(model, affine.Model):
        return model
    if affine.is_fake(model):
        return affine.from_fake(model)
    raise TypeError("pass a FakeModel here, or build the model in QenivoMOI.jl")


def solve(model, engine=None, tol=1e-8, time_limit=60.0):
    return _affine().solve(translate(model), engine=engine, tol=tol, time_limit=time_limit)
