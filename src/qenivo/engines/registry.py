"""Engine registry: the extension point of the solver core.

An engine is a function  fn(prob, tol, time_limit, backend, verbose, **options) -> Solution
plus the problem classes it accepts ("LP", "QP", "MILP", "MIQP", ...). Registering one makes
it available to `qenivo.solve(prob, engine=name)`, to the CLI (`--engine name`) and to the
server, with the same certificate and verification path as the built-in engines.

Third-party packages register engines without touching this code, through the entry-point
group "qenivo.engines" in their pyproject.toml:

    [project.entry-points."qenivo.engines"]
    my_nlp = "my_package.engine:register"      # register() calls qenivo.register_engine(...)

The optional Pro edition uses the separate group "qenivo_pro" (see docs/EDITIONS.md). Public
code never imports Pro modules; Pro only calls register_engine from its register() entry point.

Built-in engines (simplex, ipm, pdhg, pdhg-rf, pdqp, milp) are dispatched by api.solve; the
MIQP engine in engines/miqp.py is written *as a plugin* against this registry, as the worked
example of extending the core (PS 26119: "extended to MIQP, NLP and MINLP").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

_ENGINES: dict = {}
_LOADED = {"entry_points": False}


@dataclass
class EngineSpec:
    name: str
    fn: Callable
    classes: tuple = ("LP",)
    description: str = ""
    extra: dict = field(default_factory=dict)


def register_engine(name: str, fn: Callable, classes=("LP",), description: str = "", **extra) -> EngineSpec:
    spec = EngineSpec(name, fn, tuple(classes), description, extra)
    _ENGINES[name] = spec
    return spec


def problem_class(prob) -> str:
    q = prob.is_qp
    i = prob.is_mip
    return "MIQP" if (q and i) else "QP" if q else "MILP" if i else "LP"


def _load_entry_points():
    if _LOADED["entry_points"]:
        return
    _LOADED["entry_points"] = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        for group in ("qenivo.engines", "qenivo_pro"):
            try:
                group_eps = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
            except Exception:  # noqa: BLE001
                group_eps = []
            for ep in group_eps:
                try:
                    ep.load()()
                except Exception as e:  # noqa: BLE001  - a broken plugin must not break the core
                    import warnings
                    warnings.warn(f"qenivo engine plugin {ep.name!r} ({group}) failed to load: {e}")
    except Exception:  # noqa: BLE001
        pass


def get_engine(name: str) -> EngineSpec | None:
    _builtin_plugins()
    _load_entry_points()
    return _ENGINES.get(name)


def engines() -> dict:
    _builtin_plugins()
    _load_entry_points()
    return dict(_ENGINES)


def _builtin_plugins():
    if "miqp" not in _ENGINES:
        from . import miqp  # noqa: F401  (registers itself)
    if "milp-gpu" not in _ENGINES:
        from . import gpu_bnb  # noqa: F401  (registers itself)
    if "ipm-native" not in _ENGINES:
        from . import native_ipm  # noqa: F401  (registers itself)
