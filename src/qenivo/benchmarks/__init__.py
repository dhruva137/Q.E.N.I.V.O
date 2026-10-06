"""Importable benchmark helpers wrapping ``bench/run.py`` without editing it.

HiGHS / highspy is used only as an optional comparator when the underlying runner
is invoked without ``--no-ref``. The solve path never imports a foreign solver.
"""
from __future__ import annotations

from .json_format import problem_from_json, problem_to_json, read_json, write_json
from .runner import available_sets, run_benchmark

__all__ = [
    "available_sets",
    "run_benchmark",
    "problem_from_json",
    "problem_to_json",
    "read_json",
    "write_json",
]
