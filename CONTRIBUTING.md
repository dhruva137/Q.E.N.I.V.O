# Contributing to QENIVO

## Setup

```bash
cd qenivo
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
qenivo doctor
python -m pytest -q
```

Optional extras: `[gpu]`, `[server]`, `[docs]`.

## Tests

- Fast CPU suite: `python -m pytest -q`
- GPU tests skip without CUDA
- Provenance tripwires must stay clean

## Benchmark protocol

1. Use `qenivo bench <set> …` or `python bench/run.py …`
2. Write results under `bench/results/` with commit and machine metadata
3. Never invent numbers in docs or slides — cite a results file
4. HiGHS (`highspy`) is an **optional comparator only**, never a dependency of the package

## Sovereignty rule (PS 26119)

The solve path must not call foreign optimisation or sparse-direct-solver libraries
(HiGHS, SCIP, OR-Tools, GLPK, SuiteSparse, MUMPS, PETSc, Eigen solvers,
`scipy.sparse.linalg`, `scipy.optimize`). NumPy / SciPy arrays and dense BLAS/LAPACK
are allowed. Tests assert zero tripwire hits.

## Adding an engine

1. Implement `fn(prob, tol, time_limit, backend, verbose, **options) -> Solution`
2. Return answers through the same certification path as built-ins
3. Register with `qenivo.register_engine(...)` or the `qenivo.engines` entry-point group
4. See `docs/site/plugins.md` and `engines/miqp.py`

## Style

- `ruff check src tests`
- Prefer small, focused PRs that stay inside the file ownership of your task
- Do not rewrite `docs/ARCHITECTURE.md`, `EVIDENCE.md`, or `GROUND_REALITY.md` casually —
  add new pages under `docs/site/` when documenting packaging / CLI work

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
