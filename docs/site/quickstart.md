# Quickstart

## Install

```bash
# from the qenivo/ package root
python -m pip install -e ".[dev]"
python -m pip install -e ".[gpu]"      # optional CUDA 12
python -m pip install -e ".[server]"   # optional console API
python -m pip install -e ".[docs]"     # optional MkDocs site
```

Version is defined once in `qenivo.__version__` and read by packaging.

## First solve

```bash
qenivo model transportation -o transport.mps
qenivo solve transport.mps --engine auto --cert transport.cert.json
qenivo verify transport.mps transport.cert.json
```

Python:

```python
import qenivo
sol = qenivo.solve("transport.mps")
print(sol.summary())
sol.save_certificate("transport.cert.json")
```

## Check the environment

```bash
qenivo doctor
qenivo engines --json
```

## Sovereignty rule

Do not add HiGHS, SCIP, OR-Tools, GLPK, SuiteSparse, MUMPS, PETSc, Eigen solvers,
`scipy.sparse.linalg`, or `scipy.optimize` to the solve path or to `install_requires`.
NumPy / SciPy arrays are fine. HiGHS may be installed separately for `bench/` comparisons only.
