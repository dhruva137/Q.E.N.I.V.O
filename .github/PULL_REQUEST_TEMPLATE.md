---
name: Pull request
about: Changes to QENIVO
---

## Summary

-

## Sovereignty

- [ ] No HiGHS / SCIP / OR-Tools / GLPK / SuiteSparse / MUMPS / PETSc / Eigen solvers / `scipy.sparse.linalg` / `scipy.optimize` on the solve path or in `install_requires`
- [ ] Answers still go through certification (`api._finish` or equivalent)

## Test plan

- [ ] `python -m pytest -q`
- [ ] `qenivo doctor` (if CLI / packaging touched)
- [ ] Docs / CI updated if needed
