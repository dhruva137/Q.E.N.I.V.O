# QENIVO

Sovereign, certified optimisation for LP, MILP and convex QP — built for refinery and
industrial planning (SIH 2026, PS 26119).

```bash
pip install -e ".[dev]"
qenivo doctor
qenivo engines
qenivo solve plan.mps --cert plan.cert.json
qenivo verify plan.mps plan.cert.json
```

Every answer is re-checked on the original model and can be independently verified.
No foreign solver library sits on the solve path.

Continue with the [Quickstart](quickstart.md), [CLI](cli.md), and the package file `docs/ARCHITECTURE.md`.
