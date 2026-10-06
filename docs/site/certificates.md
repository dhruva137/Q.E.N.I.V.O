# Certificates

Every proven answer becomes a JSON certificate (model SHA-256, verdict, residuals or ray,
engine, routing reason, environment). Re-check on another machine:

```bash
qenivo solve plan.mps --cert plan.cert.json
qenivo verify plan.mps plan.cert.json
qenivo verify plan.mps plan.cert.json --exact   # rational arithmetic
```

Verdicts:

- **optimal** — primal/dual residuals and gap on the original model meet the tolerance
- **infeasible** — Farkas ray validated
- **unbounded** — recession ray validated
- anything else — `not_proven` (exit code 1)

The verifier shares no code with the engines. Measured coverage is summarised in the package file `docs/EVIDENCE.md`.
