# Engines

| Name | Role |
|---|---|
| `auto` | Router picks engine and device; reason recorded in the certificate |
| `simplex` | Bounded revised primal/dual simplex (native C++ when available) |
| `ipm` | Mehrotra interior-point |
| `pdhg` | Batched restarted-Halpern PDHG |
| `pdhg-rf` | float32 PDHG + float64 iterative refinement |
| `pdqp` | Convex QP |
| `milp` | Branch and cut |
| `miqp` | Plugin via `register_engine` |
| `race` | Verifier-gated multi-engine race on CPU cores |

List what is loaded in this install:

```bash
qenivo engines --json
```

Architecture detail lives in the package file `docs/ARCHITECTURE.md`.
