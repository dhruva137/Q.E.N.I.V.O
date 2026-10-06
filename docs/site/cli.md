# CLI reference

Authoritative help: `qenivo --help` and `qenivo <cmd> --help`.
This page mirrors the argparse surface in `qenivo.cli`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Proven verdict (`optimal` / `infeasible` / `unbounded`) or successful utility |
| 1 | Not proven / soft failure |
| 2 | Usage error |
| 3 | Read / I/O error |
| 4 | Verification rejected |

Global `--json` works before or after the subcommand: `qenivo --json doctor`, `qenivo doctor --json`.

## Commands

### solve

```text
qenivo solve model.mps [--engine auto|simplex|ipm|pdhg|pdhg-rf|pdqp|milp|miqp|race|…]
                       [--tol 1e-6] [--backend auto|numpy|cupy] [--time-limit SEC]
                       [--cert out.json] [--sol out.sol] [--json] [--verbose]
```

### verify / explain / range / cases / recursion

```text
qenivo verify model.mps cert.json [--exact] [--tol TOL] [--json]
qenivo explain model.mps [--top N] [--json]
qenivo range model.mps [--top N] [--json]
qenivo cases model.mps cases.json|cases.csv|cases.xlsx [--tol] [--backend] [--out] [--json]
qenivo recursion haverly1|haverly2|haverly3 [--method …] [--q0] [--verbose] [--json]
```

### model / demo / serve / info

```text
qenivo model [name] [-p k=v] [-o out.mps] [--json]
qenivo demo
qenivo serve [--host] [--port] [--token-file] [--proxy-user-header] [--trusted-proxy]
             [--audit-dir] [--retention-days]
qenivo info [--json]
```

### bench

Wraps `bench/run.py` through `qenivo.benchmarks` (no edit to the runner).

```text
qenivo bench <set> [--engine auto] [--time-limit SEC] [--out PATH]
                   [--backend numpy] [--jobs 1] [--with-ref] [--json]
```

Sets come from `bench/sets.py` (`netlib`, `small`, …). Default is `--no-ref` (HiGHS optional via `--with-ref`).

### convert

```text
qenivo convert in.mps|in.json out.mps|out.json|out.lp [--json]
```

MPS uses existing `io.mps`. JSON uses the sparse-triplet helper in `qenivo.benchmarks.json_format`.
If LP support is absent, converting to `.lp` exits **2** with a clear message (no fake writer).

### presolve

```text
qenivo presolve in.mps [out.mps] [--report] [--json]
```

Calls the public API of `engines.presolve.presolve`. If the reduced model cannot be written,
prints the report as JSON and states what was not written.

### doctor / engines / completion

```text
qenivo doctor [--json]
qenivo engines [--json]
qenivo completion bash|zsh|powershell [--json]
```

`doctor` checks Python ≥ 3.10, NumPy, SciPy, CuPy, `gpu_status` / `gpu_info`,
`native_simplex.status()`, a C/C++ compiler on PATH, and the provenance guard — with one fix line per failure.
