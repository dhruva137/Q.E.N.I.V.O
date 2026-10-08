# QENIVO

A certified optimisation engine for LP, MILP and convex QP, written from the published mathematics, with GPU-batched what-if planning for refineries and other process industries.

QENIVO is open-source research software, developed and maintained by one person, and it is in development.

## What it does

QENIVO solves a stack of what-if cases (crude prices, unit capacities, demand) over the same refinery matrix. The GPU solves the whole stack in one kernel launch; the CPU turns every GPU point into an exact optimal vertex with exact shadow prices; and an independent verifier re-checks a certificate for every answer before it is reported.

```text
$ qenivo solve plan.mps --cert plan.cert.json
plan: OPTIMAL  objective 211365.1348

$ qenivo verify plan.mps plan.cert.json --exact
RESULT    VERIFIED
```

## Install

```bash
pip install qenivo                  # CPU (NumPy + SciPy only)
pip install "qenivo[gpu]"           # with CUDA 12 (CuPy)
pip install "qenivo[server]"        # REST server and planner console
```

Python 3.10 or newer. The native C++ cores are compiled on the target at first use with the machine's C++ compiler and cached per user; without a compiler QENIVO falls back to its Python engines.

## Quick start

```python
import qenivo

sol = qenivo.solve("plan.mps")          # routed engine, certificate attached
print(sol.summary())
print(sol.table("y", top=10))          # shadow prices of the tightest limits
sol.save_certificate("plan.cert.json")
```

Command line: `qenivo solve`, `qenivo verify`, `qenivo cases` (what-if stack from Excel), `qenivo explain`, `qenivo range`, `qenivo demo`, `qenivo serve`.

## Measured results

Each number comes from a results file in the repository that records the commit, machine and library versions. Failures stay in the files.

- 512 refinery LPs on one A100, all certified: 13.7 s against 767 s for HiGHS cold on 12 cores.
- 256 cases on 2x Tesla T4: 22.3 s against 975 s for HiGHS on the same machine.
- Netlib LP: 97 of 98 verified optimal, matching HiGHS within 1e-6.
- Native C++ simplex: median 1.7x HiGHS time on 14 hard Netlib LPs, so it is still slower than established solvers there.
- No solver libraries in the solve path, checked at run time. HiGHS appears only in the benchmark folder as a comparator.

The full table, including where QENIVO is slower, is in docs/EVIDENCE.md in the repository.

## Links

- Homepage: https://github.com/dhruva137/Q.E.N.I.V.O
- Repository: https://github.com/dhruva137/Q.E.N.I.V.O
- Documentation: https://dhruva137.github.io/Q.E.N.I.V.O/

Part of Paper To Anything (https://papertoanything.com), research software developed and maintained by Dhruva P Gowda.

## Licence

Apache-2.0.
