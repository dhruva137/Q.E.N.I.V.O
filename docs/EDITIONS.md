# QENIVO editions: public core vs Pro

Open-core split for the repository that goes live after 5 Oct 2026. Public code
must stand alone; Pro modules are optional plugins. A public document must not
imply that the public code reproduces a number it cannot.

Related: `packaging/private_paths.py`, `packaging/export_public.py`,
`packaging/build_pro.py`, `qenivo.edition`.

## Editions

| Edition | Package | Licence | How it loads |
|---|---|---|---|
| **public** | `qenivo` (this repo) | Apache-2.0 | default install |
| **pro** | `qenivo-pro` (private) | All rights reserved | entry-point group `qenivo_pro` calling `qenivo.register_engine` |

`qenivo.edition.detect_edition()` returns `"pro"` iff any `qenivo_pro` entry
point is installed; otherwise `"public"`. Public source never imports
`qenivo_pro` or any path listed under Private candidates.

---

## Inventory

### Public core (PS 26119 requirements — stays in the public tree)

Everything needed to solve, certify, and demonstrate the problem statement
without Pro plugins.

| Area | Paths | Role |
|---|---|---|
| API / CLI / server | `api.py`, `cli.py`, `server.py`, `__main__.py`, `demo.py` | solve entry, HTTP |
| Model / I/O | `model.py`, `io/*` | Problem, MPS/LP/JSON/sheets/GAMS |
| Engines (baseline) | `engines/simplex.py`, `ipm.py`, `pdhg.py`, `pdqp.py`, `milp.py`, `presolve.py`, `cuts.py`, `crossover.py`, `precondition.py`, `refine.py`, `fj.py`, `exact_lp.py`, `native_simplex.py`, `native_ipm.py`, `miqp.py`, `gpu_bnb.py`, `multigpu.py`, `nlp_*.py`, `registry.py` | simplex, IPM, PDHG, PDQP, B&C, plugins |
| Native | `native/simplex_core.cpp`, `native/ipm/*`, `native/exact/*`, `native/mps_reader.cpp` | indigenous factorisations |
| Kernels (baseline) | `kernels/backend.py`, `cuda_kernels.py`, `persistent.py` | CuPy/NVRTC; public bandwidth path |
| Certify | `certify/*` | KKT / Farkas / independent verify |
| Workload (public) | `workload/cases.py`, `router.py` (published thresholds), `ranging.py`, `parallel.py`, `explain.py`, `recursion.py`, `slp.py`, `bilinear_bound.py` | case stacks, routing, ranging |
| Sector models | `models/williams.py`, `refinery.py`, `industry.py`, `industry_ext/*`, `refinery_full/*` | PS sector demos (synthetic / literature) |
| NLP | `nlp/*` | convex NLP / pooling helpers |
| Provenance / security | `provenance.py`, `security.py` | R1 tripwires, auth helpers |
| UI | `ui/*` | planner console assets |
| Bench / tests / docs | `bench/*` (minus MRPL private data), `tests/*`, `docs/*` (this file included) | evidence discipline |
| Packaging (public) | `packaging/export_public.py`, `make_upload_zip.py`, Docker/IIS helpers | public export gate |

Built-in engines register through `qenivo.register_engine` or the
`qenivo.engines` entry-point group. That mechanism is also how Pro attaches.

### Private candidates (competitive — excluded from public export)

These are **new this week** (W02–W06) or customer-shaped assets. They are not
required to satisfy the PS checklist with the public core. Until checked in,
paths are reserved in `packaging/private_paths.py` so export stays honest.

| Workstream | Candidate paths | Why private |
|---|---|---|
| **W02** parametric crude-value | `workload/parametric.py`, `workload/breakpoints.py`, `engines/parametric_simplex.py` | flagship cargo break-even / value curves |
| **W04** byte-diet kernels | `kernels/byte_diet.py` (or package) | tuned f32 / shared-bounds / L2 tiling / active-set compaction beyond the public baseline kernels |
| **W05** Stack Simplex | `research_prototypes/stack_simplex/`, later `engines/stack_simplex.py` | GPU lock-step exact-vertex batch |
| **W06** mixed-precision backbone | `research_prototypes/mixed_precision/`, `engines/mixed_precision.py`, `native/mixed_precision/` | fp32 factor + fp64/exact decide |
| **W03** router thresholds | `workload/router_thresholds_pro.py`, `router_policy_pro.json` | calibrated proprietary cut-points; **public `router.py` keeps published defaults** from laptop evidence |
| **MRPL-shaped data** | `models/mrpl/`, `data/mrpl/`, `bench/instances/mrpl/` | customer-shaped models and matrices not for public release |

Preferred packaging: move implementations into a separate `qenivo-pro` package
under `src/qenivo_pro/` (also on the private path list) rather than leaving
dead imports in public modules.

---

## Plugin design

1. **Public never imports private.** No `import qenivo_pro`, no direct import of
   the candidate paths above from public modules or public tests.
2. **Registration only.** Pro ships a `register()` callable that calls
   `qenivo.register_engine(name, fn, classes=..., description=...)` (and may
   install optional policy hooks that public code looks up only through the
   registry / documented optional APIs).
3. **Entry points.** Pro declares:

   ```toml
   [project.entry-points."qenivo_pro"]
   register = "qenivo_pro:register"
   ```

   The public registry loads both `qenivo.engines` and `qenivo_pro` (see
   `engines/registry.py`). A missing Pro install is normal: public tests must
   pass without it.
4. **Worked public example.** `engines/miqp.py` already registers via
   `register_engine` and is the pattern Pro engines copy.
5. **Export gate.** `python packaging/export_public.py` copies a tree with
   private paths removed and fails if any private import marker remains.

---

## Results honesty (R4)

Every results file under `bench/results/` (JSON or JSONL) must record which
edition produced it:

* File-level: `meta.edition` is `"public"` or `"pro"`.
* Per-row JSONL: each object includes `"edition": "public"|"pro"` (same value
  as the run that wrote it).

Helpers: `qenivo.edition.detect_edition()` and `edition_meta()`.

**Rule for public docs (README, EVIDENCE, decks, chat):**

* Numbers measured with only the public core may be cited as public.
* Numbers that need W02 parametric valuation, W04 byte-diet kernels, W05 Stack
  Simplex, W06 mixed-precision backbone, Pro router thresholds, or MRPL-shaped
  private data must be labelled **Pro edition** (or omitted from public-facing
  claims).
* Today’s published EVIDENCE Netlib / MIPLIB / A100 batch numbers are from the
  **public** engine set (baseline kernels + public router). Do not imply they
  require Pro. Future Pro-only speedups must not be folded into those tables
  without an `edition: "pro"` results file and an explicit label.

A “cannot reproduce” accusation against the public repo is worse than a rival’s
claim: if the public tree cannot produce a number, do not publish it as public.

---

## Licence notes

| Tree | Licence | Header |
|---|---|---|
| Public (`qenivo`) | Apache-2.0 (`LICENSE` in this repo) | existing Apache notices |
| Private / Pro modules | **All rights reserved** | each private source file must carry a short proprietary header; not Apache-2.0 |

Checks:

* Do not copy Apache-licensed third-party solver code into private modules
  (R2 already forbids copying solver projects; re-check when Pro files land).
* Allowed runtime deps for Pro are the same as public (NumPy/SciPy arrays,
  CuPy, NVRTC kernels written here). Comparators (HiGHS, etc.) stay in `bench/`
  only.
* `NOTICE` remains for public third-party attributions (e.g. cuPDLPx
  Apache-2.0 survey notes); Pro does not relicense those.

Suggested private file header:

```text
# Copyright (c) <year> <owner>. All rights reserved.
# Proprietary. Not licensed under Apache-2.0. Do not distribute.
```

---

## Export / build commands

```bash
# Public tree + import grep + short pytest (skipped if free RAM < 1.5 GB)
python packaging/export_public.py

# Tree only
python packaging/export_public.py --skip-pytest

# Pro wheel outline (deferred until private modules exist)
python packaging/build_pro.py
```

`.gitattributes` marks reserved private paths with `export-ignore` so archive
exports omit them even before the files exist.
