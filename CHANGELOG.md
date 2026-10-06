# Changelog

All notable changes to the `qenivo` package are recorded here.

## Unreleased

### Added

- SciPy-grade CLI surface: `bench`, `convert`, `presolve`, `doctor`, `engines`,
  `completion`, and global `--json` on every command.
- Importable `qenivo.benchmarks` wrapper around `bench/run.py` plus JSON sparse-triplet
  helpers for `qenivo convert`.
- Packaging extras `[gpu]`, `[server]`, `[dev]`, `[docs]`; `native/simplex_core.cpp`
  shipped as package data; single version source (`qenivo.__version__`).
- MkDocs Material site under `docs/site/`.
- GitHub Actions matrix (Ubuntu 3.10–3.13, Windows, macOS), ruff, sdist/wheel, docs
  build, CPU bench smoke; issue/PR templates; Dependabot for dev tooling.
- `CONTRIBUTING.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`.
