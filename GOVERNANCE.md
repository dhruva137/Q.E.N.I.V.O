# Governance

QENIVO began as Team Epoch Zero's entry to SIH 2026 (PS 26119) and is intended as a
long-lived open-source numerical library.

## Maintainers

Maintainers are listed in `CITATION.cff` / the project authors field. They merge pull
requests, cut releases, and decide scope for the public API.

## Decision process

1. **Routine changes** (bug fixes, docs, tests, small CLI/packaging work): pull request
   review by one maintainer.
2. **Engine or certificate contract changes**: pull request plus explicit note in
   `CHANGELOG.md`; tests must cover the new contract.
3. **Sovereignty exceptions**: none without a written design note explaining why a
   dependency cannot live only in `bench/` as a comparator. Default answer is no.
4. **Release**: version bump in `qenivo.__version__` (single source), changelog entry,
   tag.

## Plugins

Third-party engines are welcome via `register_engine` / entry points. Plugins that pull
a forbidden solver into the process will fail provenance checks and will not be accepted
into the core repository.
