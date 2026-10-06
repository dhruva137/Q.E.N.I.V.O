# Security policy

## Supported versions

The `main` / release branch of the `qenivo` package receives security fixes.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email the maintainers (Team Epoch Zero / SIH 2026 PS 26119 contacts listed with the
project) with:

* a description of the issue and its impact
* steps to reproduce (no proprietary plant data)
* affected version / commit

You should receive an acknowledgement within a few business days.

## Solve-path integrity

A class of issues we treat seriously: any dependency or import that places a foreign
solver or sparse-direct library on the solve path, defeating provenance tripwires.
Report those the same way.
