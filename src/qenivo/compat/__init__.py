"""Familiar front doors for planners who already script a commercial solver.

    shell.py   `qenivo-io`: a command shell (read, optimize, display, set, write, quit) that
               follows the command layout of the classic interactive optimizers, written from
               their public user manuals (see docs/COMPATIBILITY.md); every solve is certified.
    shadow.py  run an external solver command and QENIVO on the same model and diff the
               verdict, objective, primal and duals, with both answers checked on the model.

No external solver is imported here: shadow mode runs the other solver as a separate process.
"""
