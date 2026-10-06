# Writing a plugin engine

Engines register through `qenivo.register_engine` (and optionally the entry-point group
`qenivo.engines`). The MIQP engine in `engines/miqp.py` is the worked example.

```python
from qenivo import register_engine
from qenivo.api import Solution  # engines normally return via api._finish

def my_engine(prob, tol=1e-6, time_limit=3600.0, backend="numpy", verbose=False, **options):
    # Solve, then return a Solution produced the same way as built-ins
    # (certificate + float64 check on the original model). Prefer calling
    # through qenivo.api helpers rather than inventing a verdict.
    raise NotImplementedError("see engines/miqp.py for the full pattern")

register_engine(
    "my_nlp",
    my_engine,
    classes=("NLP",),
    description="example plugin",
)
```

Third-party packages can avoid importing qenivo at install time by declaring:

```toml
[project.entry-points."qenivo.engines"]
my_nlp = "my_package.engine:register"
```

where `register()` calls `qenivo.register_engine(...)`.

Confirm discovery:

```bash
qenivo engines
```
