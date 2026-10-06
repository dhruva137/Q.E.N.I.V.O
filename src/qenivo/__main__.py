"""python -m qenivo → same entry as the `qenivo` console script."""
from .cli import main

raise SystemExit(main())
