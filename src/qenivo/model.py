"""The one problem form every QENIVO engine works on.

    minimise    0.5 x'Qx + c'x + c0
    subject to  lc <= A x <= uc          (rows: equality, <=, >=, ranged or free)
                lx <=  x  <= ux          (columns: any bounds, possibly infinite)
                x_j integer for j in `integer`

Maximisation problems are stored negated (obj_sign = -1) so engines always minimise;
`objective()` reports the value in the sense the user wrote.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Problem:
    c: np.ndarray
    A: sp.csr_matrix
    lc: np.ndarray
    uc: np.ndarray
    lx: np.ndarray
    ux: np.ndarray
    c0: float = 0.0
    obj_sign: float = 1.0
    name: str = ""
    row_names: list | None = None
    col_names: list | None = None
    integer: np.ndarray | None = None     # bool mask, MILP only
    Q: sp.csr_matrix | None = None        # QP only (symmetric positive semidefinite)
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)   # generator-specific data (e.g. pooling layout)

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        self.A = sp.csr_matrix(self.A, dtype=np.float64)
        self.A.sort_indices()
        for k in ("lc", "uc", "lx", "ux"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64).reshape(-1))
        m, n = self.A.shape
        if self.c.shape != (n,) or self.lx.shape != (n,) or self.ux.shape != (n,):
            raise ValueError(f"column data must have length n={n}")
        if self.lc.shape != (m,) or self.uc.shape != (m,):
            raise ValueError(f"row data must have length m={m}")
        if self.Q is not None:
            self.Q = sp.csr_matrix(self.Q, dtype=np.float64)
        if np.any(self.lx > self.ux) or np.any(self.lc > self.uc):
            self.notes.append("some lower bounds exceed upper bounds (trivially infeasible)")

    # ------------------------------------------------------------------ shape
    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def nnz(self) -> int:
        return int(self.A.nnz)

    @property
    def is_qp(self) -> bool:
        return self.Q is not None and self.Q.nnz > 0

    @property
    def is_mip(self) -> bool:
        return self.integer is not None and bool(np.any(self.integer))

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    def objective(self, x: np.ndarray) -> float:
        """Objective in the original (min or max) sense."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        v = float(self.c @ x) + self.c0
        if self.is_qp:
            v += 0.5 * float(x @ (self.Q @ x))
        return self.obj_sign * v

    def size_str(self) -> str:
        return f"{self.m} rows x {self.n} cols, {self.nnz} nnz"

    def names(self):
        rn = self.row_names if self.row_names and len(self.row_names) == self.m else [f"R{i}" for i in range(self.m)]
        cn = self.col_names if self.col_names and len(self.col_names) == self.n else [f"C{j}" for j in range(self.n)]
        return rn, cn

    def fingerprint(self) -> str:
        """SHA-256 over the numerical data, so a certificate is tied to one exact model."""
        h = hashlib.sha256()
        A = self.A
        for arr in (A.indptr, A.indices, A.data, self.c, self.lc, self.uc, self.lx, self.ux):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(np.float64(self.c0).tobytes())
        h.update(np.float64(self.obj_sign).tobytes())
        if self.is_qp:
            for arr in (self.Q.indptr, self.Q.indices, self.Q.data):
                h.update(np.ascontiguousarray(arr).tobytes())
        if self.is_mip:
            h.update(np.ascontiguousarray(self.integer).tobytes())
        return h.hexdigest()

    def copy(self) -> "Problem":
        return Problem(c=self.c.copy(), A=self.A.copy(), lc=self.lc.copy(), uc=self.uc.copy(),
                       lx=self.lx.copy(), ux=self.ux.copy(), c0=self.c0, obj_sign=self.obj_sign,
                       name=self.name, row_names=list(self.row_names) if self.row_names else None,
                       col_names=list(self.col_names) if self.col_names else None,
                       integer=None if self.integer is None else self.integer.copy(),
                       Q=None if self.Q is None else self.Q.copy(), notes=list(self.notes),
                       meta=dict(self.meta))


class ModelBuilder:
    """Small row/column builder for writing models in code (examples, generators, tests)."""

    def __init__(self, name: str = "model", maximize: bool = False):
        self.name = name
        self.maximize = maximize
        self.cols, self.lx, self.ux, self.c, self.is_int = [], [], [], [], []
        self.rows, self.lc, self.uc = [], [], []
        self.ri, self.ci, self.v = [], [], []
        self.col = {}

    def var(self, name, lo=0.0, hi=np.inf, obj=0.0, integer=False):
        if name in self.col:
            raise ValueError(f"duplicate column {name!r}")
        self.col[name] = len(self.cols)
        self.cols.append(name); self.lx.append(lo); self.ux.append(hi)
        self.c.append(obj); self.is_int.append(integer)
        return name

    def row(self, name, coefs: dict, lo=-np.inf, hi=np.inf):
        i = len(self.rows)
        self.rows.append(name); self.lc.append(lo); self.uc.append(hi)
        for k, a in coefs.items():
            self.ri.append(i); self.ci.append(self.col[k]); self.v.append(float(a))
        return name

    def build(self) -> Problem:
        A = sp.csr_matrix((self.v, (self.ri, self.ci)), shape=(len(self.rows), len(self.cols)))
        A.eliminate_zeros()
        sign = -1.0 if self.maximize else 1.0
        integer = np.array(self.is_int, dtype=bool)
        return Problem(c=sign * np.array(self.c, dtype=float), A=A, lc=np.array(self.lc, float),
                       uc=np.array(self.uc, float), lx=np.array(self.lx, float),
                       ux=np.array(self.ux, float), obj_sign=sign, name=self.name,
                       row_names=list(self.rows), col_names=list(self.cols),
                       integer=integer if integer.any() else None)
