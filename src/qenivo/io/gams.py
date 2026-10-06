"""Reader for GAMS models in "scalar" form (the output of GAMS Convert).

Supports what refinery planning models written by GAMS Convert contain: continuous, positive
and binary variables; .lo / .up / .fx / .l assignments; equations `eN..  expr =E=|=L=|=G= rhs;`
whose left side is a sum of terms  c*x_i  and  c*x_i*x_j  (bilinear); and the final
`Solve m using ... maximizing|minimizing x_k;`. Any other nonlinear function is rejected with
a clear error rather than misread.

Returns a `BilinearModel`: the linear part as a qenivo Problem (objective = the objective
variable), plus the bilinear terms (row, i, j, coefficient) and the GAMS starting point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..model import Problem

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VAR = r"[xbi]\d+"
_FUNCS = re.compile(r"\b(sqr|sqrt|exp|log|log10|power|abs|min|max|sin|cos|tan|errorf|div|rpower|cvpower|vcpower)\s*\(",
                    re.I)


@dataclass
class BilinearModel:
    linear: Problem                     # all constraints with bilinear terms removed
    terms: list                         # [(row, i, j, coef)]
    start: np.ndarray                   # GAMS .l values (defaults 0 clipped to bounds)
    objective_var: int
    sense: str
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def nonlinear_rows(self):
        return sorted({t[0] for t in self.terms})

    def evaluate(self, x) -> np.ndarray:
        """Row activities of the full (nonlinear) model at x."""
        act = self.linear.A @ x
        for r, i, j, c in self.terms:
            act[r] += c * x[i] * x[j]
        return act

    def violation(self, x) -> np.ndarray:
        act = self.evaluate(x)
        p = self.linear
        return np.maximum(p.lc - act, 0.0) + np.maximum(act - p.uc, 0.0)

    def objective(self, x) -> float:
        return float(x[self.objective_var])

    def size_str(self) -> str:
        p = self.linear
        return (f"{p.m} rows x {p.n} cols, {p.nnz} linear nnz, {len(self.terms)} bilinear terms in "
                f"{len(self.nonlinear_rows)} rows" + (f", {int(p.integer.sum())} binaries" if p.is_mip else ""))


_TOK = re.compile(rf"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<var>{_VAR})|(?P<op>[-+*()]))")


def _tokens(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse near {s[pos:pos + 40]!r}")
        out.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
        while pos < len(s) and s[pos].isspace():
            pos += 1
    return out


def _poly_mul(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(sorted(ka + kb))
            out[k] = out.get(k, 0.0) + va * vb
    return out


def _poly_add(a, b, sign=1.0):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + sign * v
    return out


def _split_terms(expr: str):
    """Expand a polynomial expression: '3*x1 - x2*(x3 + 2*x4)' -> [(coef, [vars])]."""
    toks = _tokens(expr)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def take():
        t = peek()
        pos[0] += 1
        return t

    def expression():
        sign = 1.0
        if peek()[1] in ("+", "-"):
            sign = -1.0 if take()[1] == "-" else 1.0
        acc = {k: sign * v for k, v in product().items()}
        while peek()[1] in ("+", "-"):
            s = -1.0 if take()[1] == "-" else 1.0
            acc = _poly_add(acc, product(), s)
        return acc

    def product():
        acc = factor()
        while peek()[1] == "*":
            take()
            acc = _poly_mul(acc, factor())
        return acc

    def factor():
        kind, val = take()
        if kind == "num":
            return {(): float(val)}
        if kind == "var":
            return {(val,): 1.0}
        if val == "(":
            e = expression()
            if take()[1] != ")":
                raise ValueError("unbalanced parenthesis")
            return e
        if val == "-":
            return {k: -v for k, v in factor().items()}
        raise ValueError(f"unexpected token {val!r}")

    poly = expression()
    if pos[0] != len(toks):
        raise ValueError(f"trailing tokens in {expr[:60]!r}")
    return [(v, list(k)) for k, v in poly.items() if v != 0.0]


def read_gams(path) -> BilinearModel:
    text = open(path, encoding="utf-8", errors="replace").read()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("*"))
    if _FUNCS.search(text.split("Model")[0]):
        raise ValueError(f"{path}: nonlinear functions other than products are not supported")

    def decl(keyword):
        m = re.search(keyword + r"\s+(.*?);", text, re.S | re.I)
        return re.findall(_VAR, m.group(1)) if m else []

    # the general Variables block comes first; the objective variable may be declared there only
    all_vars = decl(r"(?<!Positive )(?<!Binary )(?<!Integer )\bVariables")
    positive = set(decl(r"Positive\s+Variables"))
    binary = set(decl(r"Binary\s+Variables"))
    integer = set(decl(r"Integer\s+Variables"))
    names = list(dict.fromkeys(all_vars + sorted(positive | binary | integer - set(all_vars),
                                                  key=lambda v: (v[0], int(v[1:])))))
    idx = {v: k for k, v in enumerate(names)}
    n = len(names)
    lo = np.full(n, -np.inf)
    up = np.full(n, np.inf)
    for v in positive | binary | integer:
        lo[idx[v]] = 0.0
    for v in binary:
        up[idx[v]] = 1.0
    start = np.zeros(n)
    for v, attr, val in re.findall(rf"({_VAR})\.(lo|up|fx|l)\s*=\s*({_NUM})\s*;", text):
        k, val = idx[v], float(val)
        if attr == "lo":
            lo[k] = val
        elif attr == "up":
            up[k] = val
        elif attr == "fx":
            lo[k] = up[k] = val
        else:
            start[k] = val

    rows, lc, uc, terms = [], [], [], []
    ri, ci, vv = [], [], []
    for name, lhs, rel, rhs in re.findall(rf"(e\d+)\.\.\s*(.*?)=([EeLlGg])=\s*({_NUM})\s*;", text, re.S):
        r = len(rows)
        rows.append(name)
        b = float(rhs)
        rel = rel.upper()
        lc.append(b if rel in "EG" else -np.inf)
        uc.append(b if rel in "EL" else np.inf)
        for coef, vs in _split_terms(lhs):
            if len(vs) == 0:                     # constant on the left: move it right
                lc[-1] -= coef
                uc[-1] -= coef
            elif len(vs) == 1:
                if vs[0] not in idx:
                    raise ValueError(f"{name}: unknown variable {vs[0]}")
                ri.append(r)
                ci.append(idx[vs[0]])
                vv.append(coef)
            elif len(vs) == 2:
                if vs[0] not in idx or vs[1] not in idx:
                    raise ValueError(f"{name}: unknown variable in product {vs}")
                i, j = idx[vs[0]], idx[vs[1]]
                terms.append((r, i, j, coef))
            else:
                raise ValueError(f"{name}: product of {len(vs)} variables not supported")
    m = len(rows)
    sm = re.search(rf"Solve\s+\w+\s+using\s+\w+\s+(maximizing|minimizing)\s+({_VAR})\s*;", text, re.I)
    if not sm:
        raise ValueError("no Solve ... maximizing|minimizing statement found")
    obj_name = sm.group(2)
    if obj_name not in idx:
        raise ValueError(f"objective variable {obj_name} is not declared")
    sense, objv = sm.group(1).lower(), idx[obj_name]
    A = sp.csr_matrix((vv, (ri, ci)), shape=(m, n))
    sign = -1.0 if sense == "maximizing" else 1.0
    c = np.zeros(n)
    c[objv] = sign
    is_int = np.array([v in binary or v in integer for v in names])
    lin = Problem(c=c, A=A, lc=np.array(lc), uc=np.array(uc), lx=lo, ux=up, obj_sign=sign,
                  name=str(path).replace("\\", "/").split("/")[-1].split(".")[0], row_names=rows, col_names=names,
                  integer=is_int if is_int.any() else None)
    return BilinearModel(linear=lin, terms=terms, start=np.clip(start, lo, up), objective_var=objv,
                         sense=sense, source=str(path))
