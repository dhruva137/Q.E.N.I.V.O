"""MPS reader and writer (free and fixed format, .gz/.bz2 transparently).

Semantics follow the common reader conventions (and HiGHS, our comparator, so
both solvers see the same model):
  * first N row is the objective; other N rows are dropped
  * RHS on the objective row is the negated objective constant
  * RANGES: E row R>0 -> [b, b+|R|], R<0 -> [b-|R|, b]; L -> [b-|R|, b]; G -> [b, b+|R|]
  * UP with a negative value on a column whose lower bound was not set -> lower = -inf
  * MI -> lower = -inf (upper unchanged); PL -> upper = +inf; BV -> [0,1] integer
  * integer columns come from MARKER INTORG/INTEND blocks and BV/LI/UI bounds
  * QUADOBJ (lower triangle, mirrored) and QMATRIX (full) give Q in 0.5 x'Qx
"""
from __future__ import annotations

import bz2
import gzip
import math
import os
from array import array
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..model import Problem as LPProblem

INF = math.inf
_OBJ, _DROP = -1, -2


def _open_text(path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", errors="replace")
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", errors="replace")
    return open(p, "r", errors="replace")


def _fixed_fields(line: str) -> list[str]:
    """Fixed-format MPS fields (columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61)."""
    spans = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)]
    return [line[a:b].strip() for a, b in spans if line[a:b].strip()]


def _suffixes(path: str) -> str:
    p = path.lower()
    for s in (".mps.gz", ".mps.bz2", ".lp.gz", ".lp.bz2", ".json.gz", ".json.bz2",
              ".mps", ".lp", ".json"):
        if p.endswith(s):
            return s.split(".")[1] if s.count(".") == 1 else s.split(".")[-2]
    return Path(path).suffix.lstrip(".").lower()


def read(path, name: str | None = None) -> LPProblem:
    """Read MPS / LP / JSON by suffix (.gz / .bz2 accepted). Used by api.read via read_mps."""
    kind = _suffixes(str(path))
    if kind == "lp":
        from .lpformat import read_lp
        return read_lp(path, name=name)
    if kind == "json":
        from .jsonmodel import read_json
        return read_json(path, name=name)
    return read_mps(path, name=name)


def read_mps(path, name: str | None = None) -> LPProblem:
    """Read an MPS file; prefer the native C++ reader, fall back to pure Python."""
    kind = _suffixes(str(path))
    if kind == "lp":
        from .lpformat import read_lp
        return read_lp(path, name=name)
    if kind == "json":
        from .jsonmodel import read_json
        return read_json(path, name=name)
    if os.environ.get("QENIVO_MPS_PYTHON", "0") != "1":
        try:
            from .native_mps import read_mps_native
            return read_mps_native(path, name=name)
        except Exception:
            pass
    return _read_mps_python(path, name=name)


def _read_mps_python(path, name: str | None = None) -> LPProblem:
    row_idx: dict[str, int] = {}
    row_names: list[str] = []
    row_types: list[str] = []
    obj_row = None
    col_idx: dict[str, int] = {}
    col_names: list[str] = []
    cost = array("d")
    is_int = array("b")
    ri, ci, vv = array("i"), array("i"), array("d")
    rhs: dict[int, float] = {}
    rng: dict[int, float] = {}
    bnd_ops: list[tuple[str, int, float]] = []
    q_entries: list[tuple[int, int, float, bool]] = []   # (i, j, v, mirror)
    c0 = 0.0
    sense = 1.0
    rhs_set = rng_set = bnd_set = None
    in_int = False
    section = None
    prob_name = name or Path(str(path)).name.split(".")[0]
    objname_hint = None

    with _open_text(path) as fh:
        for raw in fh:
            if not raw.strip() or raw[0] == "*":
                continue
            line = raw.rstrip("\n\r")
            if line[0] not in " \t":                       # section header
                tok = line.split()
                head = tok[0].upper()
                section = head
                if head == "NAME" and len(tok) > 1 and name is None:
                    prob_name = tok[1]
                elif head == "OBJSENSE" and len(tok) > 1:
                    sense = -1.0 if tok[1].upper().startswith("MAX") else 1.0
                    section = None
                elif head == "OBJSENSE":
                    section = "OBJSENSE"
                elif head == "OBJNAME" and len(tok) > 1:
                    objname_hint = tok[1]
                elif head in ("SOS", "QCMATRIX", "INDICATORS", "GENERAL"):
                    raise NotImplementedError(f"MPS section {head} not supported")
                elif head == "ENDATA":
                    break
                continue

            t = line.split()
            if section == "COLUMNS":
                nt = len(t)
                if nt >= 3 and t[1] == "'MARKER'":
                    if "'INTORG'" in t[2:]:
                        in_int = True
                    elif "'INTEND'" in t[2:]:
                        in_int = False
                    continue
                if nt < 3 or nt % 2 == 0 or any(t[k] not in row_idx for k in range(1, nt - 1, 2)):
                    tf = _fixed_fields(line)           # names with spaces: fixed-format columns
                    if len(tf) >= 3 and all(tf[k] in row_idx for k in range(1, len(tf) - 1, 2)):
                        t, nt = tf, len(tf)
                cname = t[0]
                j = col_idx.get(cname)
                if j is None:
                    j = len(col_names)
                    col_idx[cname] = j
                    col_names.append(cname)
                    cost.append(0.0)
                    is_int.append(1 if in_int else 0)
                k = -1
                while True:
                    k += 2
                    if k + 1 > len(t) - 1:           # t may switch to fixed fields mid-line
                        break
                    r_k = t[k]
                    i = row_idx.get(r_k)
                    if i is None:
                        t_fix = _fixed_fields(line)
                        if len(t_fix) > k and t_fix[k] in row_idx:
                            t = t_fix
                            r_k = t[k]
                            i = row_idx.get(r_k)
                    if i is None:
                        continue
                    try:
                        v = float(t[k + 1])
                    except (ValueError, IndexError):
                        continue
                    if i >= 0:
                        ri.append(i)
                        ci.append(j)
                        vv.append(v)
                    elif i == _OBJ:
                        cost[j] += v
            elif section == "ROWS":
                if len(t) != 2:
                    t = _fixed_fields(line)
                if len(t) < 2 or not t[0] or not t[1]:
                    raise ValueError(f"malformed MPS ROWS line {line[:120]!r}")
                typ, rname = t[0].upper(), t[1]
                if typ == "N":
                    if obj_row is None and (objname_hint is None or rname == objname_hint):
                        obj_row = rname
                        row_idx[rname] = _OBJ
                    else:
                        row_idx[rname] = _DROP
                else:
                    row_idx[rname] = len(row_names)
                    row_names.append(rname)
                    row_types.append(typ)
            elif section in ("RHS", "RANGES"):
                if len(t) < 2:
                    t = _fixed_fields(line)
                else:                                        # names with spaces: fixed columns
                    body = t[1:] if len(t) % 2 == 1 else t
                    if any(body[k] not in row_idx for k in range(0, len(body) - 1, 2)):
                        t = _fixed_fields(line)
                if len(t) % 2 == 1:                          # leading set name
                    setname, pairs = t[0], t[1:]
                    if section == "RHS":
                        rhs_set = rhs_set or setname
                        if setname != rhs_set:
                            continue
                    else:
                        rng_set = rng_set or setname
                        if setname != rng_set:
                            continue
                else:
                    pairs = t
                for k in range(0, len(pairs) - 1, 2):
                    i = row_idx.get(pairs[k])
                    if i is None:
                        raise ValueError(f"{section} references unknown row {pairs[k]}")
                    v = float(pairs[k + 1])
                    if section == "RHS":
                        if i >= 0:
                            rhs[i] = v
                        elif i == _OBJ:
                            c0 = -v
                    elif i >= 0:
                        rng[i] = v
            elif section == "BOUNDS":
                if not t:
                    raise ValueError(f"malformed MPS BOUNDS line {line[:120]!r}")
                typ = t[0].upper()
                if typ == "SC":
                    raise NotImplementedError("bound type SC not supported")
                if len(t) >= 3 and t[2] in col_idx:
                    setname, cname, val = t[1], t[2], (t[3] if len(t) > 3 else None)
                elif len(t) >= 2 and t[1] in col_idx:
                    setname, cname, val = None, t[1], (t[2] if len(t) > 2 else None)
                else:
                    t = _fixed_fields(line)
                    if len(t) < 3 or not t[2]:
                        raise ValueError(f"malformed MPS BOUNDS line {line[:120]!r}")
                    setname, cname, val = t[1], t[2], (t[3] if len(t) > 3 else None)
                if cname not in col_idx:
                    raise ValueError(f"BOUNDS references unknown column {cname!r}")
                if setname is not None:
                    bnd_set = bnd_set or setname
                    if setname != bnd_set:
                        continue
                try:
                    bound_value = 0.0 if val is None else float(val)
                except ValueError as exc:
                    raise ValueError(f"malformed MPS BOUNDS value {val!r}") from exc
                bnd_ops.append((typ, col_idx[cname], bound_value))
            elif section in ("QUADOBJ", "QSECTION", "QMATRIX"):
                if len(t) < 3:
                    raise ValueError(f"malformed MPS {section} line {line[:120]!r}")
                if t[0] not in col_idx or t[1] not in col_idx:
                    raise ValueError(f"{section} references an unknown column in {line[:120]!r}")
                try:
                    qv = float(t[2])
                except ValueError as exc:
                    raise ValueError(f"malformed MPS {section} coefficient in {line[:120]!r}") from exc
                q_entries.append((col_idx[t[0]], col_idx[t[1]], qv, section != "QMATRIX"))
            elif section == "OBJSENSE":
                sense = -1.0 if t[0].upper().startswith("MAX") else 1.0

    m, n = len(row_names), len(col_names)
    A = sp.coo_matrix((np.frombuffer(vv, dtype=np.float64),
                       (np.frombuffer(ri, dtype=np.int32), np.frombuffer(ci, dtype=np.int32))),
                      shape=(m, n)).tocsr()          # duplicates are summed
    A.eliminate_zeros()

    # Row bounds from type, RHS and RANGES.
    b = np.zeros(m)
    for i, v in rhs.items():
        b[i] = v
    types = np.array(row_types) if m else np.array([], dtype="<U1")
    lc = np.where(types == "L", -INF, b)
    uc = np.where(types == "G", INF, b)
    for i, r in rng.items():
        typ = row_types[i]
        if typ == "E":
            if r > 0:
                uc[i] = b[i] + abs(r)
            elif r < 0:
                lc[i] = b[i] - abs(r)
        elif typ == "L":
            lc[i] = b[i] - abs(r)
        elif typ == "G":
            uc[i] = b[i] + abs(r)

    # Column bounds.
    lx = np.zeros(n)
    ux = np.full(n, INF)
    integer = np.frombuffer(is_int, dtype=np.int8).astype(bool).copy()
    lower_set = np.zeros(n, dtype=bool)
    notes = []
    for typ, j, v in bnd_ops:
        if typ in ("UP", "UI"):
            ux[j] = v
            if v < 0 and not lower_set[j] and lx[j] == 0.0:
                lx[j] = -INF
                notes.append(f"UP<0 on {col_names[j]}: lower bound set to -inf")
            if typ == "UI":
                integer[j] = True
        elif typ in ("LO", "LI"):
            lx[j] = v
            lower_set[j] = True
            if typ == "LI":
                integer[j] = True
        elif typ == "FX":
            lx[j] = ux[j] = v
            lower_set[j] = True
        elif typ == "FR":
            lx[j], ux[j] = -INF, INF
            lower_set[j] = True
        elif typ == "MI":
            lx[j] = -INF
            lower_set[j] = True
        elif typ == "PL":
            ux[j] = INF
        elif typ == "BV":
            lx[j], ux[j] = 0.0, 1.0
            lower_set[j] = True
            integer[j] = True
        else:
            raise NotImplementedError(f"bound type {typ} not supported")

    Q = None
    if q_entries:
        qi, qj, qv = [], [], []
        for i, j, v, mirror in q_entries:
            qi.append(i)
            qj.append(j)
            qv.append(v)
            if mirror and i != j:
                qi.append(j)
                qj.append(i)
                qv.append(v)
        Q = sp.csr_matrix((qv, (qi, qj)), shape=(n, n))

    c = np.frombuffer(cost, dtype=np.float64).copy()
    if sense < 0:
        c, c0 = -c, -c0
        if Q is not None:
            Q = -Q
    return LPProblem(c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=c0, obj_sign=sense,
                     name=prob_name, row_names=row_names, col_names=col_names,
                     integer=integer if integer.any() else None, Q=Q, notes=notes)


# Public alias for tests / forced-Python comparisons.
read_mps_python = _read_mps_python


def _fmt(v: float) -> str:
    return repr(float(v))


def write_mps(prob: LPProblem, path, name: str | None = None) -> None:
    """Write free-format MPS in the user's objective sense (OBJSENSE MAX for maximisation).

    Sections are built with NumPy masks and bulk string joins (not one append per nonzero).
    """
    m, n = prob.m, prob.n
    sgn = float(prob.obj_sign)
    cost = np.asarray(sgn * prob.c, dtype=np.float64)
    c0 = float(sgn * prob.c0)
    rn = list(prob.row_names) if prob.row_names else [f"R{i}" for i in range(m)]
    cn = list(prob.col_names) if prob.col_names else [f"C{j}" for j in range(n)]
    lc, uc = np.asarray(prob.lc, float), np.asarray(prob.uc, float)
    A = sp.csc_matrix(prob.A, dtype=np.float64)
    A.sort_indices()

    parts: list[str] = [f"NAME {name or prob.name or 'LP'}\n"]
    if sgn < 0:
        parts.append("OBJSENSE\n    MAX\n")
    parts.append("ROWS\n N OBJ\n")

    eq = lc == uc
    free = np.isinf(lc) & np.isinf(uc)
    le = np.isinf(lc) & ~np.isinf(uc) & ~eq
    ge = np.isinf(uc) & ~np.isinf(lc) & ~eq
    ranged = ~(eq | free | le | ge)
    rtype = np.full(m, "E", dtype=object)
    rtype[free] = "N"
    rtype[le] = "L"
    rtype[ge | ranged] = "G"
    rhs = np.zeros(m, dtype=np.float64)
    rhs[eq] = lc[eq]
    rhs[le] = uc[le]
    rhs[ge | ranged] = lc[ge | ranged]
    rng_idx = np.flatnonzero(ranged)
    rng_vals = uc[ranged] - lc[ranged]
    parts.append("".join(f" {t} {rn[i]}\n" for i, t in enumerate(rtype)))

    parts.append("COLUMNS\n")
    integer = (prob.integer if prob.integer is not None
               else np.zeros(n, dtype=bool))
    integer = np.asarray(integer, dtype=bool)
    indptr, indices, data = A.indptr, A.indices, A.data
    col_chunks: list[str] = []
    in_int = False
    for j in range(n):
        if integer[j] and not in_int:
            col_chunks.append(" MARKER 'MARKER' 'INTORG'\n")
            in_int = True
        elif not integer[j] and in_int:
            col_chunks.append(" MARKER 'MARKER' 'INTEND'\n")
            in_int = False
        s, e = int(indptr[j]), int(indptr[j + 1])
        lines = []
        if cost[j] != 0.0:
            lines.append(f" {cn[j]} OBJ {_fmt(cost[j])}\n")
        if e > s:
            cj = cn[j]
            lines.extend(f" {cj} {rn[int(i)]} {_fmt(float(v))}\n"
                         for i, v in zip(indices[s:e], data[s:e]))
        elif cost[j] == 0.0:
            lines.append(f" {cn[j]} OBJ 0\n")
        col_chunks.extend(lines)
    if in_int:
        col_chunks.append(" MARKER 'MARKER' 'INTEND'\n")
    parts.append("".join(col_chunks))

    parts.append("RHS\n")
    rhs_lines = []
    if c0 != 0.0:
        rhs_lines.append(f" RHS OBJ {_fmt(-c0)}\n")
    nz = np.flatnonzero(rhs != 0.0)
    rhs_lines.extend(f" RHS {rn[i]} {_fmt(float(rhs[i]))}\n" for i in nz)
    parts.append("".join(rhs_lines))

    if rng_idx.size:
        parts.append("RANGES\n")
        parts.append("".join(f" RNG {rn[int(i)]} {_fmt(float(r))}\n"
                             for i, r in zip(rng_idx, rng_vals)))

    parts.append("BOUNDS\n")
    lx, ux = np.asarray(prob.lx, float), np.asarray(prob.ux, float)
    fx = lx == ux
    fr = np.isinf(lx) & np.isinf(ux)
    mi = np.isinf(lx) & ~fr & ~fx
    lo_set = (~np.isinf(lx)) & (lx != 0.0) & ~fx
    up_set = (~np.isinf(ux)) & ~fx
    bnd = []
    for j in range(n):
        if fx[j]:
            bnd.append(f" FX BND {cn[j]} {_fmt(float(lx[j]))}\n")
            continue
        if fr[j]:
            bnd.append(f" FR BND {cn[j]}\n")
            continue
        if mi[j]:
            bnd.append(f" MI BND {cn[j]}\n")
        elif lo_set[j]:
            bnd.append(f" LO BND {cn[j]} {_fmt(float(lx[j]))}\n")
        if up_set[j]:
            bnd.append(f" UP BND {cn[j]} {_fmt(float(ux[j]))}\n")
    parts.append("".join(bnd))

    if prob.Q is not None and prob.Q.nnz:
        L = sp.tril(sgn * prob.Q).tocoo()
        order = np.lexsort((L.col, L.row))
        qlines = ["QUADOBJ\n"]
        qlines.extend(f" {cn[int(j)]} {cn[int(i)]} {_fmt(float(v))}\n"
                      for i, j, v in zip(L.row[order], L.col[order], L.data[order])
                      if v != 0.0)
        parts.append("".join(qlines))
    parts.append("ENDATA\n")
    Path(path).write_text("".join(parts))

def write_sol(path, prob, x, y=None, status="optimal", kkt=None, has_basis=False,
              col_statuses=None, row_statuses=None):
    """Writes a standard .sol solution file (plain text: status, objective, primal and dual values).

    If ``col_statuses`` / ``row_statuses`` are given (e.g. from the revised
    simplex), they are written verbatim so the file carries the exact basis
    (m basic entries); otherwise the status is inferred from the bounds.
    """
    x = np.asarray(x).reshape(-1)
    m, n = prob.m, prob.n
    if y is None:
        y = np.zeros(m)
    else:
        y = np.asarray(y).reshape(-1)

    Ax = prob.A @ x
    ATy = prob.A.T @ y
    # In internal space, prob.c is minimisation cost.
    red_cost_internal = prob.c - ATy
    if prob.is_qp:
        red_cost_internal += prob.Q @ x

    # Original duals and reduced costs for .sol file:
    row_dual = prob.obj_sign * y
    col_dual = prob.obj_sign * red_cost_internal

    obj = float(prob.c @ x + prob.c0)
    if prob.is_qp:
        obj += float(0.5 * (x @ (prob.Q @ x)))
    obj = prob.obj_sign * obj

    cn = prob.col_names if (prob.col_names and len(prob.col_names) == n) else [f"x{j}" for j in range(n)]
    rn = prob.row_names if (prob.row_names and len(prob.row_names) == m) else [f"c{i}" for i in range(m)]

    lines = [
        "# QENIVO solution file",
        f"model {prob.name or 'UNKNOWN'}",
        f"status {status}",
        f"objective {obj:.16g}",
    ]
    if kkt:
        for k, v in kkt.items():
            lines.append(f"{k} {float(v):.8e}")

    lines.append("begin columns")
    for j in range(n):
        val = x[j]
        rc = col_dual[j]
        st = "unknown"
        if col_statuses is not None and j < len(col_statuses):
            st = col_statuses[j]
        elif has_basis:
            st = "basic"
            if val <= prob.lx[j] + 1e-7:
                st = "at_lower"
            elif val >= prob.ux[j] - 1e-7:
                st = "at_upper"
        lines.append(f"{cn[j]} {val:.16g} {rc:.16g} {st}")
    lines.append("end columns")

    lines.append("begin rows")
    for i in range(m):
        act = Ax[i]
        dual = row_dual[i]
        st = "unknown"
        if row_statuses is not None and i < len(row_statuses):
            st = row_statuses[i]
        elif has_basis:
            st = "basic"
            if act <= prob.lc[i] + 1e-7:
                st = "at_lower"
            elif act >= prob.uc[i] - 1e-7:
                st = "at_upper"
        lines.append(f"{rn[i]} {act:.16g} {dual:.16g} {st}")
    lines.append("end rows")

    Path(path).write_text("\n".join(lines) + "\n")

