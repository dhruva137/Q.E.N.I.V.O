"""CPLEX LP format reader and writer (bounds, general/binary, ranges)."""
from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..model import Problem as LPProblem

INF = math.inf

_SEC_HEAD = re.compile(
    r"^\s*(minimize|minimise|maximize|maximise|minimum|maximum|"
    r"subject\s+to|such\s+that|st|s\.t\.?|"
    r"bounds?|generals?|gen|binaries|binary|bin|integers?|"
    r"semi-continuous|semis|end)\s*$",
    re.I,
)
_OBJ_HEAD = re.compile(r"^\s*(minimize|minimise|maximize|maximise|minimum|maximum)\b", re.I)
_NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_name(t: str) -> bool:
    if not t or t in ("+", "-", "<=", ">=", "=", "<", ">", "≤", "≥", "−"):
        return False
    if _NUM.match(t):
        return False
    return True


def _open_text(path):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", errors="replace")
    return open(p, "r", errors="replace")


def _norm_sec(line: str) -> str | None:
    m = _SEC_HEAD.match(line.strip())
    if not m:
        return None
    t = re.sub(r"\s+", " ", m.group(1).strip().lower())
    if t in ("minimize", "minimise", "minimum"):
        return "min"
    if t in ("maximize", "maximise", "maximum"):
        return "max"
    if t in ("subject to", "such that", "st", "s.t", "s.t."):
        return "st"
    if t in ("bound", "bounds"):
        return "bounds"
    if t in ("general", "generals", "gen", "integer", "integers"):
        return "general"
    if t in ("binary", "binaries", "bin"):
        return "binary"
    if t in ("semi-continuous", "semis"):
        return "semi"
    if t == "end":
        return "end"
    return None


def _strip_comment(line: str) -> str:
    if "\\" in line:
        line = line.split("\\", 1)[0]
    return line.strip()


def _tokenize_expr(s: str) -> list[str]:
    """Tokenize a linear expression / constraint body into numbers, signs, names, ops."""
    s = s.replace("<=", " ≤ ").replace(">=", " ≥ ").replace("=", " = ")
    s = s.replace("<", " < ").replace(">", " > ")
    s = s.replace("+", " + ").replace("-", " - ")
    return [t for t in s.split() if t]


def _parse_lin(tokens: list[str]) -> tuple[dict[str, float], float]:
    """Parse tokens of a linear form into coefs and constant."""
    coefs: dict[str, float] = {}
    const = 0.0
    i = 0
    sign = 1.0
    while i < len(tokens):
        t = tokens[i]
        if t in ("+", "−"):
            sign = 1.0
            i += 1
            continue
        if t == "-":
            sign = -1.0
            i += 1
            continue
        if t in ("≤", "≥", "=", "<", ">"):
            break
        if _NUM.match(t):
            num = sign * float(t)
            i += 1
            if i < len(tokens) and _is_name(tokens[i]):
                name = tokens[i]
                coefs[name] = coefs.get(name, 0.0) + num
                i += 1
            else:
                const += num
            sign = 1.0
            continue
        if _is_name(t):
            coefs[t] = coefs.get(t, 0.0) + sign * 1.0
            sign = 1.0
            i += 1
            continue
        raise ValueError(f"bad LP token {t!r} in {' '.join(tokens)}")
    return coefs, const


def _parse_constraint(line: str):
    """Return (name, coefs, lo, hi)."""
    name = None
    body = line
    if ":" in line:
        left, right = line.split(":", 1)
        if _is_name(left.strip()) or left.strip().isalnum():
            name = left.strip()
            body = right.strip()
    # ranged: lo <= expr <= hi
    m = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|-inf|\+?inf)\s*<=\s*(.+?)\s*<=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|-inf|\+?inf)\s*$",
        body, re.I | re.S)
    if m:
        lo, mid, hi = _num(m.group(1)), m.group(2).strip(), _num(m.group(3))
        coefs, const = _parse_lin(_tokenize_expr(mid))
        return name, coefs, lo - const, hi - const
    for op in ("<=", ">=", "=", "<", ">"):
        if op in body:
            left, right = body.split(op, 1)
            coefs, const = _parse_lin(_tokenize_expr(left))
            rhs = float(right.strip()) - const
            if op == "=":
                return name, coefs, rhs, rhs
            if op in ("<=", "<"):
                return name, coefs, -INF, rhs
            return name, coefs, rhs, INF
    raise ValueError(f"constraint without sense: {line}")


def _parse_bound(line: str):
    """Return (name, lo, hi) or None."""
    s = line.strip()
    if not s:
        return None
    low = s.lower()
    if low.endswith(" free"):
        return s[:-5].strip(), -INF, INF
    parts = s.split()
    if len(parts) == 2 and parts[1].lower() == "free":
        return parts[0], -INF, INF
    # lo <= x <= hi
    m = re.match(
        r"^\s*(\S+)\s*<=\s*(\S+)\s*<=\s*(\S+)\s*$", s)
    if m and not _is_name(m.group(1)):
        return m.group(2), _num(m.group(1)), _num(m.group(3))
    m = re.match(r"^\s*(\S+)\s*=\s*(\S+)\s*$", s)
    if m and _is_name(m.group(1)):
        v = _num(m.group(2))
        return m.group(1), v, v
    m = re.match(r"^\s*(\S+)\s*(<=|>=|<|>)\s*(\S+)\s*$", s)
    if m and _is_name(m.group(1)):
        name, op, v = m.group(1), m.group(2), _num(m.group(3))
        if op in ("<=", "<"):
            return name, None, v
        return name, v, None
    m = re.match(r"^\s*(\S+)\s*(<=|<)\s*(\S+)\s*$", s)
    if m and not _is_name(m.group(1)) and _is_name(m.group(3)):
        return m.group(3), _num(m.group(1)), None
    raise ValueError(f"bad bounds line: {line}")


def _num(s: str) -> float:
    t = s.strip().lower()
    if t in ("inf", "+inf", "+infinity", "infinity"):
        return INF
    if t in ("-inf", "-infinity"):
        return -INF
    return float(t)


def read_lp(path, name: str | None = None) -> LPProblem:
    lines = [_strip_comment(L) for L in _open_text(path).readlines()]
    lines = [L for L in lines if L]
    sense = 1.0
    section = None
    obj_chunks: list[str] = []
    row_lines: list[str] = []
    bound_lines: list[str] = []
    generals: list[str] = []
    binaries: list[str] = []
    prob_name = name or Path(str(path)).name.split(".")[0]
    i = 0
    while i < len(lines):
        sec = _norm_sec(lines[i])
        if sec in ("min", "max"):
            sense = -1.0 if sec == "max" else 1.0
            section = "obj"
            # remainder of the header line after Minimize/Maximize may hold obj start
            rest = _OBJ_HEAD.sub("", lines[i], count=1).strip()
            if rest:
                obj_chunks.append(rest)
            i += 1
            continue
        if sec == "st":
            section = "st"
            i += 1
            continue
        if sec == "bounds":
            section = "bounds"
            i += 1
            continue
        if sec == "general":
            section = "general"
            i += 1
            continue
        if sec == "binary":
            section = "binary"
            i += 1
            continue
        if sec == "semi":
            raise NotImplementedError("semi-continuous variables not supported in LP reader")
        if sec == "end":
            break
        if section == "obj":
            obj_chunks.append(lines[i])
        elif section == "st":
            row_lines.append(lines[i])
        elif section == "bounds":
            bound_lines.append(lines[i])
        elif section == "general":
            generals.extend(lines[i].split())
        elif section == "binary":
            binaries.extend(lines[i].split())
        i += 1

    obj_text = " ".join(obj_chunks)
    if ":" in obj_text:
        left, right = obj_text.split(":", 1)
        if _is_name(left.strip()) or left.strip().isalnum():
            obj_text = right
    obj_coefs, c0 = _parse_lin(_tokenize_expr(obj_text)) if obj_text.strip() else ({}, 0.0)

    rows = []
    for rl in row_lines:
        rname, coefs, lo, hi = _parse_constraint(rl)
        rows.append((rname or f"R{len(rows)}", coefs, lo, hi))

    bounds: dict[str, list] = {}
    bound_order: list[str] = []
    for bl in bound_lines:
        b = _parse_bound(bl)
        if b is None:
            continue
        vname, lo, hi = b
        if vname not in bounds:
            bound_order.append(vname)
        cur = bounds.setdefault(vname, [0.0, INF])
        if lo is not None:
            cur[0] = lo
        if hi is not None:
            cur[1] = hi

    col_names: list[str] = []
    col_idx: dict[str, int] = {}

    def ensure(v: str):
        if v and v not in col_idx:
            col_idx[v] = len(col_names)
            col_names.append(v)

    # Prefer Bounds declaration order so MPS -> LP -> MPS keeps column indices.
    for v in bound_order:
        ensure(v)
    for v in obj_coefs:
        ensure(v)
    for _, coefs, _, _ in rows:
        for v in coefs:
            ensure(v)
    for v in generals + binaries:
        ensure(v)

    n, m = len(col_names), len(rows)
    c = np.zeros(n)
    for v, a in obj_coefs.items():
        c[col_idx[v]] = a
    ri, ci, vv = [], [], []
    lc = np.zeros(m)
    uc = np.zeros(m)
    row_names = []
    for irow, (rname, coefs, lo, hi) in enumerate(rows):
        row_names.append(rname)
        lc[irow], uc[irow] = lo, hi
        for v, a in coefs.items():
            if a == 0.0:
                continue
            ri.append(irow); ci.append(col_idx[v]); vv.append(a)
    A = sp.csr_matrix((vv, (ri, ci)), shape=(m, n)) if n else sp.csr_matrix((m, 0))
    A.eliminate_zeros()

    lx = np.zeros(n)
    ux = np.full(n, INF)
    integer = np.zeros(n, dtype=bool)
    for v, (lo, hi) in bounds.items():
        j = col_idx[v]
        lx[j], ux[j] = lo, hi
    for v in binaries:
        j = col_idx[v]
        lx[j], ux[j] = 0.0, 1.0
        integer[j] = True
    for v in generals:
        integer[col_idx[v]] = True

    if sense < 0:
        c, c0 = -c, -c0
    return LPProblem(
        c=c, A=A, lc=lc, uc=uc, lx=lx, ux=ux, c0=c0, obj_sign=sense,
        name=prob_name, row_names=row_names, col_names=col_names,
        integer=integer if integer.any() else None,
    )


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return repr(float(v))


def _term(coef: float, name: str, first: bool) -> str:
    if coef == 0.0:
        return ""
    a = abs(coef)
    body = name if a == 1.0 else f"{_fmt(a)} {name}"
    if first:
        return body if coef > 0 else "-" + body
    return (" + " if coef > 0 else " - ") + body



def _safe_name(name: str, prefix: str = "V") -> str:
    s = re.sub(r"[^\w]", "_", str(name))
    if not s:
        s = prefix
    if not (s[0].isalpha() or s[0] == "_"):
        s = prefix + s
    return s


def _uniq(names):
    seen = {}
    out = []
    for nm in names:
        k = seen.get(nm, 0)
        seen[nm] = k + 1
        out.append(nm if k == 0 else f"{nm}_{k}")
    return out


def write_lp(prob: LPProblem, path, name: str | None = None) -> None:
    m, n = prob.m, prob.n
    sgn = prob.obj_sign
    cost = sgn * prob.c
    c0 = sgn * prob.c0
    rn0, cn0 = prob.names()
    rn = _uniq([_safe_name(x, "R") for x in rn0])
    cn = _uniq([_safe_name(x, "C") for x in cn0])
    lines = [f"\\ {name or prob.name or 'LP'}", "Maximize" if sgn < 0 else "Minimize"]
    bits, first = [], True
    for j in range(n):
        if cost[j] == 0.0:
            continue
        bits.append(_term(float(cost[j]), cn[j], first))
        first = False
    if c0 != 0.0:
        bits.append(_fmt(c0) if first else (f" + {_fmt(c0)}" if c0 > 0 else f" - {_fmt(-c0)}"))
    if not bits:
        bits = ["0"]
    lines.append("obj: " + "".join(bits))
    lines.append("Subject To")
    A = prob.A.tocsr()
    for i in range(m):
        lo, hi = float(prob.lc[i]), float(prob.uc[i])
        s, e = A.indptr[i], A.indptr[i + 1]
        first, parts = True, []
        for j, v in zip(A.indices[s:e], A.data[s:e]):
            parts.append(_term(float(v), cn[j], first))
            first = False
        body = "".join(parts) if parts else "0"
        if lo == hi:
            lines.append(f" {rn[i]}: {body} = {_fmt(lo)}")
        elif np.isinf(lo) and not np.isinf(hi):
            lines.append(f" {rn[i]}: {body} <= {_fmt(hi)}")
        elif not np.isinf(lo) and np.isinf(hi):
            lines.append(f" {rn[i]}: {body} >= {_fmt(lo)}")
        elif np.isinf(lo) and np.isinf(hi):
            continue
        else:
            lines.append(f" {rn[i]}: {_fmt(lo)} <= {body} <= {_fmt(hi)}")
    lines.append("Bounds")
    integer = prob.integer if prob.integer is not None else np.zeros(n, dtype=bool)
    binaries, generals = [], []
    for j in range(n):
        lo, hi = float(prob.lx[j]), float(prob.ux[j])
        if integer[j] and lo == 0.0 and hi == 1.0:
            binaries.append(cn[j])
            # still emit a placeholder so column order is preserved on read
            lines.append(f" 0 <= {cn[j]} <= 1")
            continue
        if np.isinf(lo) and np.isinf(hi):
            lines.append(f" {cn[j]} free")
        elif lo == hi:
            lines.append(f" {cn[j]} = {_fmt(lo)}")
        else:
            lo_s = "-inf" if np.isinf(lo) else _fmt(lo)
            hi_s = "+inf" if np.isinf(hi) else _fmt(hi)
            lines.append(f" {lo_s} <= {cn[j]} <= {hi_s}")
        if integer[j]:
            generals.append(cn[j])
    if generals:
        lines += ["General", " " + " ".join(generals)]
    if binaries:
        lines += ["Binary", " " + " ".join(binaries)]
    lines.append("End")
    Path(path).write_text("\n".join(lines) + "\n")
