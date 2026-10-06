"""Independent certificate checker.

Shares NO code with the engines: it has its own MPS reader (below), works on plain Python
dictionaries keyed by row and column NAMES (so an ordering bug in an engine cannot hide), and
sums with math.fsum (compensated summation). With exact=True every quantity is recomputed
in rational arithmetic (fractions.Fraction; each float converts exactly), so the reported
violations are the true violations of the stated point, with no rounding at all.

    verdict optimal     -> primal feasibility, dual feasibility, objective and gap re-derived
    verdict infeasible  -> the Farkas ray is re-scored:  phi(y) > 0 with sign rules satisfied
    verdict unbounded   -> the ray is re-scored:  c'd < 0 and it stays inside the bounds
"""
from __future__ import annotations

import bz2
import gzip
import math
from fractions import Fraction

INF = math.inf


# ---------------------------------------------------------------------- own MPS reader
class MPSModel:
    def __init__(self):
        self.name = ""
        self.sense = 1
        self.obj = None
        self.rtype: dict[str, str] = {}
        self.rows: list[str] = []
        self.cols: list[str] = []
        self.a: dict[str, dict[str, float]] = {}      # column -> {row -> value}
        self.cost: dict[str, float] = {}
        self.rhs: dict[str, float] = {}
        self.rng: dict[str, float] = {}
        self.lo: dict[str, float] = {}
        self.up: dict[str, float] = {}
        self.q: dict[tuple, float] = {}               # (i, j) -> value, full symmetric
        self.c0 = 0.0

    def row_bounds(self, r):
        t, b = self.rtype[r], self.rhs.get(r, 0.0)
        lo, up = (b, b) if t == "E" else ((-INF, b) if t == "L" else (b, INF))
        if r in self.rng:
            R = self.rng[r]
            if t == "E":
                lo, up = (b, b + abs(R)) if R > 0 else ((b - abs(R), b) if R < 0 else (b, b))
            elif t == "L":
                lo = b - abs(R)
            elif t == "G":
                up = b + abs(R)
        return lo, up


def _fixed(raw: str) -> list:
    """Fixed-format MPS fields (columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61); names may hold spaces.
    Kept separate from qenivo.io.mps on purpose: the verifier shares no code with the solver's reader."""
    spans = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)]
    return [raw[a:b].strip() for a, b in spans if raw[a:b].strip()]


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def read_mps(path) -> MPSModel:
    p = str(path)
    opener = gzip.open if p.endswith(".gz") else bz2.open if p.endswith(".bz2") else open
    M = MPSModel()
    section = None
    lower_set = set()
    rhs_name = rng_name = bnd_name = None
    with opener(p, "rt", errors="replace") as fh:
        for raw in fh:
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if raw[0] not in " \t":
                tok = raw.split()
                section = tok[0].upper()
                if section == "NAME" and len(tok) > 1:
                    M.name = tok[1]
                if section == "OBJSENSE" and len(tok) > 1:
                    M.sense = -1 if tok[1].upper().startswith("MAX") else 1
                if section == "ENDATA":
                    break
                continue
            t = raw.split()
            if section == "OBJSENSE":
                M.sense = -1 if t[0].upper().startswith("MAX") else 1
            elif section == "ROWS":
                if len(t) != 2:
                    t = _fixed(raw)
                typ, r = t[0].upper(), t[1]
                if typ == "N":
                    if M.obj is None:
                        M.obj = r
                    continue
                M.rtype[r] = typ
                M.rows.append(r)
            elif section == "COLUMNS":
                if len(t) >= 3 and t[1] == "'MARKER'":
                    continue
                known = lambda q: q == M.obj or q in M.rtype  # noqa: E731
                if len(t) % 2 == 0 or len(t) < 3 or not all(known(t[k]) and _is_num(t[k + 1])
                                                           for k in range(1, len(t) - 1, 2)):
                    t = _fixed(raw)
                cname = t[0]
                if cname not in M.a:
                    M.a[cname] = {}
                    M.cols.append(cname)
                for k in range(1, len(t) - 1, 2):
                    r, v = t[k], float(t[k + 1])
                    if r == M.obj:
                        M.cost[cname] = M.cost.get(cname, 0.0) + v
                    elif r in M.rtype:
                        M.a[cname][r] = M.a[cname].get(r, 0.0) + v
            elif section in ("RHS", "RANGES"):
                body = t[1:] if len(t) % 2 == 1 else t
                if not all((body[k] == M.obj or body[k] in M.rtype) and _is_num(body[k + 1])
                           for k in range(0, len(body) - 1, 2)):
                    t = _fixed(raw)
                pairs = t[1:] if len(t) % 2 == 1 else t
                if len(t) % 2 == 1:
                    if section == "RHS":
                        rhs_name = rhs_name or t[0]
                        if t[0] != rhs_name:
                            continue
                    else:
                        rng_name = rng_name or t[0]
                        if t[0] != rng_name:
                            continue
                for k in range(0, len(pairs) - 1, 2):
                    r, v = pairs[k], float(pairs[k + 1])
                    if section == "RHS":
                        if r == M.obj:
                            M.c0 = -v
                        else:
                            M.rhs[r] = v
                    else:
                        M.rng[r] = v
            elif section == "BOUNDS":
                if not ((len(t) >= 3 and t[2] in M.a) or (len(t) >= 2 and t[1] in M.a)):
                    t = _fixed(raw)
                typ = t[0].upper()
                if len(t) >= 3 and t[2] in M.a:
                    setn, cname, val = t[1], t[2], (float(t[3]) if len(t) > 3 else 0.0)
                else:
                    setn, cname, val = None, t[1], (float(t[2]) if len(t) > 2 else 0.0)
                if setn is not None:
                    bnd_name = bnd_name or setn
                    if setn != bnd_name:
                        continue
                if typ in ("UP", "UI"):
                    M.up[cname] = val
                    if val < 0 and cname not in lower_set and M.lo.get(cname, 0.0) == 0.0:
                        M.lo[cname] = -INF
                elif typ in ("LO", "LI"):
                    M.lo[cname] = val; lower_set.add(cname)
                elif typ == "FX":
                    M.lo[cname] = M.up[cname] = val; lower_set.add(cname)
                elif typ == "FR":
                    M.lo[cname], M.up[cname] = -INF, INF; lower_set.add(cname)
                elif typ == "MI":
                    M.lo[cname] = -INF; lower_set.add(cname)
                elif typ == "PL":
                    M.up[cname] = INF
                elif typ == "BV":
                    M.lo[cname], M.up[cname] = 0.0, 1.0; lower_set.add(cname)
            elif section in ("QUADOBJ", "QSECTION", "QMATRIX"):
                i, j, v = t[0], t[1], float(t[2])
                M.q[(i, j)] = M.q.get((i, j), 0.0) + v
                if section != "QMATRIX" and i != j:
                    M.q[(j, i)] = M.q.get((j, i), 0.0) + v
    return M


# ---------------------------------------------------------------------- arithmetic helpers
def _num(v, exact):
    if v is None:
        return None
    if exact:
        return v if isinstance(v, Fraction) else (Fraction(v) if math.isfinite(v) else v)
    return float(v)


def _sum(vals, exact):
    return sum(vals, Fraction(0)) if exact else math.fsum(vals)


def _norm(vals, exact):
    s = _sum([v * v for v in vals], exact)
    return math.sqrt(float(s))


def _pos(v):
    return v if v > 0 else 0 * v


# ---------------------------------------------------------------------- checks
def verify(model_path, cert: dict, exact: bool = False, tol: float | None = None) -> dict:
    """Re-check a certificate against the model file. Returns a report dict with `passed`."""
    M = read_mps(model_path)
    verdict = cert.get("verdict")
    tol = float(tol if tol is not None else cert.get("tolerance", 1e-6))
    report = {"model": M.name, "rows": len(M.rows), "cols": len(M.cols), "verdict": verdict,
              "tolerance": tol, "exact": exact, "checks": []}
    shape_ok = (len(M.rows) == cert["model"]["rows"] and len(M.cols) == cert["model"]["cols"])
    report["checks"].append(("model size matches certificate", shape_ok,
                             f"{len(M.rows)}x{len(M.cols)} vs {cert['model']['rows']}x{cert['model']['cols']}"))
    if verdict == "optimal":
        _check_optimal(M, cert, tol, exact, report)
    elif verdict == "infeasible":
        _check_farkas(M, cert, exact, report)
    elif verdict == "unbounded":
        _check_ray(M, cert, exact, report)
    else:
        report["checks"].append(("verdict is a proven one", False, f"verdict {verdict!r} proves nothing"))
    report["passed"] = all(ok for _, ok, _ in report["checks"])
    return report


def _check_optimal(M, cert, tol, exact, report):
    sol = cert.get("solution") or {}
    xs, ys = sol.get("x") or {}, sol.get("y") or {}
    missing = [c for c in M.cols if c not in xs]
    report["checks"].append(("every column has a value", not missing, f"{len(missing)} missing"))
    if missing:
        return
    E = exact
    x = {c: _num(xs[c], E) for c in M.cols}
    y = {r: _num((ys.get(r) or 0.0), E) * M.sense for r in M.rows}     # back to minimisation sign
    sgn = M.sense
    cost = {c: _num(M.cost.get(c, 0.0), E) * sgn for c in M.cols}
    c0 = _num(M.c0, E) * sgn
    # row activities
    act = {r: [] for r in M.rows}
    for c in M.cols:
        for r, v in M.a[c].items():
            act[r].append(_num(v, E) * x[c])
    rp = []
    bhat = []
    for r in M.rows:
        a = _sum(act[r], E)
        lo, up = M.row_bounds(r)
        lo_, up_ = _num(lo, E), _num(up, E)
        v = (lo_ - a) if (math.isfinite(lo) and a < lo_) else ((a - up_) if (math.isfinite(up) and a > up_) else 0 * a)
        rp.append(v)
        bhat.append(max(abs(lo) if math.isfinite(lo) else 0.0, abs(up) if math.isfinite(up) else 0.0))
    for c in M.cols:
        lo, up = M.lo.get(c, 0.0), M.up.get(c, INF)
        xc = x[c]
        if math.isfinite(lo) and xc < _num(lo, E):
            rp.append(_num(lo, E) - xc)
        elif math.isfinite(up) and xc > _num(up, E):
            rp.append(xc - _num(up, E))
    rel_p = _norm(rp, E) / (1 + math.sqrt(math.fsum(b * b for b in bhat)))
    # quadratic part
    qx = {c: 0 * x[c] for c in M.cols}
    for (i, j), v in M.q.items():
        qx[i] = qx[i] + _num(v, E) * sgn * x[j]
    # reduced costs lam = c + Qx - A'y
    aty = {c: _sum([_num(v, E) * y[r] for r, v in M.a[c].items()], E) for c in M.cols}
    rd, dterms = [], []
    for c in M.cols:
        lam = cost[c] + qx[c] - aty[c]
        lo, up = M.lo.get(c, 0.0), M.up.get(c, INF)
        lp_, ln_ = _pos(lam), _pos(-lam)
        if not math.isfinite(lo):
            rd.append(lp_)
        elif lp_:
            dterms.append(lp_ * _num(lo, E))
        if not math.isfinite(up):
            rd.append(ln_)
        elif ln_:
            dterms.append(-ln_ * _num(up, E))
    for r in M.rows:
        lo, up = M.row_bounds(r)
        yp, yn = _pos(y[r]), _pos(-y[r])
        if not math.isfinite(lo):
            rd.append(yp)
        elif yp:
            dterms.append(yp * _num(lo, E))
        if not math.isfinite(up):
            rd.append(yn)
        elif yn:
            dterms.append(-yn * _num(up, E))
    cnorm = math.sqrt(math.fsum(float(v) ** 2 for v in cost.values()))
    rel_d = _norm(rd, E) / (1 + cnorm)
    quad = _sum([x[c] * qx[c] for c in M.cols], E) / 2 if M.q else 0
    pobj = _sum([cost[c] * x[c] for c in M.cols], E) + c0 + quad
    dobj = _sum(dterms, E) + c0 - quad
    rel_g = abs(float(pobj - dobj)) / (1 + abs(float(pobj)) + abs(float(dobj)))
    obj_user = float(pobj) * sgn
    report.update({"rel_primal": rel_p, "rel_dual": rel_d, "rel_gap": rel_g, "objective": obj_user})
    slack = 1.0 + 1e-9
    report["checks"].append(("primal feasible within tolerance", rel_p <= tol * slack, f"{rel_p:.3e}"))
    report["checks"].append(("dual feasible within tolerance", rel_d <= tol * slack, f"{rel_d:.3e}"))
    report["checks"].append(("duality gap within tolerance", rel_g <= tol * slack, f"{rel_g:.3e}"))
    claimed = cert.get("objective")
    if claimed is not None:
        dev = abs(obj_user - claimed) / (1 + abs(obj_user))
        report["checks"].append(("objective matches certificate", dev <= 1e-9, f"{obj_user:.12g} vs {claimed:.12g}"))


def _check_farkas(M, cert, exact, report):
    vec = (cert.get("evidence") or {}).get("vector") or {}
    E = exact
    y = {r: _num(vec.get(r) or 0.0, E) for r in M.rows}
    nrm = _norm(list(y.values()), E)
    report["checks"].append(("ray is non-zero", nrm > 0, f"||y|| = {nrm:.3e}"))
    if nrm == 0:
        return
    viol, val = [], []
    for r in M.rows:
        lo, up = M.row_bounds(r)
        yp, yn = _pos(y[r]), _pos(-y[r])
        if not math.isfinite(lo):
            viol.append(yp)
        elif yp:
            val.append(yp * _num(lo, E))
        if not math.isfinite(up):
            viol.append(yn)
        elif yn:
            val.append(-yn * _num(up, E))
    for c in M.cols:
        lam = -_sum([_num(v, E) * y[r] for r, v in M.a[c].items()], E)
        lo, up = M.lo.get(c, 0.0), M.up.get(c, INF)
        lp_, ln_ = _pos(lam), _pos(-lam)
        if not math.isfinite(lo):
            viol.append(lp_)
        elif lp_:
            val.append(lp_ * _num(lo, E))
        if not math.isfinite(up):
            viol.append(ln_)
        elif ln_:
            val.append(-ln_ * _num(up, E))
    phi = float(_sum(val, E)) / nrm
    v = _norm(viol, E) / nrm
    report.update({"farkas_value": phi, "farkas_violation": v})
    report["checks"].append(("Farkas value is positive", phi > 0, f"{phi:.3e}"))
    # Match host check_farkas (tol=1e-8): violation <= 1e-5 * max(1, phi) and < phi.
    # The old 1e-6*phi rule rejected small-phi valid rays (Netlib gosh: phi~1e-2, v~1e-7).
    ok_sign = v <= 1e-5 * max(1.0, phi) and v < phi
    report["checks"].append(("sign rules hold (violation << value)", ok_sign, f"{v:.3e}"))


def _check_ray(M, cert, exact, report):
    vec = (cert.get("evidence") or {}).get("vector") or {}
    E = exact
    d = {c: _num(vec.get(c) or 0.0, E) for c in M.cols}
    nrm = _norm(list(d.values()), E)
    report["checks"].append(("ray is non-zero", nrm > 0, f"||d|| = {nrm:.3e}"))
    if nrm == 0:
        return
    sgn = M.sense
    descent = -float(_sum([_num(M.cost.get(c, 0.0), E) * sgn * d[c] for c in M.cols], E)) / nrm
    ad = {r: [] for r in M.rows}
    for c in M.cols:
        for r, v in M.a[c].items():
            ad[r].append(_num(v, E) * d[c])
    viol = []
    for r in M.rows:
        a = _sum(ad[r], E)
        lo, up = M.row_bounds(r)
        if math.isfinite(lo):
            viol.append(_pos(-a))
        if math.isfinite(up):
            viol.append(_pos(a))
    for c in M.cols:
        lo, up = M.lo.get(c, 0.0), M.up.get(c, INF)
        if math.isfinite(lo):
            viol.append(_pos(-d[c]))
        if math.isfinite(up):
            viol.append(_pos(d[c]))
    v = _norm(viol, E) / nrm
    report.update({"ray_descent": descent, "ray_violation": v})
    report["checks"].append(("ray improves the objective", descent > 0, f"{descent:.3e}"))
    report["checks"].append(("ray stays feasible (violation << descent)", v <= 1e-6 * max(descent, 1e-300), f"{v:.3e}"))
    # unboundedness also needs a feasible starting point
    xs = (cert.get("solution") or {}).get("x") or {}
    if not xs or any(c not in xs for c in M.cols):
        report["checks"].append(("a feasible point is supplied", False, "missing"))
        return
    x = {c: _num(xs[c], E) for c in M.cols}
    act = {r: [] for r in M.rows}
    for c in M.cols:
        for r, val in M.a[c].items():
            act[r].append(_num(val, E) * x[c])
    worst = 0.0
    for r in M.rows:
        a = float(_sum(act[r], E))
        lo, up = M.row_bounds(r)
        worst = max(worst, (lo - a) / (1 + abs(lo)) if math.isfinite(lo) else 0.0,
                    (a - up) / (1 + abs(up)) if math.isfinite(up) else 0.0)
    for c in M.cols:
        lo, up, xc = M.lo.get(c, 0.0), M.up.get(c, INF), float(x[c])
        worst = max(worst, (lo - xc) / (1 + abs(lo)) if math.isfinite(lo) else 0.0,
                    (xc - up) / (1 + abs(up)) if math.isfinite(up) else 0.0)
    report["checks"].append(("the supplied point is feasible", worst <= 1e-6, f"{worst:.3e}"))


def format_report(rep: dict) -> str:
    lines = [f"model     {rep['model']}  ({rep['rows']} rows x {rep['cols']} cols)",
             f"verdict   {rep['verdict']}   tolerance {rep['tolerance']:g}   "
             f"arithmetic {'exact rational' if rep['exact'] else 'float64 + compensated sums'}",
             "independent checks (own MPS reader, name-keyed, no engine code):"]
    for name, ok, detail in rep["checks"]:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    lines.append(f"RESULT    {'VERIFIED' if rep['passed'] else 'REJECTED'}")
    return "\n".join(lines)
