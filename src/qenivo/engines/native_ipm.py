"""Sparse-Cholesky interior-point engine ("ipm-native").

The shared library is compiled on first use, the same way as engines/native_simplex.py:
one cache entry per source hash, static on Windows, -fPIC on Linux, and a pure-Python
sparse factor when no compiler is available. The Python path factors the symbolic
Cholesky pattern itself. It does not call scipy.sparse.linalg.

Bounds and the returned dictionary match engines/ipm.py. The normal equations are the
bounded-variable system of Lustig, Marsten and Shanno (Linear Algebra Appl. 152, 1991)
and Mehrotra (SIAM J. Optim. 2, 1992), with Gondzio's centrality correctors
(Comput. Optim. Appl. 6, 1996). A diagonal Hessian is folded into those equations; a
general convex Hessian uses the primal-dual regularised augmented system of Friedlander
and Orban (Math. Program. Comput. 4, 2012). The homogeneous option is the Xu-Hung-Ye
self-dual embedding (Ann. Oper. Res. 62, 1996). Ordering, the elimination tree and the
supernodal LDL' live in native/ipm and are cited there.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..model import Problem
from ._native_abi import native_abi_tag
from .registry import register_engine

_DIR = Path(__file__).resolve().parents[1] / "native" / "ipm"
_LIB: dict = {}


def _sources() -> list[Path]:
    files = [p for p in sorted(_DIR.glob("*.cpp")) if p.name != "test_factor.cpp"]
    files += sorted(_DIR.glob("*.hpp"))
    return files


def _cache_dir() -> Path:
    base = os.environ.get("QENIVO_CACHE") or (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo"
        if os.name == "nt"
        else Path.home() / ".cache" / "qenivo"
    )
    d = Path(base) / "native"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compiler():
    for cxx in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def library():
    """The loaded native library, or None when it cannot be built."""
    if "lib" in _LIB:
        return _LIB["lib"]
    _LIB["lib"], _LIB["reason"] = None, "disabled"
    if os.environ.get("QENIVO_NATIVE", "1") == "0":
        _LIB["reason"] = "QENIVO_NATIVE=0; the Python sparse factor is used"
        return None
    blob = b"".join(p.read_bytes() for p in _sources()) + native_abi_tag()
    h = hashlib.sha256(blob).hexdigest()[:16]
    ext = (
        ".dll"
        if os.name == "nt"
        else (".dylib" if platform.system() == "Darwin" else ".so")
    )
    out = _cache_dir() / f"qenivo_ipm_{h}{ext}"
    srcs = [str(p) for p in _sources() if p.suffix == ".cpp"]
    if not out.exists():                 # a prebuilt library (installer) needs no compiler
        cxx = _compiler()
        if cxx is None:
            _LIB["reason"] = "no C++ compiler found (set CXX); the Python sparse factor is used"
            return None
        cmd = [cxx, "-O3", "-std=c++17", "-shared", "-pthread", "-o", str(out), *srcs]
        if os.name == "nt":
            cmd.append("-static")
        else:
            cmd.append("-fPIC")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _LIB["reason"] = "build failed: " + (r.stderr or "")[-400:]
            return None
    try:
        lib = ctypes.CDLL(str(out))
    except OSError as exc:
        _LIB["reason"] = (
            f"native library could not be loaded ({exc}); the Python sparse factor is used"
        )
        return None
    P = ctypes.POINTER
    lib.nr_ipm_order.restype = ctypes.c_int
    lib.nr_ipm_order.argtypes = [
        ctypes.c_int,
        P(ctypes.c_int),
        P(ctypes.c_int),
        ctypes.c_int,
        P(ctypes.c_int),
    ]
    lib.nr_ipm_symbolic.restype = ctypes.c_int
    lib.nr_ipm_symbolic.argtypes = [
        ctypes.c_int,
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_int),
    ]
    lib.nr_ipm_factor.restype = ctypes.c_void_p
    lib.nr_ipm_factor.argtypes = [
        ctypes.c_int,
        P(ctypes.c_int),
        P(ctypes.c_int),
        P(ctypes.c_double),
        P(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        P(ctypes.c_int),
        P(ctypes.c_double),
    ]
    lib.nr_ipm_solve.restype = ctypes.c_int
    lib.nr_ipm_solve.argtypes = [
        ctypes.c_void_p,
        P(ctypes.c_double),
        P(ctypes.c_double),
    ]
    lib.nr_ipm_free.restype = None
    lib.nr_ipm_free.argtypes = [ctypes.c_void_p]
    _LIB["lib"], _LIB["reason"], _LIB["path"] = lib, "ok", str(out)
    return lib


def status() -> str:
    library()
    return _LIB.get("reason", "unknown")


def _ptr(a, ct):
    return np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(ct))


def _csc_tuple(A):
    C = sp.csc_matrix(A, dtype=np.float64)
    C.sum_duplicates()
    C.sort_indices()
    return (
        C.shape[0],
        np.ascontiguousarray(C.indptr, dtype=np.int32),
        np.ascontiguousarray(C.indices, dtype=np.int32),
        np.ascontiguousarray(C.data, dtype=np.float64),
        C.shape[1],
    )


def _is_perm(perm, n):
    perm = np.asarray(perm, dtype=int)
    return (
        perm.shape == (n,)
        and len(set(perm.tolist())) == n
        and perm.min(initial=0) >= 0
        and perm.max(initial=-1) < n
    )


# ---------------------------------------------------------------- Python sparse factor
def _adjacency(n, indptr, indices):
    adj = [[] for _ in range(n)]
    for j in range(n):
        for p in range(int(indptr[j]), int(indptr[j + 1])):
            i = int(indices[p])
            if i != j and 0 <= i < n:
                adj[j].append(i)
                adj[i].append(j)
    for i in range(n):
        adj[i] = sorted(set(adj[i]))
    return adj


def _amd(adj):
    """Quotient-graph approximate minimum degree (Amestoy, Davis, Duff 1996)."""
    n = len(adj)
    A = [list(row) for row in adj]
    E = [[] for _ in range(n)]
    elem_vars = [[] for _ in range(n)]
    elem_w = [0] * n
    weight = [1] * n
    rep = list(range(n))
    alive = [1] * n
    members = [[i] for i in range(n)]
    degree = [len(A[i]) for i in range(n)]
    mark = [0] * n
    stamp = 1
    order = []

    def find(v):
        r = v
        while rep[r] != r:
            r = rep[r]
        while v != r:
            nxt = rep[v]
            rep[v] = r
            v = nxt
        return r

    def approx(i, remain):
        d = sum(weight[v] for v in A[i])
        for e in E[i]:
            d += elem_w[e] - weight[i]
        if d < 0:
            d = 0
        cap = remain - weight[i]
        return int(min(d, cap if cap > 0 else 0, n))

    def eliminate(p, remain):
        nonlocal stamp
        stamp += 1
        if stamp > 10**9:
            mark[:] = [0] * n
            stamp = 1
        clique = []

        def add(v):
            v = find(v)
            if not alive[v] or v == p or mark[v] == stamp:
                return
            mark[v] = stamp
            clique.append(v)

        for v in A[p]:
            add(v)
        for e in E[p]:
            for v in elem_vars[e]:
                add(v)
        absorbed = set(E[p])
        elem_vars[p] = list(clique)
        elem_w[p] = sum(weight[v] for v in clique)
        for i in clique:
            na = []
            for v in A[i]:
                v = find(v)
                if alive[v] and v != i and v != p and mark[v] != stamp:
                    na.append(v)
            A[i] = sorted(set(na))
            ne = [e for e in E[i] if e != p and e not in absorbed]
            ne.append(p)
            E[i] = sorted(set(ne))
            degree[i] = approx(i, remain)
        buckets = {}
        for i in clique:
            if not alive[i]:
                continue
            key = (tuple(A[i]), tuple(E[i]))
            buckets.setdefault(key, []).append(i)
        for ids in buckets.values():
            for a, i in enumerate(ids):
                if not alive[i]:
                    continue
                for j in ids[a + 1 :]:
                    if alive[j] and A[i] == A[j] and E[i] == E[j]:
                        weight[i] += weight[j]
                        members[i].extend(members[j])
                        members[j] = []
                        alive[j] = 0
                        rep[j] = i
                        degree[i] = approx(i, remain)
        alive[p] = 0
        got = sorted(set(members[p]))
        order.extend(got)
        return remain - len(got)

    remain = n
    guard = 0
    while remain > 0 and guard < n + 2:
        guard += 1
        alive_ids = [i for i in range(n) if alive[i]]
        if not alive_ids:
            break
        dmin = min(degree[i] for i in alive_ids)
        blocked = [0] * n
        batch = []
        for i in alive_ids:
            if degree[i] != dmin or blocked[i]:
                continue
            batch.append(i)
            blocked[i] = 1
            for v in A[i]:
                blocked[find(v)] = 1
            for e in E[i]:
                for v in elem_vars[e]:
                    blocked[find(v)] = 1
            if len(batch) >= 32:
                break
        if not batch:
            break
        for p in batch:
            p = find(p)
            if alive[p]:
                remain = eliminate(p, remain)
    seen = set()
    cleaned = []
    for v in order:
        if 0 <= v < n and v not in seen:
            seen.add(v)
            cleaned.append(v)
    cleaned.extend(i for i in range(n) if i not in seen)
    return cleaned


def _nd(adj):
    """Nested dissection: heavy-edge coarsening, Kernighan-Lin, then Fiduccia-Mattheyses."""
    n = len(adj)
    nodes = list(range(n))

    def amd_subset(block):
        if len(block) <= 1:
            return list(block)
        local = {g: i for i, g in enumerate(block)}
        sub = [[] for _ in block]
        for i, g in enumerate(block):
            for nb in adj[g]:
                j = local.get(nb)
                if j is not None and j != i:
                    sub[i].append(j)
            sub[i] = sorted(set(sub[i]))
        return [block[i] for i in _amd(sub)]

    def induced(block):
        local = {g: i for i, g in enumerate(block)}
        gadj = [[] for _ in block]
        for i, g in enumerate(block):
            for nb in adj[g]:
                j = local.get(nb)
                if j is not None and j > i:
                    gadj[i].append((j, 1))
                    gadj[j].append((i, 1))
        return gadj

    def cut_of(gadj, side):
        cut = 0
        for v, row in enumerate(gadj):
            for u, w in row:
                if v < u and side[v] != side[u]:
                    cut += w
        return cut

    def gains(gadj, side):
        g = [0] * len(gadj)
        for v, row in enumerate(gadj):
            for u, w in row:
                g[v] += w if side[u] != side[v] else -w
        return g

    def initial(gadj, vw):
        nloc = len(gadj)
        side = [1] * nloc

        def bfs(src):
            dist = [-1] * nloc
            q = [src]
            dist[src] = 0
            for v in q:
                for u, _w in gadj[v]:
                    if dist[u] < 0:
                        dist[u] = dist[v] + 1
                        q.append(u)
            return q

        reach = bfs(0) if nloc else []
        reach = bfs(reach[-1] if reach else 0) if nloc else []
        seen = set()
        order = []
        for v in reach:
            if v not in seen:
                seen.add(v)
                order.append(v)
        for v in range(nloc):
            if v not in seen:
                for u in bfs(v):
                    if u not in seen:
                        seen.add(u)
                        order.append(u)
        half = max(1, sum(vw) // 2)
        acc = 0
        for v in order:
            if acc < half:
                side[v] = 0
                acc += vw[v]
        return side

    def kernighan_lin(gadj, side):
        nloc = len(gadj)
        weight = {}
        for v, row in enumerate(gadj):
            for u, w in row:
                if v < u:
                    weight[(v, u)] = w
        for _pass in range(3):
            locked = [0] * nloc
            swaps = []
            delta = 0
            best_delta = 0
            best_len = 0
            for _step in range(max(1, nloc // 2)):
                gain = gains(gadj, side)
                best, ba, bb = None, -1, -1
                for a in range(nloc):
                    if locked[a] or side[a] != 0:
                        continue
                    for b in range(nloc):
                        if locked[b] or side[b] != 1:
                            continue
                        edge = weight.get((a, b) if a < b else (b, a), 0)
                        gpair = gain[a] + gain[b] - 2 * edge
                        if best is None or gpair > best:
                            best, ba, bb = gpair, a, b
                if ba < 0:
                    break
                side[ba], side[bb] = 1, 0
                locked[ba] = locked[bb] = 1
                delta += best
                swaps.append((ba, bb))
                if delta > best_delta:
                    best_delta = delta
                    best_len = len(swaps)
            for a, b in reversed(swaps[best_len:]):
                side[a], side[b] = 0, 1
            if best_len == 0:
                break
        return side

    def fm(gadj, side, vw):
        nloc = len(gadj)
        if nloc < 2:
            return side
        total = sum(vw)
        slack = max(1, total // 5)
        for _pass in range(6):
            gain = gains(gadj, side)
            locked = [0] * nloc
            w0 = sum(vw[v] for v in range(nloc) if side[v] == 0)
            cut0 = cut_of(gadj, side)
            cut = cut0
            best_cut = cut0
            best = list(side)
            moved = False
            for _step in range(nloc):
                pick, pg = -1, None
                for v in range(nloc):
                    if locked[v]:
                        continue
                    nw = w0 + (-vw[v] if side[v] == 0 else vw[v])
                    if nw < total // 2 - slack or nw > total // 2 + slack:
                        continue
                    if pg is None or gain[v] > pg:
                        pick, pg = v, gain[v]
                if pick < 0:
                    break
                old = side[pick]
                g0 = gain[pick]
                locked[pick] = 1
                side[pick] = 1 - old
                w0 += -vw[pick] if old == 0 else vw[pick]
                cut -= g0
                moved = True
                for u, w in gadj[pick]:
                    if locked[u]:
                        continue
                    gain[u] += 2 * w if side[u] == old else -2 * w
                if cut < best_cut:
                    best_cut = cut
                    best = list(side)
            side[:] = best
            if not moved or best_cut >= cut0:
                break
        return side

    def coarsen(gadj, vw):
        nloc = len(gadj)
        match = [-1] * nloc
        order = sorted(range(nloc), key=lambda v: -sum(w for _u, w in gadj[v]))
        for v in order:
            if match[v] != -1:
                continue
            best, bw = -1, -1
            for u, w in gadj[v]:
                if match[u] == -1 and w > bw:
                    best, bw = u, w
            if best >= 0:
                match[v] = best
                match[best] = v
            else:
                match[v] = v
        mapping = [-1] * nloc
        nc = 0
        for v in range(nloc):
            if mapping[v] >= 0:
                continue
            mapping[v] = nc
            if match[v] != v:
                mapping[match[v]] = nc
            nc += 1
        em = [dict() for _ in range(nc)]
        for v, row in enumerate(gadj):
            for u, w in row:
                if v < u and mapping[v] != mapping[u]:
                    a, b = mapping[v], mapping[u]
                    em[a][b] = em[a].get(b, 0) + w
        cadj = [[] for _ in range(nc)]
        cvw = [0] * nc
        for v in range(nloc):
            cvw[mapping[v]] += vw[v]
        for a, row in enumerate(em):
            for b, w in row.items():
                cadj[a].append((b, w))
                cadj[b].append((a, w))
        return cadj, cvw, mapping

    def bisect(gadj, vw):
        levels = [(gadj, vw)]
        maps = []
        while len(levels[-1][0]) > 36:
            cadj, cvw, mapping = coarsen(*levels[-1])
            if len(cadj) < 2 or len(cadj) >= int(len(levels[-1][0]) * 0.9):
                break
            maps.append(mapping)
            levels.append((cadj, cvw))
        gadj_c, vw_c = levels[-1]
        side = initial(gadj_c, vw_c)
        if len(gadj_c) <= 48:
            kernighan_lin(gadj_c, side)
        fm(gadj_c, side, vw_c)
        for level in range(len(levels) - 2, -1, -1):
            up = [side[maps[level][i]] for i in range(len(levels[level][0]))]
            fm(levels[level][0], up, levels[level][1])
            side = up
        return side

    def rec(block, out):
        if len(block) <= 12:
            out.extend(amd_subset(block))
            return
        gadj = induced(block)
        vw = [1] * len(block)
        side = bisect(gadj, vw)
        boundary = [0] * len(block)
        bcount = 0
        for v in range(len(block)):
            if side[v] != 0:
                continue
            if any(side[u] == 1 for u, _w in gadj[v]):
                boundary[v] = 1
                bcount += 1
        other = [0] * len(block)
        ocount = 0
        for v in range(len(block)):
            if side[v] != 1:
                continue
            if any(side[u] == 0 for u, _w in gadj[v]):
                other[v] = 1
                ocount += 1
        if 0 < ocount < bcount:
            boundary, bcount = other, ocount
        part_a, part_b, sep = [], [], []
        for i, g in enumerate(block):
            if boundary[i]:
                sep.append(g)
            elif side[i] == 0:
                part_a.append(g)
            else:
                part_b.append(g)
        if not part_a or not part_b or bcount * 2 > len(block):
            out.extend(amd_subset(block))
            return
        rec(part_a, out)
        rec(part_b, out)
        out.extend(amd_subset(sep))

    out = []
    rec(nodes, out)
    seen = set()
    cleaned = []
    for v in out:
        if 0 <= v < n and v not in seen:
            seen.add(v)
            cleaned.append(v)
    cleaned.extend(i for i in range(n) if i not in seen)
    return cleaned


def _assemble(n, indptr, indices, data, perm):
    inv = np.empty(n, dtype=int)
    inv[np.asarray(perm, dtype=int)] = np.arange(n)
    lower = {}
    upper = {}

    def key(col, row):
        return (int(col) << 32) | int(row)

    if data is None:
        data = np.ones(len(indices), dtype=float)
    for j in range(n):
        for p in range(int(indptr[j]), int(indptr[j + 1])):
            i = int(indices[p])
            if not 0 <= i < n:
                continue
            v = float(data[p])
            if v == 0.0 or not math.isfinite(v):
                continue
            if i >= j:
                k = key(j, i)
                lower[k] = lower.get(k, 0.0) + v
            else:
                k = key(i, j)
                upper[k] = upper.get(k, 0.0) + v
    cols = [[] for _ in range(n)]
    for k, v in lower.items():
        col, row = k >> 32, k & 0xFFFFFFFF
        pc, pr = int(inv[col]), int(inv[row])
        if pr < pc:
            pr, pc = pc, pr
        cols[pc].append((pr, v))
    for k, v in upper.items():
        if k in lower:
            continue
        col, row = k >> 32, k & 0xFFFFFFFF
        pc, pr = int(inv[col]), int(inv[row])
        if pr < pc:
            pr, pc = pc, pr
        cols[pc].append((pr, v))
    strict = [[] for _ in range(n)]
    diag = np.zeros(n)
    rows, vals, colptr = [], [], [0]
    for j in range(n):
        acc = {}
        for r, v in cols[j]:
            acc[r] = acc.get(r, 0.0) + v
        for r in sorted(acc):
            s = acc[r]
            if s == 0.0 or not math.isfinite(s):
                continue
            rows.append(r)
            vals.append(s)
            if r == j:
                diag[j] = s
            elif r > j:
                strict[j].append(r)
        colptr.append(len(rows))
    scale = float(np.max(np.abs(diag))) if n else 1.0
    if not scale:
        scale = 1.0
    return colptr, rows, vals, strict, diag, scale


def _extends(previous, current, column):
    if len(previous) != len(current) + 1:
        return False
    q = 0
    saw = False
    for row in previous:
        if q < len(current) and row == current[q]:
            q += 1
        elif row == column and not saw:
            saw = True
        else:
            return False
    return saw and q == len(current)


def _symbolic(strict):
    n = len(strict)
    upper = [[] for _ in range(n)]
    for j, rows in enumerate(strict):
        for i in rows:
            if j < i < n:
                upper[i].append(j)
    parent = [-1] * n
    ancestor = [-1] * n
    for j in range(n):
        for i in upper[j]:
            r = i
            while ancestor[r] != -1 and ancestor[r] != j:
                nxt = ancestor[r]
                ancestor[r] = j
                r = nxt
            if ancestor[r] == -1:
                ancestor[r] = j
                parent[r] = j
    children = [[] for _ in range(n)]
    for j, pnt in enumerate(parent):
        if pnt >= 0:
            children[pnt].append(j)
    pattern = [[] for _ in range(n)]
    colcount = [1] * n
    for j in range(n):
        acc = set(strict[j])
        for c in children[j]:
            for row in pattern[c]:
                if row > j:
                    acc.add(row)
        acc.discard(j)
        pattern[j] = sorted(acc)
        colcount[j] = len(pattern[j]) + 1
    sn_id = [0] * n
    for j in range(1, n):
        if (
            parent[j - 1] == j
            and colcount[j - 1] == colcount[j] + 1
            and _extends(pattern[j - 1], pattern[j], j)
        ):
            sn_id[j] = sn_id[j - 1]
        else:
            sn_id[j] = sn_id[j - 1] + 1
    ns = (sn_id[-1] + 1) if n else 0
    return parent, colcount, pattern, sn_id, ns


def _factor_python(n, indptr, indices, data, perm, spd, reg):
    colptr, rows, vals, strict, _diag, scale = _assemble(n, indptr, indices, data, perm)
    _parent, _colcount, pattern, _sn, _ns = _symbolic(strict)
    Acols = [dict() for _ in range(n)]
    for j in range(n):
        for p in range(colptr[j], colptr[j + 1]):
            r = rows[p]
            if r >= j:
                Acols[j][r] = Acols[j].get(r, 0.0) + vals[p]
    upd = [[] for _ in range(n)]
    for j in range(n):
        for i in pattern[j]:
            upd[i].append(j)
    L = [dict() for _ in range(n)]
    D = np.zeros(n)
    tiny = max(float(reg), 1e-15 * scale)
    nneg = 0
    reg_added = 0.0
    work = np.zeros(n)
    for k in range(n):
        work[:] = 0.0
        for r, v in Acols[k].items():
            work[r] = v
        for j in upd[k]:
            lkj = L[j].get(k, 0.0)
            coef = lkj * D[j]
            work[k] -= coef * lkj
            for i, lij in L[j].items():
                if i > k:
                    work[i] -= coef * lij
        diag = float(work[k])
        if spd:
            if not diag > tiny:
                reg_added += tiny - diag
                diag = tiny
        else:
            if not abs(diag) > tiny:
                target = math.copysign(tiny, 1.0 if diag >= 0.0 else -1.0)
                reg_added += abs(target - diag)
                diag = target
            if diag < 0.0:
                nneg += 1
        D[k] = diag
        inv = 1.0 / diag
        for i in pattern[k]:
            L[k][i] = work[i] * inv
    return L, D, nneg, reg_added, pattern


def _solve_python(perm, L, D, b):
    n = len(perm)
    perm = np.asarray(perm, dtype=int)
    z = np.asarray(b, dtype=float)[perm].copy()
    for j in range(n):
        zj = z[j]
        for i, lij in L[j].items():
            z[i] -= lij * zj
    z /= D
    for j in range(n - 1, -1, -1):
        acc = z[j]
        for i, lij in L[j].items():
            acc -= lij * z[i]
        z[j] = acc
    x = np.zeros(n)
    x[perm] = z
    return x


class _Factor:
    """One LDL' factor of a symmetric CSC matrix, native or Python."""

    def __init__(self, A, spd=True, reg=0.0, ordering="amd", impl="auto", threads=0):
        n, indptr, indices, data, n2 = _csc_tuple(A)
        if n != n2:
            raise ValueError("the factored matrix must be square")
        self.n = n
        self.ordering = ordering
        self.regularisation = 0.0
        self.nneg = 0
        self._handle = None
        self._py = None
        use_native = impl != "python" and library() is not None
        self.impl = "supernodal-ldl" if use_native else "python-sparse-ldl"
        if n == 0:
            self.perm = np.zeros(0, dtype=np.int32)
            return
        self.perm = order_matrix(
            n, indptr, indices, ordering, impl="native" if use_native else "python"
        )
        if use_native:
            nneg = ctypes.c_int()
            reg_used = ctypes.c_double()
            handle = library().nr_ipm_factor(
                n,
                _ptr(indptr, ctypes.c_int),
                _ptr(indices, ctypes.c_int),
                _ptr(data, ctypes.c_double),
                _ptr(self.perm, ctypes.c_int),
                1 if spd else 0,
                float(reg),
                int(threads),
                ctypes.byref(nneg),
                ctypes.byref(reg_used),
            )
            if not handle:
                raise RuntimeError("native factor failed")
            self._handle = handle
            self._keep = (indptr, indices, data, self.perm)
            self.nneg = int(nneg.value)
            self.regularisation = float(reg_used.value)
        else:
            L, D, nneg, reg_added, _pattern = _factor_python(
                n, indptr, indices, data, self.perm, spd, reg
            )
            self._py = (L, D)
            self.nneg = nneg
            self.regularisation = reg_added
            self._keep = (indptr, indices, data)

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if self.n == 0:
            return np.zeros(0)
        out = np.zeros(self.n)
        if self._handle is not None:
            rc = library().nr_ipm_solve(
                self._handle, _ptr(rhs, ctypes.c_double), _ptr(out, ctypes.c_double)
            )
            if rc != 0:
                raise RuntimeError("native solve failed")
            return out
        L, D = self._py
        return _solve_python(self.perm, L, D, rhs)

    def close(self):
        if self._handle is not None:
            library().nr_ipm_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def order_matrix(n, indptr, indices, method="amd", impl="auto"):
    """Elimination order. ``perm[k]`` is the original index placed in position k."""
    n = int(n)
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    use_native = impl != "python" and library() is not None
    if use_native and impl != "python":
        perm = np.zeros(n, dtype=np.int32)
        code = 0 if method == "amd" else 1
        rc = library().nr_ipm_order(
            n,
            _ptr(indptr, ctypes.c_int),
            _ptr(indices, ctypes.c_int),
            code,
            _ptr(perm, ctypes.c_int),
        )
        if rc == 0 and _is_perm(perm, n):
            return perm
    adj = _adjacency(n, indptr, indices)
    perm = _nd(adj) if method == "nd" else _amd(adj)
    return np.asarray(perm, dtype=np.int32)


def symbolic_matrix(n, indptr, indices, perm=None, impl="auto"):
    """Elimination tree, column counts and fundamental supernode ids of P A P^T."""
    n = int(n)
    indptr = np.ascontiguousarray(indptr, dtype=np.int32)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    if perm is None:
        perm = np.arange(n, dtype=np.int32)
    perm = np.ascontiguousarray(perm, dtype=np.int32)
    use_native = impl != "python" and library() is not None
    if use_native:
        parent = np.zeros(n, dtype=np.int32)
        colcount = np.zeros(n, dtype=np.int32)
        sn_id = np.zeros(n, dtype=np.int32)
        ns = ctypes.c_int()
        library().nr_ipm_symbolic(
            n,
            _ptr(indptr, ctypes.c_int),
            _ptr(indices, ctypes.c_int),
            _ptr(perm, ctypes.c_int),
            _ptr(parent, ctypes.c_int),
            _ptr(colcount, ctypes.c_int),
            _ptr(sn_id, ctypes.c_int),
            ctypes.byref(ns),
        )
        return {
            "parent": parent,
            "colcount": colcount,
            "supernode": sn_id,
            "supernodes": int(ns.value),
        }
    _cp, _rows, _vals, strict, _diag, _scale = _assemble(n, indptr, indices, None, perm)
    parent, colcount, _pattern, sn_id, ns = _symbolic(strict)
    return {
        "parent": np.asarray(parent, dtype=np.int32),
        "colcount": np.asarray(colcount, dtype=np.int32),
        "supernode": np.asarray(sn_id, dtype=np.int32),
        "supernodes": int(ns),
    }


def factor_solve(n, indptr, indices, data, rhs, method="amd", spd=True, impl="auto"):
    """Solve A x = rhs with the sparse LDL' factor. A is symmetric CSC."""
    n = int(n)
    A = sp.csc_matrix((data, indices, indptr), shape=(n, n))
    factor = _Factor(A, spd=spd, ordering=method, impl=impl)
    try:
        return factor.solve(rhs)
    finally:
        factor.close()


# ---------------------------------------------------------------- interior point
def _max_step(v, dv):
    if v.size == 0:
        return 1.0
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _assemble_model(prob: Problem):
    lx, ux, lc, uc = prob.lx, prob.ux, prob.lc, prob.uc
    fixed = lx == ux
    keepc = ~fixed
    xfix = np.where(fixed, lx, 0.0)
    A = prob.A.tocsr()
    shift = np.asarray(A @ xfix).reshape(-1)
    fin_l, fin_u = np.isfinite(lc), np.isfinite(uc)
    free_row = ~fin_l & ~fin_u
    eq = fin_l & fin_u & (np.abs(uc - lc) <= 1e-12 * (1.0 + np.abs(uc)))
    ineq = ~eq & ~free_row
    rows_eq, rows_in = np.flatnonzero(eq), np.flatnonzero(ineq)
    keep_idx = np.flatnonzero(keepc)
    Ac = A[:, keepc]
    k_in = len(rows_in)
    n_keep = int(keepc.sum())
    if len(rows_eq) + k_in == 0:
        K = sp.csr_matrix((0, n_keep + k_in))
    else:
        K = sp.vstack(
            [
                sp.hstack(
                    [Ac[rows_eq], sp.csr_matrix((len(rows_eq), k_in))], format="csr"
                ),
                sp.hstack([Ac[rows_in], -sp.eye(k_in, format="csr")], format="csr"),
            ],
            format="csr",
        )
    b = np.concatenate([uc[rows_eq] - shift[rows_eq], np.zeros(k_in)])
    c = np.concatenate([prob.c[keepc], np.zeros(k_in)])
    lower = np.concatenate([lx[keepc], lc[rows_in] - shift[rows_in]])
    u = np.concatenate([ux[keepc], uc[rows_in] - shift[rows_in]])
    c0 = float(prob.c0 + prob.c @ xfix)
    Q = None
    if prob.is_qp:
        Qfull = prob.Q.tocsr()
        Qk = Qfull[keep_idx][:, keep_idx]
        if np.any(fixed):
            fix_idx = np.flatnonzero(fixed)
            c[:n_keep] = c[:n_keep] + np.asarray(
                Qfull[keep_idx][:, fix_idx] @ xfix[fix_idx]
            ).reshape(-1)
            c0 += 0.5 * float(
                xfix[fix_idx] @ (Qfull[fix_idx][:, fix_idx] @ xfix[fix_idx])
            )
        Q = (
            sp.block_diag((Qk, sp.csr_matrix((k_in, k_in))), format="csr")
            if k_in
            else Qk.tocsr()
        )
    return {
        "K": K,
        "b": b,
        "c": c,
        "l": lower,
        "u": u,
        "keepc": keepc,
        "xfix": xfix,
        "rows_eq": rows_eq,
        "rows_in": rows_in,
        "nx": n_keep,
        "c0": c0,
        "Q": Q,
    }


def _factor_spd(matrix, ordering, impl):
    """Factor a symmetric matrix, adding a diagonal shift until the pivots stay positive."""
    n = matrix.shape[0]
    if n == 0:
        factor = _Factor(matrix, spd=True, ordering=ordering, impl=impl)
        factor.delta = 0.0
        return factor
    base = sp.csc_matrix(matrix, dtype=np.float64)
    diag = np.abs(np.asarray(base.diagonal()).reshape(-1))
    dmax = float(diag.max()) if diag.size else 1.0
    last = None
    for rel in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        delta = rel * max(dmax, 1.0)
        M = base if delta == 0.0 else (base + delta * sp.eye(n, format="csc"))
        if last is not None:
            last.close()
        last = _Factor(M, spd=True, ordering=ordering, impl=impl)
        last.delta = delta
        if last.nneg == 0 and last.regularisation <= 1e-8 * max(dmax, 1.0) * max(n, 1):
            break
    return last


def _mehrotra(F, tol, time_limit, t0, ordering, impl, max_iter, verbose):
    K, b, c, lower, u = F["K"], F["b"], F["c"], F["l"], F["u"]
    Q = F["Q"]
    KT = K.T.tocsr()
    p, q = K.shape
    hl, hu = np.isfinite(lower), np.isfinite(u)
    free = ~hl & ~hu
    general_q = Q is not None and (Q - sp.diags(Q.diagonal())).nnz > 0
    qdiag = np.zeros(q) if Q is None else np.asarray(Q.diagonal()).reshape(-1)
    info = {"regularisation": 0.0, "factor": None, "centrality_corrections": 0}
    if q == 0:
        status = (
            "optimal"
            if np.linalg.norm(b) <= tol * (1.0 + np.linalg.norm(b))
            else "infeasible"
        )
        return status, np.zeros(0), np.zeros(p), 0, info
    if p:
        eye_d = np.ones(q)
        ns0 = _factor_spd((K @ sp.diags(eye_d) @ KT), ordering, impl)
        v = np.asarray(KT @ ns0.solve(b)).reshape(-1)
        y = ns0.solve(K @ c)
        ns0.close()
    else:
        v = np.zeros(q)
        y = np.zeros(0)
    r = c - np.asarray(KT @ y).reshape(-1)
    if Q is not None:
        r = r + np.asarray(Q @ v).reshape(-1)
    tl = np.where(hl, v - lower, 1.0)
    tu = np.where(hu, u - v, 1.0)
    both = hl & hu
    zl = np.where(hl & ~hu, r, 0.0) + np.where(both, np.maximum(r, 0.0), 0.0)
    zu = np.where(hu & ~hl, -r, 0.0) + np.where(both, np.maximum(-r, 0.0), 0.0)
    tmin = min(tl[hl].min(initial=np.inf), tu[hu].min(initial=np.inf))
    zmin = min(zl[hl].min(initial=np.inf), zu[hu].min(initial=np.inf))
    dp = max(-1.5 * tmin, 0.0) if np.isfinite(tmin) else 0.0
    dd = max(-1.5 * zmin, 0.0) if np.isfinite(zmin) else 0.0
    tl, tu = np.where(hl, tl + dp, 1.0), np.where(hu, tu + dp, 1.0)
    zl, zu = np.where(hl, zl + dd, 0.0), np.where(hu, zu + dd, 0.0)
    tz = float(tl[hl] @ zl[hl] + tu[hu] @ zu[hu])
    st, sz = float(tl[hl].sum() + tu[hu].sum()), float(zl[hl].sum() + zu[hu].sum())
    if st > 0 and sz > 0:
        tl, tu = (
            np.where(hl, tl + 0.5 * tz / sz, 1.0),
            np.where(hu, tu + 0.5 * tz / sz, 1.0),
        )
        zl, zu = (
            np.where(hl, zl + 0.5 * tz / st, 0.0),
            np.where(hu, zu + 0.5 * tz / st, 0.0),
        )
    tl = np.where(hl, np.maximum(tl, 1e-8), 1.0)
    tu = np.where(hu, np.maximum(tu, 1e-8), 1.0)
    zl = np.where(hl, np.maximum(zl, 1e-8), 0.0)
    zu = np.where(hu, np.maximum(zu, 1e-8), 0.0)
    bnorm, cnorm = 1.0 + np.linalg.norm(b), 1.0 + np.linalg.norm(c)
    rho = 1e-10
    rho_bumps = 0
    ncomp = int(hl.sum() + hu.sum())
    lf, uf = np.where(hl, lower, 0.0), np.where(hu, u, 0.0)
    best_err, hist, hist_err = np.inf, [], []
    with np.errstate(all="ignore"):
        for it in range(max_iter):
            if time.perf_counter() - t0 > time_limit:
                return "time_limit", v, y, it, info
            rb = b - K @ v
            rc = c - np.asarray(KT @ y).reshape(-1) - zl + zu
            if Q is not None:
                rc = rc + np.asarray(Q @ v).reshape(-1)
            rl = np.where(hl, (v - lower) - tl, 0.0)
            ru = np.where(hu, (u - v) - tu, 0.0)
            mu = float((tl[hl] @ zl[hl] + tu[hu] @ zu[hu]) / max(ncomp, 1))
            quad = 0.5 * float(v @ (Q @ v)) if Q is not None else 0.0
            pobj = float(c @ v) + quad
            dobj = float(b @ y + lf @ zl - uf @ zu) - quad
            gap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
            rp = max(np.linalg.norm(rb), np.linalg.norm(rl), np.linalg.norm(ru)) / bnorm
            rd = np.linalg.norm(rc) / cnorm
            if verbose:
                print(
                    f"PROGRESS ipm it={it} pobj={pobj:+.8e} dobj={dobj:+.8e} "
                    f"rp={rp:.1e} rd={rd:.1e} gap={gap:.1e} mu={mu:.1e}",
                    flush=True,
                )
            if rp <= tol and rd <= tol and gap <= tol:
                return "optimal", v, y, it, info
            err = max(rp, rd, gap)
            best_err = min(best_err, err)
            hist.append(best_err)
            hist_err.append(err)
            # Prefer a fast honest not_proven over burning wall-clock on a dead plateau
            # (CVXQP class: µ collapses while primal residual sticks ~1e-5).
            flat = len(hist_err) >= 12 and hist_err[-1] >= 0.99 * hist_err[-12]
            flat_short = len(hist_err) >= 4 and hist_err[-1] >= 0.99 * hist_err[-4]
            if it >= 15 and flat and rho < 1e-4:
                rho = min(max(rho * 10.0, 1e-10), 1e-4)
                rho_bumps += 1
            # Dense / general Q: exit early so auto→PDQP fallback still has budget.
            if general_q and it >= 5 and flat_short and err > 10 * tol:
                return "stalled", v, y, it, info
            if it >= 35 and flat and (rho_bumps >= 2 or err > 10 * tol):
                return "stalled", v, y, it, info
            if it >= 80 and len(hist) >= 41 and hist[-41] <= 1.01 * best_err:
                return "stalled", v, y, it, info
            if not (np.all(np.isfinite(v)) and np.all(np.isfinite(y))):
                return "numerical_error", v, y, it, info
            if max(np.abs(v).max(initial=0), np.abs(y).max(initial=0)) > 1e15:
                return (
                    (
                        "unbounded"
                        if np.abs(v).max() > np.abs(y).max()
                        else "infeasible"
                    ),
                    v,
                    y,
                    it,
                    info,
                )
            # Free variables have no bound duals: give them an adaptive primal ridge
            # (Friedlander–Orban style) so the KKT matrix stays nonsingular.
            q_scale = max(float(np.max(np.abs(qdiag))), 1.0) if qdiag.size else 1.0
            free_ridge = max(1e-8, 1e-10 * q_scale, rho)
            theta = (
                np.where(hl, zl / np.maximum(tl, 1e-300), 0.0)
                + np.where(hu, zu / np.maximum(tu, 1e-300), 0.0)
                + rho
                + np.where(free, free_ridge, 0.0)
            )
            # Clamp extreme bound dual / slack ratios so LDL' stays sane after µ collapse.
            theta = np.minimum(np.maximum(theta, rho), 1e12)
            if general_q:
                H = (Q + sp.diags(theta)).tocsc()
                hdiag = np.abs(np.asarray(H.diagonal()).reshape(-1))
                scale = max(float(hdiag.max()) if hdiag.size else 0.0, 1.0)
                factor = None
                delta = 0.0
                # Dense / ill-conditioned Q: climb the dual regularisation ladder further.
                for rel in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
                    delta = rel * scale
                    top = sp.hstack([-H, KT], format="csc")
                    bot = sp.hstack(
                        [K.tocsc(), delta * sp.eye(p, format="csc")], format="csc"
                    )
                    M = sp.vstack([top, bot], format="csc")
                    if factor is not None:
                        factor.close()
                    factor = _Factor(
                        M, spd=False, ordering=ordering, impl=impl, reg=1e-14 * scale
                    )
                    if factor.regularisation <= 1e-6 * scale * (p + q):
                        break
                info["regularisation"] = max(
                    info["regularisation"], factor.regularisation + delta
                )
                info["factor"] = factor.impl

                def direction(sl, su, factor=factor):
                    rt = (
                        rc
                        - np.where(hl, (sl - zl * rl) / tl, 0.0)
                        + np.where(hu, (su - zu * ru) / tu, 0.0)
                    )
                    sol = factor.solve(np.concatenate([rt, rb]))
                    dv, dy = sol[:q], sol[q:]
                    dtl = np.where(hl, dv + rl, 0.0)
                    dtu = np.where(hu, -dv + ru, 0.0)
                    dzl = np.where(hl, (sl - zl * dtl) / tl, 0.0)
                    dzu = np.where(hu, (su - zu * dtu) / tu, 0.0)
                    return dv, dy, dtl, dtu, dzl, dzu
            else:
                Dinv = theta + qdiag
                D = 1.0 / Dinv
                if p:
                    factor = _factor_spd(K @ sp.diags(D) @ KT, ordering, impl)
                    info["regularisation"] = max(
                        info["regularisation"], factor.regularisation + factor.delta
                    )
                    info["factor"] = factor.impl
                else:
                    factor = None

                def direction(sl, su, factor=factor, D=D):
                    rt = (
                        rc
                        - np.where(hl, (sl - zl * rl) / tl, 0.0)
                        + np.where(hu, (su - zu * ru) / tu, 0.0)
                    )
                    if p:
                        dy = factor.solve(rb + K @ (D * rt))
                    else:
                        dy = np.zeros(0)
                    dv = D * (np.asarray(KT @ dy).reshape(-1) - rt)
                    dtl = np.where(hl, dv + rl, 0.0)
                    dtu = np.where(hu, -dv + ru, 0.0)
                    dzl = np.where(hl, (sl - zl * dtl) / tl, 0.0)
                    dzu = np.where(hu, (su - zu * dtu) / tu, 0.0)
                    return dv, dy, dtl, dtu, dzl, dzu

            dv, dy, dtl, dtu, dzl, dzu = direction(
                np.where(hl, -tl * zl, 0.0), np.where(hu, -tu * zu, 0.0)
            )
            ap = min(_max_step(tl[hl], dtl[hl]), _max_step(tu[hu], dtu[hu]))
            ad = min(_max_step(zl[hl], dzl[hl]), _max_step(zu[hu], dzu[hu]))
            mu_aff = float(
                (
                    (tl + ap * dtl)[hl] @ (zl + ad * dzl)[hl]
                    + (tu + ap * dtu)[hu] @ (zu + ad * dzu)[hu]
                )
                / max(ncomp, 1)
            )
            sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
            sl = np.where(hl, sigma * mu - tl * zl - dtl * dzl, 0.0)
            su = np.where(hu, sigma * mu - tu * zu - dtu * dzu, 0.0)
            dv, dy, dtl, dtu, dzl, dzu = direction(sl, su)
            for _corr in range(3):
                ap_c = min(_max_step(tl[hl], dtl[hl]), _max_step(tu[hu], dtu[hu]))
                ad_c = min(_max_step(zl[hl], dzl[hl]), _max_step(zu[hu], dzu[hu]))
                mu_t = max(sigma * mu, 0.0)
                lo, hi_b = 0.1 * mu_t, 10.0 * mu_t
                sl2, su2 = sl.copy(), su.copy()
                changed = False
                pred_l = (tl + ap_c * dtl) * (zl + ad_c * dzl)
                pred_u = (tu + ap_c * dtu) * (zu + ad_c * dzu)
                for i in np.flatnonzero(hl):
                    pr = pred_l[i]
                    if pr < lo:
                        sl2[i] = lo - tl[i] * zl[i] - dtl[i] * dzl[i]
                        changed = True
                    elif mu_t > 0 and pr > hi_b:
                        sl2[i] = hi_b - tl[i] * zl[i] - dtl[i] * dzl[i]
                        changed = True
                for i in np.flatnonzero(hu):
                    pr = pred_u[i]
                    if pr < lo:
                        su2[i] = lo - tu[i] * zu[i] - dtu[i] * dzu[i]
                        changed = True
                    elif mu_t > 0 and pr > hi_b:
                        su2[i] = hi_b - tu[i] * zu[i] - dtu[i] * dzu[i]
                        changed = True
                if not changed:
                    break
                dv2, dy2, dtl2, dtu2, dzl2, dzu2 = direction(sl2, su2)
                ap2 = min(_max_step(tl[hl], dtl2[hl]), _max_step(tu[hu], dtu2[hu]))
                ad2 = min(_max_step(zl[hl], dzl2[hl]), _max_step(zu[hu], dzu2[hu]))
                if min(ap2, ad2) + 1e-15 >= min(ap_c, ad_c):
                    dv, dy, dtl, dtu, dzl, dzu = dv2, dy2, dtl2, dtu2, dzl2, dzu2
                    sl, su = sl2, su2
                    info["centrality_corrections"] += 1
                else:
                    break
            if factor is not None:
                factor.close()
            eta = 0.995 if mu > 1e-6 else 0.9995
            ap = min(
                1.0, eta * min(_max_step(tl[hl], dtl[hl]), _max_step(tu[hu], dtu[hu]))
            )
            ad = min(
                1.0, eta * min(_max_step(zl[hl], dzl[hl]), _max_step(zu[hu], dzu[hu]))
            )
            v = v + ap * dv
            tl = np.where(hl, np.maximum(tl + ap * dtl, 1e-300), 1.0)
            tu = np.where(hu, np.maximum(tu + ap * dtu, 1e-300), 1.0)
            y = y + ad * dy
            zl = np.where(hl, np.maximum(zl + ad * dzl, 1e-300), 0.0)
            zu = np.where(hu, np.maximum(zu + ad * dzu, 1e-300), 0.0)
    return "iteration_limit", v, y, max_iter, info


def _to_standard(K, b, c, lower, u):
    """Split bounds so the homogeneous embedding sees A x = b, x >= 0."""
    p, q = K.shape
    m_rows, m_cols, m_vals, cost = [], [], [], []
    shift = np.zeros(q)
    bound_rows = []
    for j in range(q):
        lj, uj = float(lower[j]), float(u[j])
        hl, hu = math.isfinite(lj), math.isfinite(uj)
        if hl and hu:
            shift[j] = lj
            t = len(cost)
            cost.append(float(c[j]))
            m_rows.append(j)
            m_cols.append(t)
            m_vals.append(1.0)
            s = len(cost)
            cost.append(0.0)
            bound_rows.append((t, s, uj - lj))
        elif hl:
            shift[j] = lj
            t = len(cost)
            cost.append(float(c[j]))
            m_rows.append(j)
            m_cols.append(t)
            m_vals.append(1.0)
        elif hu:
            shift[j] = uj
            t = len(cost)
            cost.append(-float(c[j]))
            m_rows.append(j)
            m_cols.append(t)
            m_vals.append(-1.0)
        else:
            tp = len(cost)
            cost.append(float(c[j]))
            m_rows.append(j)
            m_cols.append(tp)
            m_vals.append(1.0)
            tm = len(cost)
            cost.append(-float(c[j]))
            m_rows.append(j)
            m_cols.append(tm)
            m_vals.append(-1.0)
    nt = len(cost)
    M = (
        sp.csc_matrix((m_vals, (m_rows, m_cols)), shape=(q, nt))
        if nt
        else sp.csc_matrix((q, 0))
    )
    b_core = b - K @ shift
    A_core = (K @ M).tocsr() if p else sp.csr_matrix((0, nt))
    extra_r, extra_c, extra_v, extra_b = [], [], [], []
    for t, s, width in bound_rows:
        row = len(extra_b)
        extra_r.extend([row, row])
        extra_c.extend([t, s])
        extra_v.extend([1.0, 1.0])
        extra_b.append(width)
    if extra_b:
        A = sp.vstack(
            [
                A_core,
                sp.csr_matrix((extra_v, (extra_r, extra_c)), shape=(len(extra_b), nt)),
            ],
            format="csr",
        )
        b_std = np.concatenate([b_core, np.asarray(extra_b, dtype=float)])
    else:
        A, b_std = A_core, b_core
    const = float(c @ shift)

    def recover(tvec):
        return shift + np.asarray(M @ tvec).reshape(-1)

    return A, b_std, np.asarray(cost, dtype=float), const, recover


def _hsd(A, b, c, tol, time_limit, t0, ordering, impl, max_iter):
    """Homogeneous self-dual Mehrotra method on A x = b, x >= 0."""
    A = A.tocsr()
    m, n = A.shape
    AT = A.T.tocsr()
    if n == 0:
        ok = np.linalg.norm(b) <= tol
        return ("optimal" if ok else "infeasible"), np.zeros(0), np.zeros(m), 0
    x = np.ones(n)
    s = np.ones(n)
    y = np.zeros(m)
    tau, kappa = 1.0, 1.0
    for it in range(max_iter):
        if time.perf_counter() - t0 > time_limit:
            return "time_limit", x, y, it
        rp = np.asarray(A @ x).reshape(-1) - tau * b
        rd = -np.asarray(AT @ y).reshape(-1) + tau * c - s
        rg = float(b @ y - c @ x - kappa)
        mu = (float(x @ s) + tau * kappa) / (n + 1)
        if tau > 1e-8:
            xs, ys, ss = x / tau, y / tau, s / tau
            rpn = np.linalg.norm(A @ xs - b) / (1.0 + np.linalg.norm(b))
            rdn = np.linalg.norm(c - AT @ ys - ss) / (1.0 + np.linalg.norm(c))
            gap = abs(float(c @ xs - b @ ys)) / (
                1.0 + abs(float(c @ xs)) + abs(float(b @ ys))
            )
            if (
                rpn <= tol
                and rdn <= tol
                and gap <= tol
                and float(np.min(xs)) >= -tol
                and float(np.min(ss)) >= -tol
            ):
                return "optimal", xs, ys, it
        if it > 8 and tau <= 1e-8 * max(1.0, kappa) and kappa > 1e-10:
            yn = y / max(np.linalg.norm(y), 1e-16)
            aty = np.asarray(AT @ yn).reshape(-1)
            if float(b @ yn) > 1e-8 and np.all(aty <= 1e-6 * (1.0 + np.abs(aty))):
                return "infeasible", x, y, it
            xn = x / max(np.linalg.norm(x), 1e-16)
            if (
                float(c @ xn) < -1e-8
                and np.linalg.norm(A @ xn) <= 1e-6
                and np.all(xn >= -1e-8)
            ):
                return "unbounded", x, y, it
        if not (
            np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and math.isfinite(tau)
        ):
            return "numerical_error", x, y, it
        D = np.maximum(x, 1e-16) / np.maximum(s, 1e-16)

        def newton(sigma, dxa, dsa, dtau_a, dkappa_a):
            g = -x * s + sigma * mu - dxa * dsa
            gt = -tau * kappa + sigma * mu - dtau_a * dkappa_a
            rhat = -rd + g / np.maximum(x, 1e-16)
            Dc = D * c
            u = np.asarray(A @ Dc).reshape(-1) + b if m else np.zeros(0)
            pvec = -rp - np.asarray(A @ (D * rhat)).reshape(-1) if m else np.zeros(0)
            alpha = float(c @ (D * c) + kappa / max(tau, 1e-16))
            beta = -rg + gt / max(tau, 1e-16) + float(c @ (D * rhat))
            if m:
                Nmat = (A @ sp.diags(D) @ AT).tocsc()
                factor = _factor_spd(Nmat, ordering, impl)
                try:
                    minv_p = factor.solve(pvec)
                    minv_u = factor.solve(u)
                finally:
                    factor.close()
                vcoef = 2.0 * b - u
                denom = float(vcoef @ minv_u) + alpha
                dtau = (
                    0.0
                    if abs(denom) < 1e-18
                    else (beta - float(vcoef @ minv_p)) / denom
                )
                dy = minv_p + dtau * minv_u
                dx = D * (np.asarray(AT @ dy).reshape(-1) - c * dtau + rhat)
            else:
                dtau = 0.0 if abs(alpha) < 1e-18 else beta / alpha
                dy = np.zeros(0)
                dx = D * (-c * dtau + rhat)
            ds = g / np.maximum(x, 1e-16) - (s / np.maximum(x, 1e-16)) * dx
            dkappa = gt / max(tau, 1e-16) - (kappa / max(tau, 1e-16)) * dtau
            return dx, dy, dtau, ds, dkappa

        dx, dy, dtau, ds, dkappa = newton(0.0, np.zeros(n), np.zeros(n), 0.0, 0.0)
        ap = min(_max_step(x, dx), _max_step(np.array([tau]), np.array([dtau])))
        ad = min(_max_step(s, ds), _max_step(np.array([kappa]), np.array([dkappa])))
        mu_aff = (
            float((x + ap * dx) @ (s + ad * ds))
            + (tau + ap * dtau) * (kappa + ad * dkappa)
        ) / (n + 1)
        sigma = (max(mu_aff, 0.0) / mu) ** 3 if mu > 0 else 0.0
        dx, dy, dtau, ds, dkappa = newton(sigma, dx, ds, dtau, dkappa)
        eta = 0.995 if mu > 1e-6 else 0.9995
        ap = min(
            1.0,
            eta * min(_max_step(x, dx), _max_step(np.array([tau]), np.array([dtau]))),
        )
        ad = min(
            1.0,
            eta
            * min(_max_step(s, ds), _max_step(np.array([kappa]), np.array([dkappa]))),
        )
        x = np.maximum(x + ap * dx, 1e-16)
        s = np.maximum(s + ad * ds, 1e-16)
        y = y + ad * dy
        tau = max(tau + ap * dtau, 1e-16)
        kappa = max(kappa + ad * dkappa, 1e-16)
    return "iteration_limit", x / max(tau, 1e-16), y / max(tau, 1e-16), max_iter


def _recover(prob, F, sc, v, yk):
    xs = F["xfix"].copy()
    xs[F["keepc"]] = v[: F["nx"]]
    ys = np.zeros(prob.m)
    ne = len(F["rows_eq"])
    ys[F["rows_eq"]] = yk[:ne]
    ys[F["rows_in"]] = yk[ne : ne + len(F["rows_in"])]
    x = np.clip(sc.col * xs, prob.lx, prob.ux)
    y = sc.row * ys
    return x, y


def solve_ipm_native(
    prob: Problem,
    tol: float = 1e-8,
    time_limit: float = 3600.0,
    backend: str = "numpy",
    verbose: bool = False,
    homogeneous: bool = False,
    ordering: str = "amd",
    max_iter: int = 200,
    impl: str = "auto",
    **_,
):
    """Interior-point solve. Same status, x, y, iterations, time, regularisation and factor keys as ipm."""
    del backend
    t0 = time.perf_counter()
    if prob.is_mip:
        raise ValueError(
            "ipm-native solves continuous LP and convex QP; integers need a MIP engine"
        )
    if ordering not in ("amd", "nd"):
        raise ValueError("ordering must be 'amd' or 'nd'")
    if prob.n == 0:
        return {
            "status": "optimal",
            "x": np.zeros(0),
            "y": np.zeros(prob.m),
            "iterations": 0,
            "time": 0.0,
            "regularisation": 0.0,
            "factor": None,
        }
    if np.any(prob.lx > prob.ux) or np.any(prob.lc > prob.uc):
        return {
            "status": "infeasible",
            "x": np.clip(np.zeros(prob.n), prob.lx, prob.ux),
            "y": np.zeros(prob.m),
            "iterations": 0,
            "time": time.perf_counter() - t0,
            "regularisation": 0.0,
            "factor": None,
        }
    from .precondition import precondition

    K_s, c_s, lc_s, uc_s, lx_s, ux_s, sc = precondition(
        prob.A,
        prob.c,
        prob.lc,
        prob.uc,
        prob.lx,
        prob.ux,
        geo_iters=12,
        ruiz_iters=10,
        pc_alpha=None,
        bound_obj=False,
    )
    Q_s = None
    if prob.is_qp:
        col = np.asarray(sc.col, dtype=float).reshape(-1)
        Q_s = (sp.diags(col) @ prob.Q @ sp.diags(col)).tocsr()
    scaled = Problem(
        c=c_s,
        A=K_s,
        lc=lc_s,
        uc=uc_s,
        lx=lx_s,
        ux=ux_s,
        c0=prob.c0,
        name=prob.name,
        Q=Q_s,
    )
    F = _assemble_model(scaled)
    if homogeneous and not prob.is_qp:
        A, b_std, c_std, _const, recover = _to_standard(
            F["K"], F["b"], F["c"], F["l"], F["u"]
        )
        status, xs, ys, iters = _hsd(
            A, b_std, c_std, tol, time_limit, t0, ordering, impl, max_iter
        )
        v = recover(xs) if status == "optimal" else np.zeros(F["K"].shape[1])
        yk = (
            ys[: F["K"].shape[0]]
            if ys.size >= F["K"].shape[0]
            else np.zeros(F["K"].shape[0])
        )
        info = {
            "regularisation": 0.0,
            "factor": "supernodal-ldl"
            if library() is not None and impl != "python"
            else "python-sparse-ldl",
        }
    else:
        status, v, yk, iters, info = _mehrotra(
            F, tol, time_limit, t0, ordering, impl, max_iter, verbose
        )
    x, y = _recover(prob, F, sc, v, yk)
    if status == "optimal":
        from ..certify.kkt import kkt_residuals

        if kkt_residuals(prob, x, y)["max_rel"] > 10 * tol:
            status = "numerical_error"
    return {
        "status": status,
        "x": x,
        "y": y,
        "iterations": int(iters),
        "time": time.perf_counter() - t0,
        "regularisation": float(info["regularisation"]),
        "factor": info["factor"],
    }


def solve_registered(prob, tol=1e-6, time_limit=3600.0, backend="numpy", verbose=False, **kw):
    """Registry entry: the factorisation result, checked on the original model."""
    from ..api import _finish

    out = solve_ipm_native(prob, tol=tol, time_limit=time_limit, backend=backend, verbose=verbose, **kw)
    meta = {
        "engine": "ipm-native",
        "backend": "numpy",
        "iterations": out["iterations"],
        "time": out["time"],
        "regularisation": out.get("regularisation"),
        "factor": out.get("factor"),
        "reason": "engine chosen by the caller",
    }
    return _finish(prob, out["status"], out.get("x"), out.get("y"), tol, meta)


register_engine(
    "ipm-native",
    solve_registered,
    classes=("LP", "QP"),
    description="Mehrotra predictor-corrector with a native supernodal LDL' factor",
)
