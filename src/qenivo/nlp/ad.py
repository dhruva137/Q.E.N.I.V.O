"""Automatic differentiation on the expression tape.

Gradients are reverse mode (the Wengert-list adjoint sweep). Hessian-vector
products are forward-over-reverse: a directional derivative is carried in the
forward sweep and the adjoint sweep differentiates that derivative
(B. A. Pearlmutter, Fast exact multiplication by the Hessian, Neural
Computation 6, 1994).

Sparse Hessians are recovered from those products. The adjacency graph of the
Hessian is star-coloured; a distance-2 colouring is used when the greedy star
colouring is not valid. Either colouring lets every nonzero be read from a
Hessian-vector product with the sum of the corresponding colour class
(A. H. Gebremedhin, F. Manne and A. Pothen, What color is your Jacobian?
Graph coloring for computing derivatives, SIAM Review 47, 2005).

jax, autograd, CasADi and torch are not used.
"""
from __future__ import annotations

import numpy as np

from .expr import Expr, Graph


class ADError(RuntimeError):
    """A derivative did not match its Hessian-vector product."""


def forward_values(graph: Graph, x: np.ndarray, dot: np.ndarray | None):
    """Evaluate every tape node. If ``dot`` is set, also the directional derivative."""
    n_ops = len(graph.ops)
    val = np.empty(n_ops)
    dval = np.zeros(n_ops) if dot is not None else None
    for i, op in enumerate(graph.ops):
        kind = op[0]
        if kind == "var":
            val[i] = x[op[1]]
            if dval is not None:
                dval[i] = dot[op[1]]
        elif kind == "const":
            val[i] = op[1]
        elif kind == "add":
            a, b = op[1], op[2]
            val[i] = val[a] + val[b]
            if dval is not None:
                dval[i] = dval[a] + dval[b]
        elif kind == "sub":
            a, b = op[1], op[2]
            val[i] = val[a] - val[b]
            if dval is not None:
                dval[i] = dval[a] - dval[b]
        elif kind == "mul":
            a, b = op[1], op[2]
            val[i] = val[a] * val[b]
            if dval is not None:
                dval[i] = dval[a] * val[b] + val[a] * dval[b]
        elif kind == "div":
            a, b = op[1], op[2]
            val[i] = val[a] / val[b]
            if dval is not None:
                dval[i] = (dval[a] * val[b] - val[a] * dval[b]) / (val[b] * val[b])
        elif kind == "neg":
            a = op[1]
            val[i] = -val[a]
            if dval is not None:
                dval[i] = -dval[a]
        elif kind == "powc":
            a, p = op[1], op[2]
            val[i], d1, _d2 = _pow_all(val[a], p)
            if dval is not None:
                dval[i] = d1 * dval[a]
        elif kind == "exp":
            a = op[1]
            val[i] = np.exp(val[a])
            if dval is not None:
                dval[i] = val[i] * dval[a]
        elif kind == "log":
            a = op[1]
            val[i] = np.log(val[a])
            if dval is not None:
                dval[i] = dval[a] / val[a]
        elif kind == "sqrt":
            a = op[1]
            val[i] = np.sqrt(val[a])
            if dval is not None:
                dval[i] = 0.5 * dval[a] / val[i]
        elif kind == "sin":
            a = op[1]
            val[i] = np.sin(val[a])
            if dval is not None:
                dval[i] = np.cos(val[a]) * dval[a]
        elif kind == "cos":
            a = op[1]
            val[i] = np.cos(val[a])
            if dval is not None:
                dval[i] = -np.sin(val[a]) * dval[a]
        else:
            raise ADError(f"unknown operation {kind}")
    return val, dval


def _pow_all(va: float, p: float) -> tuple[float, float, float]:
    """Value, first derivative and second derivative of t |-> t**p."""
    if p == 2.0:
        return va * va, 2.0 * va, 2.0
    if p == 3.0:
        v2 = va * va
        return v2 * va, 3.0 * v2, 6.0 * va
    if p == 1.0:
        return va, 1.0, 0.0
    if p == 0.0:
        return 1.0, 0.0, 0.0
    if p == 0.5:
        s = np.sqrt(va)
        return s, 0.5 / s, -0.25 / (s ** 3)
    ip = int(round(p)) if abs(p - round(p)) < 1e-12 and abs(p) <= 12 else None
    if ip is not None:
        val = va ** ip
    else:
        val = va ** p
    if va == 0.0 and p < 2.0:
        d1 = 0.0 if p > 1.0 else np.nan
        d2 = 2.0 if p == 2.0 else (0.0 if p > 2.0 else np.nan)
        return val, d1, d2
    d1 = p * va ** (p - 1.0)
    d2 = p * (p - 1.0) * va ** (p - 2.0)
    return val, d1, d2


def _reverse(graph: Graph, val: np.ndarray, dval: np.ndarray | None, node: int, want_hvp: bool):
    n_ops = len(graph.ops)
    adj = np.zeros(n_ops)
    adj[node] = 1.0
    hadj = np.zeros(n_ops) if want_hvp else None
    for i in range(n_ops - 1, -1, -1):
        ai = adj[i]
        hi = hadj[i] if want_hvp else 0.0
        if ai == 0.0 and hi == 0.0:
            continue
        op = graph.ops[i]
        kind = op[0]
        if kind in ("var", "const"):
            continue
        if kind == "add":
            a, b = op[1], op[2]
            adj[a] += ai
            adj[b] += ai
            if want_hvp:
                hadj[a] += hi
                hadj[b] += hi
        elif kind == "sub":
            a, b = op[1], op[2]
            adj[a] += ai
            adj[b] -= ai
            if want_hvp:
                hadj[a] += hi
                hadj[b] -= hi
        elif kind == "mul":
            a, b = op[1], op[2]
            adj[a] += ai * val[b]
            adj[b] += ai * val[a]
            if want_hvp:
                hadj[a] += hi * val[b] + ai * dval[b]
                hadj[b] += hi * val[a] + ai * dval[a]
        elif kind == "div":
            a, b = op[1], op[2]
            va, vb = val[a], val[b]
            vb2 = vb * vb
            fa = 1.0 / vb
            fb = -va / vb2
            adj[a] += ai * fa
            adj[b] += ai * fb
            if want_hvp:
                hadj[a] += hi * fa + ai * (-dval[b] / vb2)
                hadj[b] += hi * fb + ai * (-dval[a] / vb2 + 2.0 * va * dval[b] / (vb2 * vb))
        elif kind == "neg":
            a = op[1]
            adj[a] -= ai
            if want_hvp:
                hadj[a] -= hi
        elif kind == "powc":
            a, p = op[1], op[2]
            _v, fp, fpp = _pow_all(val[a], p)
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += hi * fp + ai * fpp * dval[a]
        elif kind == "exp":
            a = op[1]
            fp = val[i]
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += (hi + ai * dval[a]) * fp
        elif kind == "log":
            a = op[1]
            va = val[a]
            fp = 1.0 / va
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += hi * fp + ai * (-dval[a] / (va * va))
        elif kind == "sqrt":
            a = op[1]
            s = val[i]
            fp = 0.5 / s
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += hi * fp + ai * (-0.25 * dval[a] / (s ** 3))
        elif kind == "sin":
            a = op[1]
            fp = np.cos(val[a])
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += hi * fp + ai * (-np.sin(val[a]) * dval[a])
        elif kind == "cos":
            a = op[1]
            fp = -np.sin(val[a])
            adj[a] += ai * fp
            if want_hvp:
                hadj[a] += hi * fp + ai * (-np.cos(val[a]) * dval[a])
        else:
            raise ADError(f"unknown operation {kind}")
    g = np.zeros(graph.n_var)
    hv = np.zeros(graph.n_var) if want_hvp else None
    for i, op in enumerate(graph.ops):
        if op[0] == "var":
            g[op[1]] += adj[i]
            if want_hvp:
                hv[op[1]] += hadj[i]
    return g, hv


def gradient(expr: Expr, x: np.ndarray) -> np.ndarray:
    """Gradient of a scalar expression with respect to the graph variables."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    val, _ = forward_values(expr.graph, x, None)
    g, _ = _reverse(expr.graph, val, None, expr.node, False)
    return g


def hvp(expr: Expr, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Hessian-vector product ∇²f(x) v by forward-over-reverse mode."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    val, dval = forward_values(expr.graph, x, v)
    _g, hv = _reverse(expr.graph, val, dval, expr.node, True)
    return hv


def sparsity(expr: Expr) -> set[tuple[int, int]]:
    """Undirected edges of the Hessian sparsity pattern, as pairs i < j."""
    _dep, edges = _pattern(expr.graph, expr.node, {})
    return edges


def hessian(expr: Expr, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Dense Hessian recovered by graph colouring. ``info`` records the colouring."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    def hv(v):
        return hvp(expr, x, v)

    return hessian_from_hvp(expr.graph.n_var, sparsity(expr), hv)


def hessian_from_hvp(n: int, edges: set[tuple[int, int]], hv_fn) -> tuple[np.ndarray, dict]:
    """Recover a symmetric Hessian from products with colour-class seeds.

    ``hv_fn(v)`` must return H v. An edge that a star colouring does not
    isolate is probed with a unit vector so the returned matrix still matches
    the products; ``info['n_products']`` counts every product.
    """
    color, adj = star_colour(n, edges)
    n_colours = (max(color) + 1) if n and color else 0
    products = []
    for c in range(n_colours):
        seed = np.zeros(n)
        for i in range(n):
            if color[i] == c:
                seed[i] = 1.0
        products.append(np.asarray(hv_fn(seed), dtype=np.float64))
    H = np.zeros((n, n))
    extra = 0
    for i in range(n):
        if n_colours:
            H[i, i] = products[color[i]][i]
    for i, j in edges:
        ci, cj = color[i], color[j]
        ni = [k for k in adj[i] if color[k] == cj]
        nj = [k for k in adj[j] if color[k] == ci]
        if len(ni) == 1 and ni[0] == j:
            hij = products[cj][i]
        elif len(nj) == 1 and nj[0] == i:
            hij = products[ci][j]
        else:
            e = np.zeros(n)
            e[j] = 1.0
            hij = float(hv_fn(e)[i])
            extra += 1
        H[i, j] = H[j, i] = hij
    info = {"n_colours": n_colours, "n_products": n_colours + extra, "n": n, "edges": len(edges)}
    if n:
        v = np.ones(n)
        v[::2] = -1.0
        pred = H @ v
        true = np.asarray(hv_fn(v), dtype=np.float64)
        err = float(np.linalg.norm(pred - true))
        info["check"] = err
        if err > 1e-6 * (1.0 + float(np.linalg.norm(true))):
            raise ADError(f"coloured Hessian disagrees with a Hessian-vector product by {err:.3e}")
    return H, info


def star_colour(n: int, edges: set[tuple[int, int]]) -> tuple[list[int], list[list[int]]]:
    """Star colouring of an undirected graph, or distance-2 colouring if that fails.

    A star colouring is a proper colouring in which every path on four vertices
    uses at least three colours (Gebremedhin, Manne and Pothen). Distance-2
    colouring is a stronger colouring from the same paper and is always a star
    colouring, so recovery stays valid.
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    color = [-1] * n
    for v in range(n):
        forbidden: set[int] = set()
        for w in adj[v]:
            if color[w] >= 0:
                forbidden.add(color[w])
        for w in adj[v]:
            if color[w] < 0:
                continue
            for u in adj[w]:
                if u == v or color[u] < 0 or color[u] == color[w]:
                    continue
                for t in adj[v]:
                    if t != w and color[t] == color[u]:
                        forbidden.add(color[u])
                        break
        c = 0
        while c in forbidden:
            c += 1
        color[v] = c
    if not _is_star(adj, color):
        color = _distance2(adj)
    return color, adj


def _distance2(adj: list[list[int]]) -> list[int]:
    n = len(adj)
    color = [-1] * n
    for v in range(n):
        forbidden: set[int] = set()
        for w in adj[v]:
            if color[w] >= 0:
                forbidden.add(color[w])
            for u in adj[w]:
                if color[u] >= 0:
                    forbidden.add(color[u])
        c = 0
        while c in forbidden:
            c += 1
        color[v] = c
    return color


def _is_star(adj: list[list[int]], color: list[int]) -> bool:
    n = len(adj)
    for v in range(n):
        for w in adj[v]:
            if color[v] == color[w]:
                return False
    for b in range(n):
        for a in adj[b]:
            for c in adj[b]:
                if c == a:
                    continue
                for d in adj[c]:
                    if d == b or d == a:
                        continue
                    if len({color[a], color[b], color[c], color[d]}) < 3:
                        return False
    return True


def _pattern(graph: Graph, node: int, memo: dict) -> tuple[set[int], set[tuple[int, int]]]:
    if node in memo:
        return memo[node]
    op = graph.ops[node]
    kind = op[0]
    if kind == "var":
        result = ({op[1]}, set())
    elif kind == "const":
        result = (set(), set())
    elif kind in ("add", "sub"):
        da, ea = _pattern(graph, op[1], memo)
        db, eb = _pattern(graph, op[2], memo)
        result = (da | db, ea | eb)
    elif kind == "neg":
        result = _pattern(graph, op[1], memo)
    elif kind == "mul":
        da, ea = _pattern(graph, op[1], memo)
        db, eb = _pattern(graph, op[2], memo)
        edges = set(ea) | set(eb)
        for i in da:
            for j in db:
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        result = (da | db, edges)
    elif kind == "div":
        da, ea = _pattern(graph, op[1], memo)
        db, eb = _pattern(graph, op[2], memo)
        edges = set(ea) | set(eb) | _clique(db)
        for i in da:
            for j in db:
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        result = (da | db, edges)
    elif kind == "powc":
        da, ea = _pattern(graph, op[1], memo)
        p = op[2]
        if p == 0.0:
            result = (set(), set())
        elif p == 1.0:
            result = (da, ea)
        else:
            result = (da, set(ea) | _clique(da))
    elif kind in ("exp", "log", "sqrt", "sin", "cos"):
        da, ea = _pattern(graph, op[1], memo)
        result = (da, set(ea) | _clique(da))
    else:
        raise ADError(f"unknown operation {kind}")
    memo[node] = result
    return result


def _clique(deps: set[int]) -> set[tuple[int, int]]:
    items = sorted(deps)
    return {(items[a], items[b]) for a in range(len(items)) for b in range(a + 1, len(items))}


def lagrangian_hvp(objective: Expr, constraints: list[Expr], lam: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Product of the Lagrangian Hessian with ``v``. Linear constraints contribute nothing."""
    hv = hvp(objective, x, v)
    for w, c in zip(lam, constraints):
        if w != 0.0:
            hv = hv + float(w) * hvp(c, x, v)
    return hv


def lagrangian_hessian(objective: Expr, constraints: list[Expr], lam: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hessian of f + λ·c by the same colouring as :func:`hessian`."""
    edges: set[tuple[int, int]] = set(sparsity(objective))
    for c in constraints:
        edges |= sparsity(c)
    lam = np.asarray(lam, dtype=np.float64)

    def hv(v):
        return lagrangian_hvp(objective, constraints, lam, x, v)

    return hessian_from_hvp(objective.graph.n_var, edges, hv)
