"""Dense symmetric indefinite factorisation for KKT systems.

Bunch-Kaufman LDL^T with 1x1 and 2x2 pivots (J. R. Bunch and L. Kaufman,
Some stable methods for calculating inertia and solving symmetric linear
systems, Mathematics of Computation 31, 1977). The inertia, read from the
signs of the eigenvalues of the block-diagonal D, is what the filter
interior-point method uses when it regularises a Newton matrix (Waechter
and Biegler, Mathematical Programming 106, 2006).

The pivoting rules below are the ones in Golub and Van Loan for that
factorisation. This is not a translation of LAPACK's ``dsytrf``.
"""
from __future__ import annotations

import numpy as np

_ALPHA = (1.0 + np.sqrt(17.0)) / 8.0


class LDLError(RuntimeError):
    """The factorisation could not be applied to this matrix."""


class LDLFactor:
    """P A P^T = L D L^T for a symmetric A, with a solve against the original A."""

    def __init__(self, matrix: np.ndarray, order: np.ndarray, factors: np.ndarray,
                 blocks: list[tuple[int, int]], zero: float, scale: float):
        self.n = int(matrix.shape[0])
        self.order = order
        self.M = factors
        self.blocks = blocks
        self.zero = zero
        self.scale = scale
        self.K = matrix
        self.inertia = _inertia(factors, blocks, zero)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A x = b with one step of iterative refinement."""
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.shape != (self.n,):
            raise ValueError(f"right-hand side has length {b.shape[0]}, expected {self.n}")
        x = self._solve_once(b)
        if self.n == 0:
            return x
        r = b - self.K @ x
        nr = float(np.linalg.norm(r))
        if nr > 1e-12 * (1.0 + float(np.linalg.norm(b))) and np.all(np.isfinite(x)):
            dx = self._solve_once(r)
            if np.all(np.isfinite(dx)):
                x = x + dx
        return x

    def _solve_once(self, b: np.ndarray) -> np.ndarray:
        n = self.n
        if n == 0:
            return np.zeros(0)
        y = np.array(b[self.order], dtype=np.float64, copy=True)
        M = self.M
        k = 0
        for size, _pos in self.blocks:
            if size == 1:
                if k + 1 < n:
                    y[k + 1:] -= M[k + 1:, k] * y[k]
                k += 1
            else:
                if k + 2 < n:
                    y[k + 2:] -= M[k + 2:, k] * y[k] + M[k + 2:, k + 1] * y[k + 1]
                k += 2
        k = 0
        for size, _pos in self.blocks:
            if size == 1:
                d = M[k, k]
                if abs(d) <= self.zero:
                    y[k] = np.nan
                else:
                    y[k] = y[k] / d
                k += 1
            else:
                y[k:k + 2] = _solve2(M[k:k + 2, k:k + 2], y[k:k + 2], self.zero)
                k += 2
        for size, pos in reversed(self.blocks):
            if size == 1:
                if pos + 1 < n:
                    y[pos] -= float(np.dot(M[pos + 1:, pos], y[pos + 1:]))
            else:
                if pos + 2 < n:
                    y[pos:pos + 2] -= M[pos + 2:, pos:pos + 2].T @ y[pos + 2:]
        x = np.empty(n)
        x[self.order] = y
        return x


def factor(A: np.ndarray) -> LDLFactor:
    """Factor a real symmetric matrix. Only the symmetric part of ``A`` is used."""
    M0 = np.asarray(A, dtype=np.float64)
    if M0.ndim != 2 or M0.shape[0] != M0.shape[1]:
        raise ValueError("KKT factorisation expects a square matrix")
    n = M0.shape[0]
    K = (M0 + M0.T) * 0.5
    scale = float(np.max(np.abs(K))) if n else 1.0
    zero = 1e-14 * max(1.0, scale)
    M = K.copy()
    order = np.arange(n)
    blocks: list[tuple[int, int]] = []
    k = 0
    while k < n:
        if k == n - 1:
            blocks.append((1, k))
            k += 1
            continue
        col = M[k + 1:, k]
        lam = float(np.max(np.abs(col))) if col.size else 0.0
        r = k + 1 + int(np.argmax(np.abs(col))) if col.size else k + 1
        akk = float(M[k, k])
        if lam <= zero and abs(akk) <= zero:
            M[k + 1:, k] = 0.0
            blocks.append((1, k))
            k += 1
            continue
        use_1x1 = abs(akk) >= _ALPHA * lam
        swap_with = None
        two = False
        if not use_1x1:
            colr = np.abs(M[k:, r]).copy()
            colr[r - k] = 0.0
            sigma = float(colr.max()) if colr.size else 0.0
            if abs(akk) * sigma >= _ALPHA * lam * lam:
                use_1x1 = True
            elif abs(float(M[r, r])) >= _ALPHA * sigma:
                use_1x1 = True
                swap_with = r
            else:
                two = True
                swap_with = r
        if use_1x1:
            if swap_with is not None:
                _swap(M, order, k, swap_with)
            _eliminate_1(M, k, zero)
            blocks.append((1, k))
            k += 1
        elif two:
            if swap_with != k + 1:
                _swap(M, order, k + 1, swap_with)
            _eliminate_2(M, k, zero)
            blocks.append((2, k))
            k += 2
        else:
            raise LDLError("pivot selection did not choose a block")
    return LDLFactor(K, order, M, blocks, zero, scale)


def _swap(M: np.ndarray, order: np.ndarray, i: int, j: int) -> None:
    if i == j:
        return
    M[[i, j], :] = M[[j, i], :]
    M[:, [i, j]] = M[:, [j, i]]
    order[i], order[j] = order[j], order[i]


def _eliminate_1(M: np.ndarray, k: int, zero: float) -> None:
    n = M.shape[0]
    d = float(M[k, k])
    if k + 1 >= n:
        return
    col = M[k + 1:, k].copy()
    if abs(d) <= zero:
        M[k + 1:, k] = 0.0
        return
    M[k + 1:, k + 1:] -= np.outer(col, col) / d
    M[k + 1:, k] = col / d
    _symmetrise_trailing(M, k + 1)


def _eliminate_2(M: np.ndarray, k: int, zero: float) -> None:
    n = M.shape[0]
    D = (M[k:k + 2, k:k + 2] + M[k:k + 2, k:k + 2].T) * 0.5
    M[k:k + 2, k:k + 2] = D
    if k + 2 >= n:
        return
    B = M[k + 2:, k:k + 2].copy()
    a, b, c = float(D[0, 0]), float(D[0, 1]), float(D[1, 1])
    det = a * c - b * b
    if abs(det) <= zero * max(1.0, abs(a) + abs(c) + abs(b)):
        M[k + 2:, k:k + 2] = 0.0
        return
    inv = np.array([[c, -b], [-b, a]], dtype=np.float64) / det
    Lblk = B @ inv
    M[k + 2:, k + 2:] -= Lblk @ B.T
    M[k + 2:, k:k + 2] = Lblk
    _symmetrise_trailing(M, k + 2)


def _symmetrise_trailing(M: np.ndarray, k: int) -> None:
    S = M[k:, k:]
    M[k:, k:] = (S + S.T) * 0.5


def _solve2(D: np.ndarray, rhs: np.ndarray, zero: float) -> np.ndarray:
    a, b, c = float(D[0, 0]), float(D[0, 1]), float(D[1, 1])
    det = a * c - b * b
    if abs(det) <= zero * max(1.0, abs(a) + abs(c) + abs(b)):
        return np.array([np.nan, np.nan])
    return np.array([(c * rhs[0] - b * rhs[1]) / det, (-b * rhs[0] + a * rhs[1]) / det])


def _inertia(M: np.ndarray, blocks: list[tuple[int, int]], zero: float) -> tuple[int, int, int]:
    pos = neg = zer = 0
    for size, k in blocks:
        if size == 1:
            p, g, z = _sign(float(M[k, k]), zero)
        else:
            p, g, z = _inertia2(M[k:k + 2, k:k + 2], zero)
        pos += p
        neg += g
        zer += z
    return pos, neg, zer


def _sign(d: float, zero: float) -> tuple[int, int, int]:
    if d > zero:
        return 1, 0, 0
    if d < -zero:
        return 0, 1, 0
    return 0, 0, 1


def _inertia2(D: np.ndarray, zero: float) -> tuple[int, int, int]:
    a, b, c = float(D[0, 0]), float(D[0, 1]), float(D[1, 1])
    tr = a + c
    det = a * c - b * b
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    s = float(np.sqrt(disc))
    e1 = 0.5 * (tr + s)
    e2 = 0.5 * (tr - s)
    tol = zero * max(1.0, abs(a) + abs(c) + abs(b))
    p1, n1, z1 = _sign(e1, tol)
    p2, n2, z2 = _sign(e2, tol)
    return p1 + p2, n1 + n2, z1 + z2
