"""Exact rational LP (engine ``exact-lp``).

Arithmetic is ours: base-2**32 limbs, schoolbook multiplication, Karatsuba above
eight limbs (Karatsuba and Ofman, 1962), and Stein's binary GCD (1967). Rationals
stay in lowest terms. No GMP and no Boost.Multiprecision.

The final basis is solved by iterative refinement to exactness: a float64
factorisation, an exact residual, a working-precision correction, and continued
fraction reconstruction, accepted only when the residual is the zero rational.
That is the basis step of

    A. Gleixner, D. E. Steffy and K. Wolter, "Iterative refinement for linear
    programming", Mathematical Programming Computation, 2016,

with the residual correction as in N. J. Higham, *Accuracy and Stability of
Numerical Algorithms* (SIAM). Reconstruction is the continued-fraction
convergent (Hardy and Wright). If refinement cannot prove the float candidate
basis, an exact revised simplex with Bland's least-index rule (Bland, Mathematics
of Operations Research, 1977) produces one that it can. The rational certificate
records the basic columns, x, y, and the reduced costs so a caller can rebuild
A_B x_B = b.

When no C++ compiler is available (or ``QENIVO_NATIVE=0``), the same algorithms
run on Python ``int`` limbs. The C++ path, if it builds, performs the basis solve.
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

from ..model import Problem
from ._native_abi import native_abi_tag
from .registry import register_engine

_BASE = 1 << 32
_MASK = _BASE - 1
_CUTOFF = 8
_KARA_BASE = 4

_BASIC, _ATLOW, _ATUPP, _FREE = 0, 1, 2, 3
_STATUS = {_BASIC: "basic", _ATLOW: "at_lower", _ATUPP: "at_upper", _FREE: "free"}
_CODE = {name: code for code, name in _STATUS.items()}

_SRC = Path(__file__).resolve().parents[1] / "native" / "exact"
_LIB: dict = {}


class Singular(ValueError):
    """A basis matrix has no inverse over the rationals."""


def _limbs_norm(d: list) -> list:
    while d and d[-1] == 0:
        d.pop()
    return d


def _cmp_limbs(a: list, b: list) -> int:
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    for i in range(len(a) - 1, -1, -1):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    return 0


def _add_limbs(a: list, b: list) -> list:
    n = max(len(a), len(b))
    out = []
    carry = 0
    for i in range(n):
        t = carry
        if i < len(a):
            t += a[i]
        if i < len(b):
            t += b[i]
        out.append(t & _MASK)
        carry = t >> 32
    if carry:
        out.append(carry)
    return out


def _sub_limbs(a: list, b: list) -> list:
    """Difference of normalised limb lists. ``a`` must be at least ``b``."""
    out = []
    borrow = 0
    for i in range(len(a)):
        t = a[i] - borrow - (b[i] if i < len(b) else 0)
        if t < 0:
            t += _BASE
            borrow = 1
        else:
            borrow = 0
        out.append(t)
    if borrow:
        raise RuntimeError("limb subtraction went negative")
    return _limbs_norm(out)


def _school_limbs(a: list, b: list) -> list:
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        carry = 0
        for j, bj in enumerate(b):
            t = out[i + j] + ai * bj + carry
            out[i + j] = t & _MASK
            carry = t >> 32
        k = i + len(b)
        while carry:
            t = out[k] + carry
            out[k] = t & _MASK
            carry = t >> 32
            k += 1
            if k == len(out):
                out.append(0)
    return _limbs_norm(out)


def _kmul(x: list, y: list) -> list:
    """Karatsuba product. Schoolbook is used only under ``_KARA_BASE`` limbs."""
    if not x or not y:
        return []
    if len(x) < _KARA_BASE or len(y) < _KARA_BASE:
        return _school_limbs(x, y)
    m = (max(len(x), len(y)) + 1) // 2
    x0, x1 = x[:m], x[m:]
    y0, y1 = y[:m], y[m:]
    z0 = _kmul(x0, y0)
    z2 = _kmul(x1, y1)
    z1 = _kmul(_add_limbs(x0, x1), _add_limbs(y0, y1))
    z1 = _sub_limbs(z1, z0)
    z1 = _sub_limbs(z1, z2)
    out = z0 + [0] * max(0, len(z1) + m - len(z0))
    carry = 0
    for i, limb in enumerate(z1):
        t = out[i + m] + limb + carry
        out[i + m] = t & _MASK
        carry = t >> 32
    k = m + len(z1)
    while carry:
        if k == len(out):
            out.append(0)
        t = out[k] + carry
        out[k] = t & _MASK
        carry = t >> 32
        k += 1
    carry = 0
    need = 2 * m + len(z2)
    if len(out) < need:
        out.extend([0] * (need - len(out)))
    for i, limb in enumerate(z2):
        t = out[i + 2 * m] + limb + carry
        out[i + 2 * m] = t & _MASK
        carry = t >> 32
    k = 2 * m + len(z2)
    while carry:
        if k == len(out):
            out.append(0)
        t = out[k] + carry
        out[k] = t & _MASK
        carry = t >> 32
        k += 1
    return _limbs_norm(out)


class BigInt:
    """Arbitrary-precision integer stored as little-endian 32-bit limbs.

    The limbs are Python ints in ``0 .. 2**32-1``. Multiplication of large
    values uses Karatsuba; ``mul_schoolbook`` and ``mul_karatsuba`` force one
    algorithm so the two can be compared.
    """

    __slots__ = ("neg", "d")

    def __init__(self, value: int | BigInt = 0):
        if isinstance(value, BigInt):
            self.neg = value.neg
            self.d = value.d[:]
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"BigInt needs an int, got {type(value).__name__}")
        self.neg = value < 0
        value = -value if self.neg else value
        self.d = []
        while value:
            self.d.append(value & _MASK)
            value >>= 32

    @classmethod
    def _from_limbs(cls, limbs: list, neg: bool) -> BigInt:
        obj = cls.__new__(cls)
        obj.d = _limbs_norm(list(limbs))
        obj.neg = bool(neg) and bool(obj.d)
        return obj

    def is_zero(self) -> bool:
        return not self.d

    def is_even(self) -> bool:
        return (not self.d) or (self.d[0] & 1) == 0

    def bit_length(self) -> int:
        if not self.d:
            return 0
        return (len(self.d) - 1) * 32 + int(self.d[-1]).bit_length()

    def to_int(self) -> int:
        """The same integer as a Python int (the export form, not the multiplier)."""
        v = 0
        for limb in reversed(self.d):
            v = (v << 32) | limb
        return -v if self.neg else v

    def __repr__(self) -> str:
        return f"BigInt({self.to_int()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BigInt):
            return NotImplemented
        return self.neg == other.neg and self.d == other.d

    def cmp(self, other: BigInt) -> int:
        if self.neg != other.neg:
            return -1 if self.neg else 1
        c = _cmp_limbs(self.d, other.d)
        return -c if self.neg else c

    def __lt__(self, other: BigInt) -> bool:
        return self.cmp(other) < 0

    def abs(self) -> BigInt:
        return BigInt._from_limbs(self.d, False)

    def negated(self) -> BigInt:
        return BigInt._from_limbs(self.d, not self.neg)

    def __neg__(self) -> BigInt:
        return self.negated()

    def __add__(self, other: BigInt) -> BigInt:
        if self.neg == other.neg:
            return BigInt._from_limbs(_add_limbs(self.d, other.d), self.neg)
        c = _cmp_limbs(self.d, other.d)
        if c == 0:
            return BigInt(0)
        if c > 0:
            return BigInt._from_limbs(_sub_limbs(self.d, other.d), self.neg)
        return BigInt._from_limbs(_sub_limbs(other.d, self.d), other.neg)

    def __sub__(self, other: BigInt) -> BigInt:
        return self + other.negated()

    def __mul__(self, other: BigInt) -> BigInt:
        if self.is_zero() or other.is_zero():
            return BigInt(0)
        if len(self.d) >= _CUTOFF and len(other.d) >= _CUTOFF:
            limbs = _kmul(self.d, other.d)
        else:
            limbs = _school_limbs(self.d, other.d)
        return BigInt._from_limbs(limbs, self.neg ^ other.neg)

    def shl(self, bits: int) -> BigInt:
        if bits < 0:
            raise ValueError("negative shift")
        if bits == 0 or self.is_zero():
            return BigInt._from_limbs(self.d, self.neg)
        limb_shift, bit_shift = bits >> 5, bits & 31
        out = [0] * limb_shift
        carry = 0
        for limb in self.d:
            cur = (limb << bit_shift) + carry
            out.append(cur & _MASK)
            carry = cur >> 32
        if carry:
            out.append(carry)
        return BigInt._from_limbs(out, self.neg)

    def shr(self, bits: int) -> BigInt:
        """Logical shift of the magnitude. Sign is kept when the result is nonzero."""
        if bits < 0:
            raise ValueError("negative shift")
        if bits == 0 or self.is_zero():
            return BigInt._from_limbs(self.d, self.neg)
        limb_shift, bit_shift = bits >> 5, bits & 31
        if limb_shift >= len(self.d):
            return BigInt(0)
        src = self.d[limb_shift:]
        if bit_shift == 0:
            return BigInt._from_limbs(src, self.neg)
        out = []
        for i, limb in enumerate(src):
            cur = limb >> bit_shift
            if i + 1 < len(src):
                cur |= (src[i + 1] & ((1 << bit_shift) - 1)) << (32 - bit_shift)
            out.append(cur)
        return BigInt._from_limbs(out, self.neg)

    def _set_bit(self, index: int) -> BigInt:
        limb, bit = index >> 5, index & 31
        d = self.d[:]
        while len(d) <= limb:
            d.append(0)
        d[limb] |= 1 << bit
        return BigInt._from_limbs(d, False)

    def divmod(self, other: BigInt) -> tuple[BigInt, BigInt]:
        """Quotient and remainder. Remainder has the dividend's sign when nonzero."""
        if other.is_zero():
            raise ZeroDivisionError("division by zero")
        q, r = _divmod_pos(self.abs(), other.abs())
        if self.neg ^ other.neg:
            q = q.negated()
        if self.neg and not r.is_zero():
            r = r.negated()
        return q, r


def _divmod_pos(a: BigInt, b: BigInt) -> tuple[BigInt, BigInt]:
    if a.cmp(b) < 0:
        return BigInt(0), a
    shift = a.bit_length() - b.bit_length()
    r = a
    q = BigInt(0)
    bshift = b.shl(shift)
    for i in range(shift, -1, -1):
        if r.cmp(bshift) >= 0:
            r = r - bshift
            q = q._set_bit(i)
        bshift = bshift.shr(1)
    return q, r


def mul_schoolbook(a: BigInt, b: BigInt) -> BigInt:
    """Product by schoolbook multiplication, with no Karatsuba cutoff."""
    if a.is_zero() or b.is_zero():
        return BigInt(0)
    return BigInt._from_limbs(_school_limbs(a.d, b.d), a.neg ^ b.neg)


def mul_karatsuba(a: BigInt, b: BigInt) -> BigInt:
    """Product by Karatsuba recursion. The base of the recursion is schoolbook."""
    if a.is_zero() or b.is_zero():
        return BigInt(0)
    return BigInt._from_limbs(_kmul(a.d, b.d), a.neg ^ b.neg)


def binary_gcd(a: BigInt, b: BigInt) -> BigInt:
    """Greatest common divisor by Stein's binary algorithm. The result is nonnegative."""
    a, b = a.abs(), b.abs()
    if a.is_zero():
        return b
    if b.is_zero():
        return a
    shift = 0
    while a.is_even() and b.is_even():
        a, b = a.shr(1), b.shr(1)
        shift += 1
    while a.is_even():
        a = a.shr(1)
    while not b.is_zero():
        while b.is_even():
            b = b.shr(1)
        if a.cmp(b) > 0:
            a, b = b, a
        b = b - a
    return a.shl(shift) if shift else a


class Rat:
    """Rational in lowest terms, denominator positive.

    Constructed from ints or :class:`BigInt`. ``Rat(4, 6)`` is ``2/3``.
    """

    __slots__ = ("n", "d")

    def __init__(self, num: int | BigInt | Rat = 0, den: int | BigInt = 1):
        if isinstance(num, Rat):
            self.n = num.n
            self.d = num.d
            return
        n = num if isinstance(num, BigInt) else BigInt(int(num))
        d = den if isinstance(den, BigInt) else BigInt(int(den))
        if d.is_zero():
            raise ZeroDivisionError("zero denominator")
        if d.neg:
            n, d = n.negated(), d.abs()
        if n.is_zero():
            self.n, self.d = BigInt(0), BigInt(1)
            return
        g = binary_gcd(n, d)
        qn, rn = n.divmod(g)
        qd, rd = d.divmod(g)
        if not rn.is_zero() or not rd.is_zero():
            raise RuntimeError("binary gcd did not divide the rational")
        self.n, self.d = qn, qd

    def is_zero(self) -> bool:
        return self.n.is_zero()

    def is_negative(self) -> bool:
        return self.n.neg

    def is_positive(self) -> bool:
        return not self.n.neg and not self.n.is_zero()

    def __repr__(self) -> str:
        return f"Rat({self})"

    def __str__(self) -> str:
        n, d = self.n.to_int(), self.d.to_int()
        return str(n) if d == 1 else f"{n}/{d}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rat):
            return NotImplemented
        return self.n == other.n and self.d == other.d

    def __lt__(self, other: Rat) -> bool:
        return (self - other).is_negative()

    def __neg__(self) -> Rat:
        return Rat(self.n.negated(), self.d)

    def __add__(self, other: Rat) -> Rat:
        return Rat(self.n * other.d + other.n * self.d, self.d * other.d)

    def __sub__(self, other: Rat) -> Rat:
        return self + (-other)

    def __mul__(self, other: Rat) -> Rat:
        return Rat(self.n * other.n, self.d * other.d)

    def __truediv__(self, other: Rat) -> Rat:
        return Rat(self.n * other.d, self.d * other.n)

    def to_float(self) -> float:
        return self.n.to_int() / self.d.to_int()


def add_rationals(a: Rat, b: Rat) -> Rat:
    """Sum of two rationals, returned in lowest terms."""
    return a + b


def rat_from_float(value: float) -> Rat:
    """The exact rational value of a float (its binary fraction, reduced)."""
    if not math.isfinite(value):
        raise ValueError("non-finite float has no rational value")
    num, den = float(value).as_integer_ratio()
    return Rat(num, den)


def _bound(value: float) -> Rat | None:
    if not math.isfinite(float(value)):
        return None
    return rat_from_float(float(value))


def residual(rows: list, x: list, rhs: list) -> list:
    """Exact residual ``rhs - A x``."""
    out = []
    for i, row in enumerate(rows):
        acc = rhs[i]
        for j, aij in enumerate(row):
            acc = acc - aij * x[j]
        out.append(acc)
    return out


def reconstruct(value: float, max_den: int) -> Rat:
    """Continued-fraction convergent of ``value`` with denominator at most ``max_den``.

    The float is first taken as its exact dyadic rational. Convergents stop at
    ``max_den``, then the closer of the last convergent and the semi-convergent
    is returned. Callers accept it only after an exact residual check.
    """
    if not math.isfinite(value):
        raise ValueError("cannot reconstruct a non-finite float")
    num, den = float(value).as_integer_ratio()
    if den <= max_den:
        return Rat(num, den)
    sign = -1 if num < 0 else 1
    n, d = abs(num), den
    p0, q0, p1, q1 = 0, 1, 1, 0
    while d:
        a = n // d
        q2 = q0 + a * q1
        if q2 > max_den:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
        n, d = d, n - a * d
    if q1 == 0:
        return Rat(0, 1)
    k = (max_den - q0) // q1 if q1 else 0
    n1, d1 = p0 + k * p1, q0 + k * q1
    n2, d2 = p1, q1
    n0, d0 = abs(num), den

    def dist(nc: int, dc: int) -> int:
        return abs(nc * d0 - n0 * dc)

    if d2 and (d1 == 0 or dist(n2, d2) <= dist(n1, d1)):
        cn, cd = n2, d2
    else:
        cn, cd = n1, d1
    return Rat(sign * cn, cd)


def _float_matrix(rows: list, rhs: list):
    a = np.array([[coef.to_float() for coef in row] for row in rows], dtype=np.float64)
    b = np.array([coef.to_float() for coef in rhs], dtype=np.float64)
    return a, b


def _float_solve(a: np.ndarray, b: np.ndarray):
    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(x)):
        return None
    return x


def correct_residual(rows: list, rhs: list, x: list, max_den: int = 1 << 20) -> list | None:
    """One refinement correction: exact ``r = b - A x``, float solve ``A d = r``, add ``d``.

    ``d`` is recovered by continued fractions. Returns ``None`` when the float
    correction is not finite. This is one iteration of the Gleixner–Steffy–Wolter
    / Higham loop.
    """
    res = residual(rows, x, rhs)
    if all(term.is_zero() for term in res):
        return list(x)
    a, _ = _float_matrix(rows, rhs)
    rf = np.array([term.to_float() for term in res], dtype=np.float64)
    delta = _float_solve(a, rf)
    if delta is None:
        return None
    try:
        step = [reconstruct(float(delta[i]), max_den) for i in range(len(x))]
    except ValueError:
        return None
    return [x[i] + step[i] for i in range(len(x))]


def iterative_refinement(rows: list, rhs: list, rounds: int = 6) -> list | None:
    """Solve ``A x = b`` by iterative refinement, or ``None`` if it does not land exactly.

    Working precision is float64. Each round corrects the exact residual of the
    current dyadic solution, then tries a continued-fraction reconstruction.
    The result is returned only when ``A x - b`` is the zero vector of rationals.
    """
    m = len(rows)
    if m == 0:
        return []
    a, b = _float_matrix(rows, rhs)
    x = _float_solve(a, b)
    if x is None:
        return None
    for round_index in range(rounds):
        try:
            dyadic = [rat_from_float(float(x[i])) for i in range(m)]
        except ValueError:
            return None
        res = residual(rows, dyadic, rhs)
        if all(term.is_zero() for term in res):
            return dyadic
        delta = _float_solve(a, np.array([term.to_float() for term in res], dtype=np.float64))
        if delta is None:
            return None
        x = x + delta
        if not np.all(np.isfinite(x)):
            return None
        cap = 1 << min(12 + 2 * round_index, 30)
        try:
            guess = [reconstruct(float(x[i]), cap) for i in range(m)]
        except ValueError:
            return None
        if all(term.is_zero() for term in residual(rows, guess, rhs)):
            return guess
    return None


def exact_solve(rows: list, rhs: list) -> list:
    """Solve ``A x = b`` by rational Gaussian elimination with partial pivoting.

    This is the exact-LU fallback: the multipliers are rationals, so the
    computed ``x`` satisfies the equation with no rounding.
    """
    m = len(rows)
    if m == 0:
        return []
    if any(len(row) != m for row in rows) or len(rhs) != m:
        raise ValueError("exact solve needs a square system")
    mat = [row[:] + [rhs[i]] for i, row in enumerate(rows)]
    for col in range(m):
        pivot = col
        for i in range(col + 1, m):
            if abs_rat(mat[i][col]) > abs_rat(mat[pivot][col]):
                pivot = i
        if mat[pivot][col].is_zero():
            raise Singular("zero pivot")
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        piv = mat[col][col]
        for i in range(col + 1, m):
            if mat[i][col].is_zero():
                continue
            factor = mat[i][col] / piv
            mat[i][col] = Rat(0, 1)
            for j in range(col + 1, m + 1):
                mat[i][j] = mat[i][j] - factor * mat[col][j]
    x = [Rat(0, 1) for _ in range(m)]
    for i in range(m - 1, -1, -1):
        acc = mat[i][m]
        for j in range(i + 1, m):
            acc = acc - mat[i][j] * x[j]
        if mat[i][i].is_zero():
            raise Singular("zero diagonal")
        x[i] = acc / mat[i][i]
    return x


def abs_rat(value: Rat) -> Rat:
    """Absolute value of a rational."""
    return Rat(value.n.abs(), value.d)


def solve_linear(rows: list, rhs: list, arithmetic: str) -> tuple[list, str]:
    """Solve a square rational system with the selected arithmetic backend.

    ``arithmetic`` is ``python`` or ``native``. Native results are re-checked
    with the Python residual before they are trusted.
    """
    if len(rows) == 0:
        return [], "empty"
    if arithmetic == "native":
        got = _native_solve(rows, rhs)
        if got is None:
            raise RuntimeError(_LIB.get("reason", "native exact solve failed"))
        x, method = got
        if not all(term.is_zero() for term in residual(rows, x, rhs)):
            raise RuntimeError("native basis solve failed the exact residual check")
        return x, "native:" + method
    refined = iterative_refinement(rows, rhs)
    if refined is not None:
        return refined, "iterative_refinement"
    solved = exact_solve(rows, rhs)
    if not all(term.is_zero() for term in residual(rows, solved, rhs)):
        raise RuntimeError("exact LU residual is not zero")
    return solved, "exact_lu"


def _dot(a: list, b: list) -> Rat:
    acc = Rat(0, 1)
    for left, right in zip(a, b):
        acc = acc + left * right
    return acc


def _fmt_list(values: list) -> list:
    return [str(v) for v in values]


class _ExactSolver:
    """Bounded-variable revised simplex over :class:`Rat`.

    The equality form is ``A x - s = 0`` with ``s`` bounded by the row bounds,
    the same standard form the float simplex uses, but every pivot is exact.
    Bland's rule picks the least index. Basis systems go through
    :func:`solve_linear`.
    """

    def __init__(self, prob: Problem, arithmetic: str, candidate: bool, time_limit: float):
        self.arithmetic = arithmetic
        self.candidate = candidate
        self.time_limit = time_limit
        self.t0 = time.perf_counter()
        self.n, self.m = prob.n, prob.m
        self.N = self.n + self.m
        self.obj_sign = -1 if prob.obj_sign < 0 else 1
        self.c0 = rat_from_float(float(prob.c0))
        dense = np.asarray(prob.A.toarray(), dtype=np.float64) if self.m else np.zeros((0, self.n))
        self.cols = [[rat_from_float(float(dense[i, j])) for i in range(self.m)] for j in range(self.n)]
        for i in range(self.m):
            slack = [Rat(0, 1) for _ in range(self.m)]
            slack[i] = Rat(-1, 1)
            self.cols.append(slack)
        self.c = [rat_from_float(float(prob.c[j])) for j in range(self.n)] + [Rat(0, 1) for _ in range(self.m)]
        self.lx = [_bound(v) for v in prob.lx]
        self.ux = [_bound(v) for v in prob.ux]
        self.lc = [_bound(v) for v in prob.lc]
        self.uc = [_bound(v) for v in prob.uc]
        self.lo = self.lx + self.lc
        self.hi = self.ux + self.uc
        self.names = list(prob.names()[1]) + [f"s{i}" for i in range(self.m)]
        self.linear_solver = "none"
        self.basis_source = "exact_simplex"
        self.candidate_iterations = 0
        self.max_iter = 2000 + 30 * (self.n + self.m)
        self._reset()

    def _reset(self) -> None:
        self.vstat = []
        for j in range(self.N):
            if self.lo[j] is not None:
                self.vstat.append(_ATLOW)
            elif self.hi[j] is not None:
                self.vstat.append(_ATUPP)
            else:
                self.vstat.append(_FREE)
        self.basis = []
        for i in range(self.m):
            self.vstat[self.n + i] = _BASIC
            self.basis.append(self.n + i)
        self.z = [Rat(0, 1) for _ in range(self.N)]
        self.iterations = 0
        self._ray = None

    def _fixed(self, j: int) -> bool:
        lo, hi = self.lo[j], self.hi[j]
        return lo is not None and hi is not None and lo == hi

    def _nb(self, j: int) -> Rat:
        if self.vstat[j] == _ATLOW:
            return self.lo[j]
        if self.vstat[j] == _ATUPP:
            return self.hi[j]
        return Rat(0, 1)

    def _bmatrix(self) -> list:
        m = self.m
        return [[self.cols[self.basis[c]][r] for c in range(m)] for r in range(m)]

    def _solve(self, rows: list, rhs: list) -> list:
        x, method = solve_linear(rows, rhs, self.arithmetic)
        self.linear_solver = method
        return x

    def _recompute(self) -> list:
        z = [Rat(0, 1) for _ in range(self.N)]
        for j in range(self.N):
            if self.vstat[j] != _BASIC:
                z[j] = self._nb(j)
        rhs = [Rat(0, 1) for _ in range(self.m)]
        for j in range(self.N):
            if self.vstat[j] == _BASIC or z[j].is_zero():
                continue
            val = z[j]
            for i in range(self.m):
                rhs[i] = rhs[i] - self.cols[j][i] * val
        if self.m:
            basic = self._solve(self._bmatrix(), rhs)
            for i, j in enumerate(self.basis):
                z[j] = basic[i]
        return z

    def _duals(self, cost: list) -> list:
        if self.m == 0:
            return []
        m = self.m
        bt = [[self.cols[self.basis[r]][c] for c in range(m)] for r in range(m)]
        return self._solve(bt, [cost[self.basis[i]] for i in range(m)])

    def _weights(self) -> tuple[list, Rat]:
        w = [Rat(0, 1) for _ in range(self.N)]
        inf = Rat(0, 1)
        for j in self.basis:
            lo, hi = self.lo[j], self.hi[j]
            if lo is not None and self.z[j] < lo:
                w[j] = Rat(-1, 1)
                inf = inf + (lo - self.z[j])
            elif hi is not None and self.z[j] > hi:
                w[j] = Rat(1, 1)
                inf = inf + (self.z[j] - hi)
        return w, inf

    def _choose_enter(self, cost: list, pi: list) -> tuple[int, Rat] | None:
        for j in range(self.N):
            if self.vstat[j] == _BASIC or self._fixed(j):
                continue
            rc = cost[j] - _dot(pi, self.cols[j])
            if self.vstat[j] == _ATLOW and rc.is_negative():
                return j, Rat(1, 1)
            if self.vstat[j] == _ATUPP and rc.is_positive():
                return j, Rat(-1, 1)
            if self.vstat[j] == _FREE and rc.is_negative():
                return j, Rat(1, 1)
            if self.vstat[j] == _FREE and rc.is_positive():
                return j, Rat(-1, 1)
        return None

    def _ratio(self, enter: int, sigma: Rat, t: list):
        best = None

        def consider(alpha: Rat, index: int, kind: str, side: int) -> None:
            nonlocal best
            if alpha.is_negative():
                return
            if best is None or alpha < best[0] or (alpha == best[0] and index < best[1]):
                best = (alpha, index, kind, side)

        if sigma.is_positive() and self.hi[enter] is not None:
            consider((self.hi[enter] - self.z[enter]) / sigma, enter, "flip", _ATUPP)
        if sigma.is_negative() and self.lo[enter] is not None:
            consider((self.lo[enter] - self.z[enter]) / sigma, enter, "flip", _ATLOW)
        for i, b in enumerate(self.basis):
            delta = -t[i]
            if delta.is_zero():
                continue
            if delta.is_positive() and self.hi[b] is not None:
                consider((self.hi[b] - self.z[b]) / delta, b, "leave", _ATUPP)
            if delta.is_negative() and self.lo[b] is not None:
                consider((self.lo[b] - self.z[b]) / delta, b, "leave", _ATLOW)
        return best

    def _apply(self, enter: int, step) -> None:
        _alpha, index, kind, side = step
        if kind == "flip":
            self.vstat[enter] = side
            return
        pos = self.basis.index(index)
        self.vstat[index] = side
        self.vstat[enter] = _BASIC
        self.basis[pos] = enter

    def _loop(self, phase1: bool, deadline: float) -> str:
        while self.iterations < self.max_iter:
            if time.perf_counter() > deadline:
                return "time_limit"
            try:
                self.z = self._recompute()
            except (Singular, RuntimeError):
                return "numerical"
            if phase1:
                _w, inf = self._weights()
                if inf.is_zero():
                    return "feasible"
                cost = _w
            else:
                cost = self.c
            try:
                pi = self._duals(cost)
            except (Singular, RuntimeError):
                return "numerical"
            enter = self._choose_enter(cost, pi)
            if enter is None:
                return "infeasible" if phase1 else "optimal"
            j, sigma = enter
            try:
                t = self._solve(self._bmatrix(), [sigma * self.cols[j][i] for i in range(self.m)])
            except (Singular, RuntimeError):
                return "numerical"
            step = self._ratio(j, sigma, t)
            if step is None:
                if phase1:
                    return "numerical"
                self._ray = (j, sigma, t, list(self.z))
                return "unbounded"
            self._apply(j, step)
            self.iterations += 1
        return "iteration_limit"

    def _user_objective(self) -> Rat:
        acc = self.c0
        for j in range(self.n):
            acc = acc + self.c[j] * self.z[j]
        if self.obj_sign < 0:
            acc = -acc
        return acc

    def _activity_ok(self) -> bool:
        for j in range(self.N):
            lo, hi = self.lo[j], self.hi[j]
            if lo is not None and self.z[j] < lo:
                return False
            if hi is not None and self.z[j] > hi:
                return False
        for i in range(self.m):
            acc = Rat(0, 1)
            for j in range(self.N):
                acc = acc + self.cols[j][i] * self.z[j]
            if not acc.is_zero():
                return False
        return True

    def _basis_rhs(self) -> tuple[list, list]:
        rhs = [Rat(0, 1) for _ in range(self.m)]
        for j in range(self.N):
            if self.vstat[j] == _BASIC or self.z[j].is_zero():
                continue
            for i in range(self.m):
                rhs[i] = rhs[i] - self.cols[j][i] * self.z[j]
        x_basic = [self.z[j] for j in self.basis]
        return x_basic, rhs

    def _optimal_certificate(self, pi: list) -> dict | None:
        if not self._activity_ok():
            return None
        reduced = [self.c[j] - _dot(pi, self.cols[j]) for j in range(self.N)]
        for j, rc in enumerate(reduced):
            status = self.vstat[j]
            if status == _BASIC or status == _FREE:
                if not rc.is_zero():
                    return None
            elif status == _ATLOW and rc.is_negative():
                return None
            elif status == _ATUPP and rc.is_positive():
                return None
        x_basic, rhs = self._basis_rhs()
        if self.m:
            image = [Rat(0, 1) for _ in range(self.m)]
            for k, j in enumerate(self.basis):
                for i in range(self.m):
                    image[i] = image[i] + self.cols[j][i] * x_basic[k]
            if image != rhs:
                return None
        obj = self._user_objective()
        return self._pack("optimal", True, objective=obj, y=pi, reduced=reduced, x_basic=x_basic, basis_rhs=rhs)

    def _pack(self, status: str, proven: bool, objective: Rat | None = None, y: list | None = None,
              reduced: list | None = None, x_basic: list | None = None, basis_rhs: list | None = None,
              farkas: list | None = None, phi: Rat | None = None, ray: list | None = None,
              feasible_x: list | None = None, descent: Rat | None = None) -> dict:
        cert = {
            "status": status,
            "proven": proven,
            "arithmetic": self.arithmetic,
            "linear_solver": self.linear_solver,
            "basis_source": self.basis_source,
            "iterations": self.iterations,
            "candidate_iterations": self.candidate_iterations,
            "basic_columns": list(self.basis),
            "basic_names": [self.names[j] for j in self.basis],
            "vstat": [_STATUS[s] for s in self.vstat],
            "z": _fmt_list(self.z),
            "x": _fmt_list(self.z[:self.n]),
        }
        if objective is not None:
            cert["objective"] = str(objective)
        if y is not None:
            cert["y"] = _fmt_list(y)
        if reduced is not None:
            cert["reduced_costs"] = _fmt_list(reduced)
        if x_basic is not None:
            cert["x_basic"] = _fmt_list(x_basic)
        if basis_rhs is not None:
            cert["basis_rhs"] = _fmt_list(basis_rhs)
        if farkas is not None:
            cert["farkas"] = _fmt_list(farkas)
        if phi is not None:
            cert["phi"] = str(phi)
        if ray is not None:
            cert["ray"] = _fmt_list(ray)
        if feasible_x is not None:
            cert["feasible_x"] = _fmt_list(feasible_x)
        if descent is not None:
            cert["descent"] = str(descent)
        return cert

    def _farkas_of(self, y: list) -> tuple[Rat, Rat]:
        phi, viol = Rat(0, 1), Rat(0, 1)
        for i, yi in enumerate(y):
            if yi.is_positive():
                if self.lc[i] is None:
                    viol = viol + yi
                else:
                    phi = phi + yi * self.lc[i]
            elif yi.is_negative():
                if self.uc[i] is None:
                    viol = viol + (-yi)
                else:
                    phi = phi + yi * self.uc[i]
        for j in range(self.n):
            lam = Rat(0, 1)
            for i in range(self.m):
                lam = lam - self.cols[j][i] * y[i]
            if lam.is_positive():
                if self.lx[j] is None:
                    viol = viol + lam
                else:
                    phi = phi + lam * self.lx[j]
            elif lam.is_negative():
                if self.ux[j] is None:
                    viol = viol + (-lam)
                else:
                    phi = phi + lam * self.ux[j]
        return phi, viol

    def _farkas_certificate(self) -> dict | None:
        w, inf = self._weights()
        if not inf.is_positive():
            return None
        try:
            pi = self._duals(w)
        except (Singular, RuntimeError):
            return None
        for guess in (pi, [-v for v in pi]):
            phi, viol = self._farkas_of(guess)
            if viol.is_zero() and phi.is_positive():
                return self._pack("infeasible", True, farkas=guess, phi=phi, y=guess)
        return None

    def _ray_certificate(self) -> dict | None:
        if self._ray is None:
            return None
        j, sigma, t, z = self._ray
        d_ext = [Rat(0, 1) for _ in range(self.N)]
        d_ext[j] = sigma
        for i, b in enumerate(self.basis):
            d_ext[b] = -t[i]
        d = d_ext[:self.n]
        x = z[:self.n]
        if not any(not component.is_zero() for component in d):
            return None
        descent = Rat(0, 1)
        for k in range(self.n):
            descent = descent - self.c[k] * d[k]
        if not descent.is_positive():
            return None
        viol = Rat(0, 1)
        for i in range(self.m):
            activity = Rat(0, 1)
            for k in range(self.n):
                activity = activity + self.cols[k][i] * d[k]
            if self.lc[i] is not None and activity.is_negative():
                viol = viol + (-activity)
            if self.uc[i] is not None and activity.is_positive():
                viol = viol + activity
        for k in range(self.n):
            if self.lx[k] is not None and d[k].is_negative():
                viol = viol + (-d[k])
            if self.ux[k] is not None and d[k].is_positive():
                viol = viol + d[k]
        feas = Rat(0, 1)
        ax = [Rat(0, 1) for _ in range(self.m)]
        for k in range(self.n):
            for i in range(self.m):
                ax[i] = ax[i] + self.cols[k][i] * x[k]
            if self.lx[k] is not None and x[k] < self.lx[k]:
                feas = feas + (self.lx[k] - x[k])
            if self.ux[k] is not None and x[k] > self.ux[k]:
                feas = feas + (x[k] - self.ux[k])
        for i in range(self.m):
            if self.lc[i] is not None and ax[i] < self.lc[i]:
                feas = feas + (self.lc[i] - ax[i])
            if self.uc[i] is not None and ax[i] > self.uc[i]:
                feas = feas + (ax[i] - self.uc[i])
        if not viol.is_zero() or not feas.is_zero():
            return None
        self.z = z
        return self._pack("unbounded", True, ray=d, feasible_x=x, descent=descent)

    def _try_candidate(self, prob: Problem, deadline: float) -> dict | None:
        try:
            from .simplex import solve_simplex
            out = solve_simplex(prob, tol=1e-9, time_limit=max(0.1, deadline - time.perf_counter()))
        except Exception:
            return None
        self.candidate_iterations = int(out.get("iterations", 0))
        if out.get("status") != "optimal":
            return None
        cols, rows = out.get("col_statuses") or [], out.get("row_statuses") or []
        if len(cols) != self.n or len(rows) != self.m:
            return None
        if cols.count("basic") + rows.count("basic") != self.m:
            return None
        vstat = []
        basis = []
        for j, name in enumerate(list(cols) + list(rows)):
            code = _CODE.get(name)
            if code is None:
                return None
            vstat.append(code)
            if code == _BASIC:
                basis.append(j)
        if len(basis) != self.m:
            return None
        self.vstat = vstat
        self.basis = basis
        try:
            self.z = self._recompute()
            pi = self._duals(self.c)
        except (Singular, RuntimeError):
            return None
        if self._choose_enter(self.c, pi) is not None:
            return None
        self.basis_source = "candidate"
        return self._optimal_certificate(pi)

    def solve(self, prob: Problem) -> dict:
        deadline = self.t0 + self.time_limit
        if self.candidate and time.perf_counter() < deadline:
            certified = self._try_candidate(prob, deadline)
            if certified is not None:
                return certified
            self._reset()
            self.basis_source = "exact_simplex"
        phase = self._loop(True, deadline)
        if phase == "feasible":
            phase = self._loop(False, deadline)
        if phase == "optimal":
            try:
                self.z = self._recompute()
                pi = self._duals(self.c)
            except (Singular, RuntimeError):
                return self._pack("not_proven", False)
            cert = self._optimal_certificate(pi)
            return cert if cert is not None else self._pack("not_proven", False)
        if phase == "infeasible":
            cert = self._farkas_certificate()
            return cert if cert is not None else self._pack("not_proven", False)
        if phase == "unbounded":
            cert = self._ray_certificate()
            return cert if cert is not None else self._pack("not_proven", False)
        return self._pack("not_proven", False, objective=None)


def _select_arithmetic(arithmetic: str) -> str:
    if arithmetic == "python":
        return "python"
    forced_off = os.environ.get("QENIVO_NATIVE", "1") == "0"
    if arithmetic == "native":
        if library() is None:
            raise RuntimeError(status())
        return "native"
    if forced_off or library() is None:
        return "python"
    return "native"


def solve_exact_lp(prob: Problem, tol: float = 1e-8, time_limit: float = 60.0, backend: str = "auto",
                   verbose: bool = False, arithmetic: str = "auto", candidate: bool = True, **_ignored):
    """Solve a linear program in exact rational arithmetic and return a :class:`Solution`.

    ``arithmetic="python"`` uses Python ints even when a compiler exists.
    ``arithmetic="native"`` requires the C++ basis solver. ``auto`` uses C++ when
    it loads and Python otherwise. ``candidate`` asks the existing float simplex
    for a basis and keeps it only when the exact certificate succeeds.

    The rational proof is ``Solution.extra["rational"]``: fraction strings, not
    floats. ``Solution.objective`` remains the float image required by the
    certificate schema.
    """
    from ..api import _finish

    if not isinstance(prob, Problem):
        raise TypeError("exact-lp expects a qenivo Problem")
    if prob.is_qp or prob.is_mip:
        raise ValueError("exact-lp solves linear programs only")
    if backend == "python" and arithmetic == "auto":
        arithmetic = "python"
    arithmetic = _select_arithmetic(arithmetic)
    t0 = time.perf_counter()
    engine = _ExactSolver(prob, arithmetic, candidate, time_limit)
    proof = engine.solve(prob)
    elapsed = time.perf_counter() - t0
    meta = {
        "engine": "exact-lp",
        "backend": arithmetic,
        "iterations": proof.get("iterations", 0),
        "time": elapsed,
        "reason": "exact rational basis certificate",
        "arithmetic": arithmetic,
        "linear_solver": proof.get("linear_solver"),
        "basis_source": proof.get("basis_source"),
    }
    status_name = proof["status"] if proof.get("proven") else "not_proven"
    x = y = ray = None
    if status_name == "optimal":
        x = np.array([_parse_rat(v).to_float() for v in proof["x"]], dtype=np.float64)
        y = np.array([_parse_rat(v).to_float() for v in proof["y"]], dtype=np.float64)
    elif status_name == "infeasible":
        ray = np.array([_parse_rat(v).to_float() for v in proof["farkas"]], dtype=np.float64)
    elif status_name == "unbounded":
        ray = np.array([_parse_rat(v).to_float() for v in proof["ray"]], dtype=np.float64)
        x = np.array([_parse_rat(v).to_float() for v in proof["feasible_x"]], dtype=np.float64)
    sol = _finish(prob, status_name, x, y, tol, meta, ray=ray)
    if proof.get("proven"):
        sol.status = status_name
        sol.verdict = status_name
        if status_name == "optimal":
            sol.objective = _parse_rat(proof["objective"]).to_float()
            sol.x = x
            sol.y = None if y is None else prob.obj_sign * y
        elif status_name == "unbounded":
            sol.x = x
            sol.ray = ray
            sol.ray_check = {"valid": True, "exact": True, "descent": proof.get("descent")}
        elif status_name == "infeasible":
            sol.ray = ray
            sol.ray_check = {"valid": True, "exact": True, "phi": proof.get("phi")}
    sol.extra["rational"] = proof
    if status_name == "optimal":
        sol.extra["basis"] = {
            "col_statuses": proof["vstat"][:prob.n],
            "row_statuses": proof["vstat"][prob.n:],
        }
    if verbose:
        shown = proof.get("objective", proof.get("phi", proof.get("descent")))
        print(f"exact-lp {sol.verdict} rational {shown} via {proof.get('basis_source')} ({arithmetic})")
    return sol


def _parse_rat(text: str) -> Rat:
    if "/" not in text:
        return Rat(int(text), 1)
    num, den = text.split("/", 1)
    return Rat(int(num), int(den))


def _cache_dir() -> Path:
    base = os.environ.get("QENIVO_CACHE") or (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "qenivo"
        if os.name == "nt" else Path.home() / ".cache" / "qenivo")
    d = Path(base) / "native"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compiler():
    for cxx in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def _source_files() -> list[Path]:
    names = ["bigint.cpp", "rational.cpp", "lu.cpp", "refine.cpp", "api.cpp",
             "bigint.hpp", "rational.hpp", "lu.hpp", "refine.hpp"]
    return [_SRC / name for name in names]


def library():
    """The loaded exact-arithmetic library, or ``None`` when it cannot be built.

    ``QENIVO_NATIVE=0`` skips the compile and does not pin that choice into the
    cache, so a later call can still load the library.
    """
    if os.environ.get("QENIVO_NATIVE", "1") == "0":
        return None
    if "lib" in _LIB:
        return _LIB["lib"]
    _LIB["lib"], _LIB["reason"] = None, "disabled"
    files = _source_files()
    missing = [p.name for p in files if p.suffix == ".cpp" and not p.exists()]
    if missing:
        _LIB["reason"] = "missing sources: " + ", ".join(missing)
        return None
    blob = native_abi_tag()
    for path in files:
        if path.exists():
            blob += path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()[:16]
    ext = ".dll" if os.name == "nt" else (".dylib" if platform.system() == "Darwin" else ".so")
    out = _cache_dir() / f"qenivo_exact_{digest}{ext}"
    cpp = [str(p) for p in files if p.suffix == ".cpp" and p.exists()]
    if not out.exists():                 # a prebuilt library (installer) needs no compiler
        cxx = _compiler()
        if cxx is None:
            _LIB["reason"] = "no C++ compiler found (set CXX); the Python arithmetic is used"
            return None
        cmd = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-shared", "-o", str(out), *cpp]
        if os.name == "nt":
            cmd.append("-static")
        else:
            cmd.append("-fPIC")
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            _LIB["reason"] = "build failed: " + (built.stderr or built.stdout)[-400:]
            return None
    try:
        lib = ctypes.CDLL(str(out))
    except OSError as exc:
        _LIB["reason"] = f"native library could not be loaded ({exc}); the Python arithmetic is used"
        return None
    lib.nr_q_add.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.nr_q_add.restype = ctypes.c_void_p
    lib.nr_mul.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.nr_mul.restype = ctypes.c_void_p
    lib.nr_gcd.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.nr_gcd.restype = ctypes.c_void_p
    lib.nr_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.nr_solve.restype = ctypes.c_void_p
    lib.nr_free.argtypes = [ctypes.c_void_p]
    lib.nr_free.restype = None
    _LIB["lib"], _LIB["reason"], _LIB["path"] = lib, "ok", str(out)
    return lib


def status() -> str:
    """Why the native library is loaded, or why the Python arithmetic is in use."""
    if os.environ.get("QENIVO_NATIVE", "1") == "0":
        return "native arithmetic disabled (QENIVO_NATIVE=0)"
    library()
    return _LIB.get("reason", "unknown")


def _native_text(ptr) -> str | None:
    if not ptr:
        return None
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    finally:
        library().nr_free(ptr)


def native_q_add(a: str, b: str) -> str | None:
    """Add two rational strings with the C++ arithmetic. ``None`` if it is not built."""
    lib = library()
    if lib is None:
        return None
    return _native_text(lib.nr_q_add(a.encode(), b.encode()))


def native_mul(a: str, b: str, mode: int) -> str | None:
    """Multiply decimal strings in C++. ``mode`` 1 is schoolbook, 2 is Karatsuba."""
    lib = library()
    if lib is None:
        return None
    return _native_text(lib.nr_mul(a.encode(), b.encode(), int(mode)))


def native_gcd(a: str, b: str) -> str | None:
    """Binary GCD of two decimal strings in C++."""
    lib = library()
    if lib is None:
        return None
    return _native_text(lib.nr_gcd(a.encode(), b.encode()))


def native_solve_text(basis: str, rhs: str) -> str | None:
    """Solve a rational system in C++. The text is ``method`` newline ``solution``."""
    lib = library()
    if lib is None:
        return None
    return _native_text(lib.nr_solve(basis.encode(), rhs.encode()))


def _native_solve(rows: list, rhs: list) -> tuple[list, str] | None:
    basis = ";".join(",".join(str(coef) for coef in row) for row in rows)
    right = ",".join(str(coef) for coef in rhs)
    text = native_solve_text(basis, right)
    if not text or "\n" not in text:
        return None
    method, body = text.split("\n", 1)
    values = [_parse_rat(part) for part in body.split(",") if part]
    if len(values) != len(rhs):
        return None
    return values, method


# Imported for registration only. The solver above is the engine.
register_engine(
    "exact-lp",
    solve_exact_lp,
    classes=("LP",),
    description="exact rational LP: own bigint arithmetic, iterative refinement, rational certificate",
)
