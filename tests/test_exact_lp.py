"""Hand checks for the exact rational LP engine.

Expected values are fractions computed by hand. Nothing here treats a float
optimum as the proof, and nothing compares the answer only to the float simplex.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from qenivo.certify.verify import format_report, verify
from qenivo.engines import exact_lp as ex
from qenivo.engines.exact_lp import (
    BigInt,
    Rat,
    add_rationals,
    binary_gcd,
    correct_residual,
    exact_solve,
    iterative_refinement,
    mul_karatsuba,
    mul_schoolbook,
    rat_from_float,
    reconstruct,
    solve_exact_lp,
)
from qenivo.io.mps import write_mps
from qenivo.model import ModelBuilder


def _arithmetic_params():
    return ["python", "native"]


@pytest.fixture(params=_arithmetic_params())
def arithmetic(request):
    if request.param == "native" and ex.library() is None:
        pytest.skip(ex.status())
    return request.param


def _textbook():
    builder = ModelBuilder("tiny")
    builder.var("x1", 0, np.inf, obj=-1)
    builder.var("x2", 0, np.inf, obj=-1)
    builder.row("r1", {"x1": 1, "x2": 2}, hi=4)
    builder.row("r2", {"x1": 2, "x2": 1}, hi=4)
    return builder.build()


def _infeasible():
    builder = ModelBuilder("infeas")
    builder.var("x", 0, np.inf, obj=0)
    builder.row("r", {"x": 1}, hi=-1)
    return builder.build()


def _unbounded():
    builder = ModelBuilder("unbounded")
    builder.var("x1", 0, np.inf, obj=-1)
    builder.var("x2", 0, np.inf, obj=0)
    builder.row("r", {"x1": -1, "x2": 1}, hi=3)
    return builder.build()


def _F(text: str) -> Fraction:
    return Fraction(text)


def test_two_thirds_sum_to_two():
    third = Rat(2, 3)
    total = add_rationals(add_rationals(third, third), third)
    assert total == Rat(2, 1)
    assert str(total) == "2"
    # Reduction is part of the same arithmetic: 4/6 is 2/3 before the additions.
    assert Rat(4, 6) == Rat(2, 3)
    assert str(add_rationals(add_rationals(Rat(4, 6), Rat(4, 6)), Rat(4, 6))) == "2"


def test_two_thirds_native():
    if ex.library() is None:
        pytest.skip(ex.status())
    first = ex.native_q_add("2/3", "2/3")
    assert first == "4/3"
    assert ex.native_q_add(first, "2/3") == "2"
    assert ex.native_gcd("84", "30") == "6"


def test_karatsuba_matches_schoolbook():
    left = BigInt(10**80 - 1)
    right = BigInt(10**80 + 1)
    school = mul_schoolbook(left, right)
    kara = mul_karatsuba(left, right)
    assert school == kara
    assert school.to_int() == 10**160 - 1
    negative = mul_karatsuba(-left, right)
    assert negative == mul_schoolbook(-left, right)
    assert negative.to_int() == -(10**160 - 1)
    wide_a = BigInt(int("9" * 90))
    wide_b = BigInt(int("7" * 80))
    assert mul_schoolbook(wide_a, wide_b) == mul_karatsuba(wide_a, wide_b)


def test_karatsuba_native_matches_schoolbook():
    if ex.library() is None:
        pytest.skip(ex.status())
    left = "9" * 90
    right = "7" * 80
    assert ex.native_mul(left, right, 1) == ex.native_mul(left, right, 2)
    assert ex.native_mul(str(10**80 - 1), str(10**80 + 1), 2) == str(10**160 - 1)


def test_binary_gcd_known_values():
    assert binary_gcd(BigInt(84), BigInt(30)).to_int() == 6
    # 2**40 * 15 and 2**100 * 21 share 2**40 * 3.
    left = BigInt((1 << 40) * 15)
    right = BigInt((1 << 100) * 21)
    assert binary_gcd(left, right).to_int() == (1 << 40) * 3


def test_basis_solvers_return_four_thirds():
    rows = [[Rat(1), Rat(2)], [Rat(2), Rat(1)]]
    rhs = [Rat(4), Rat(4)]
    assert exact_solve(rows, rhs) == [Rat(4, 3), Rat(4, 3)]
    assert iterative_refinement(rows, rhs) == [Rat(4, 3), Rat(4, 3)]
    assert correct_residual(rows, rhs, [Rat(1), Rat(1)]) == [Rat(4, 3), Rat(4, 3)]
    assert reconstruct(float(Fraction(4, 3)), 1 << 12) == Rat(4, 3)
    assert rat_from_float(0.5) == Rat(1, 2)


def test_native_basis_solve_returns_four_thirds():
    if ex.library() is None:
        pytest.skip(ex.status())
    text = ex.native_solve_text("1,2;2,1", "4,4")
    assert text is not None
    method, body = text.split("\n", 1)
    assert method == "iterative_refinement"
    assert body == "4/3,4/3"


def _assert_textbook(sol):
    proof = sol.extra["rational"]
    assert sol.verdict == "optimal"
    assert proof["proven"] is True
    assert proof["x"] == ["4/3", "4/3"]
    assert proof["y"] == ["-1/3", "-1/3"]
    assert proof["objective"] == "-8/3"
    assert proof["z"] == ["4/3", "4/3", "4", "4"]
    assert proof["vstat"] == ["basic", "basic", "at_upper", "at_upper"]
    assert proof["reduced_costs"] == ["0", "0", "-1/3", "-1/3"]
    assert "refinement" in proof["linear_solver"]
    x = [_F(v) for v in proof["x"]]
    y = [_F(v) for v in proof["y"]]
    z = [_F(v) for v in proof["z"]]
    # minimising -x1-x2 attains -8/3; the positive optimum x1+x2 is 8/3.
    assert x[0] + x[1] == Fraction(8, 3)
    assert _F(proof["objective"]) == Fraction(-8, 3)
    assert x == [Fraction(4, 3), Fraction(4, 3)]
    cols = [
        [Fraction(1), Fraction(2)],
        [Fraction(2), Fraction(1)],
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(-1)],
    ]
    cost = [Fraction(-1), Fraction(-1), Fraction(0), Fraction(0)]
    for i in range(2):
        assert sum(cols[j][i] * z[j] for j in range(4)) == 0
    for j in range(4):
        rc = cost[j] - sum(y[i] * cols[j][i] for i in range(2))
        assert rc == _F(proof["reduced_costs"][j])
        if proof["vstat"][j] == "basic":
            assert rc == 0
        elif proof["vstat"][j] == "at_upper":
            assert rc <= 0
        else:
            assert rc >= 0
    basic = proof["basic_columns"]
    assert sorted(basic) == [0, 1]
    x_basic = [z[j] for j in basic]
    rhs = [Fraction(0), Fraction(0)]
    for j in range(4):
        if j in basic:
            continue
        for i in range(2):
            rhs[i] -= cols[j][i] * z[j]
    image = [sum(cols[basic[k]][i] * x_basic[k] for k in range(2)) for i in range(2)]
    assert image == rhs
    assert image == [_F(v) for v in proof["basis_rhs"]]
    assert image == [Fraction(4), Fraction(4)]


@pytest.mark.parametrize("candidate", [False, True])
def test_textbook_optimum_is_eight_thirds(arithmetic, candidate):
    sol = solve_exact_lp(_textbook(), arithmetic=arithmetic, candidate=candidate)
    _assert_textbook(sol)
    if candidate:
        assert sol.extra["rational"]["basis_source"] == "candidate"
    else:
        assert sol.extra["rational"]["basis_source"] == "exact_simplex"


def test_textbook_through_registered_engine():
    import qenivo

    sol = qenivo.solve(_textbook(), engine="exact-lp", arithmetic="python", candidate=False)
    _assert_textbook(sol)
    assert sol.engine["engine"] == "exact-lp"


def test_infeasible_farkas_ray(arithmetic):
    sol = solve_exact_lp(_infeasible(), arithmetic=arithmetic, candidate=False)
    proof = sol.extra["rational"]
    assert sol.verdict == "infeasible"
    assert proof["farkas"] == ["-1"]
    assert proof["phi"] == "1"
    y = _F(proof["farkas"][0])
    # Row x <= -1, x >= 0. y = -1, lam = -A^T y = 1.
    lam = -y
    assert y < 0 and lam > 0
    phi = (-y) * Fraction(1)  # -y_neg * uc with uc = -1, plus lam * lx = 0
    assert phi == Fraction(1)
    assert phi == _F(proof["phi"])
    assert lam * Fraction(0) == 0


def test_unbounded_ray(arithmetic):
    sol = solve_exact_lp(_unbounded(), arithmetic=arithmetic, candidate=False)
    proof = sol.extra["rational"]
    assert sol.verdict == "unbounded"
    d = [_F(v) for v in proof["ray"]]
    x = [_F(v) for v in proof["feasible_x"]]
    assert d == [Fraction(1), Fraction(0)]
    assert x == [Fraction(0), Fraction(0)]
    cost = [Fraction(-1), Fraction(0)]
    descent = -sum(cost[j] * d[j] for j in range(2))
    assert descent == Fraction(1)
    assert descent == _F(proof["descent"])
    activity = -d[0] + d[1]
    assert activity <= 0
    assert -x[0] + x[1] <= 3
    assert d[0] >= 0 and d[1] >= 0


def test_public_verifier_exact_mode(tmp_path):
    """The public verifier has exact=True. It rationalises the float certificate.

    The rational strings are checked in the tests above; this call is the
    verifier's own exact mode on the float image of the same point.
    """
    prob = _textbook()
    sol = solve_exact_lp(prob, arithmetic="python", candidate=False)
    path = tmp_path / "tiny.mps"
    write_mps(prob, path)
    report = verify(path, sol.certificate(), exact=True)
    assert report["exact"] is True
    assert report["passed"], format_report(report)

    infeas = solve_exact_lp(_infeasible(), arithmetic="python", candidate=False)
    # An explicit lower bound of 0 must precede the negative upper bound, or the
    # verifier's MPS reader would drop the lower bound.
    mps = tmp_path / "infeas.mps"
    mps.write_text(
        "\n".join([
            "NAME infeas",
            "ROWS",
            " N OBJ",
            " L r",
            "COLUMNS",
            " x OBJ 0",
            " x r 1",
            "RHS",
            " RHS r -1",
            "BOUNDS",
            " LO BND x 0",
            " UP BND x -1",
            "ENDATA",
            "",
        ]),
        encoding="utf-8",
    )
    infeas_report = verify(mps, infeas.certificate(), exact=True)
    assert infeas_report["passed"], format_report(infeas_report)

    unbounded = solve_exact_lp(_unbounded(), arithmetic="python", candidate=False)
    ub_path = tmp_path / "unbounded.mps"
    write_mps(_unbounded(), ub_path)
    ub_report = verify(ub_path, unbounded.certificate(), exact=True)
    assert ub_report["passed"], format_report(ub_report)


def test_fallback_when_compiler_missing(monkeypatch, tmp_path):
    """No compiler and no library built yet: the Python arithmetic answers.

    (A library already in the cache loads without a compiler; that is how the installer works, see
    tests/test_native_abi.py. So the cache here is an empty directory.)
    """
    monkeypatch.setattr(ex, "_compiler", lambda: None)
    monkeypatch.setenv("QENIVO_CACHE", str(tmp_path))
    monkeypatch.delenv("QENIVO_NATIVE", raising=False)
    ex._LIB.clear()
    try:
        assert ex.library() is None
        sol = solve_exact_lp(_textbook(), arithmetic="auto", candidate=False)
        assert sol.extra["rational"]["arithmetic"] == "python"
        assert sol.extra["rational"]["x"] == ["4/3", "4/3"]
        assert _F(sol.extra["rational"]["x"][0]) + _F(sol.extra["rational"]["x"][1]) == Fraction(8, 3)
    finally:
        ex._LIB.clear()


def test_fallback_when_native_disabled(monkeypatch):
    monkeypatch.setenv("QENIVO_NATIVE", "0")
    sol = solve_exact_lp(_textbook(), arithmetic="auto", candidate=False)
    assert sol.extra["rational"]["arithmetic"] == "python"
    assert sol.extra["rational"]["objective"] == "-8/3"


def test_quadratic_is_rejected():
    prob = _textbook()
    prob.Q = prob.A  # any nonzero quadratic marks the model as a QP
    with pytest.raises(ValueError, match="linear programs"):
        solve_exact_lp(prob, arithmetic="python", candidate=False)


def test_verbose_smoke(capsys):
    solve_exact_lp(_textbook(), arithmetic="python", candidate=False, verbose=True)
    assert "exact-lp" in capsys.readouterr().out
