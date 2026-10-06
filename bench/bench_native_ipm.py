"""Measured E2 smoke. Writes bench/results/native_ipm.jsonl from this run only.

The factor residual and the hand LP must succeed. Netlib rows are the MPS files
already under tests/data. Nothing is downloaded, and a miss stays a miss.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qenivo import read  # noqa: E402
from qenivo.certify.kkt import kkt_residuals  # noqa: E402
from qenivo.engines.native_ipm import factor_solve, solve_ipm_native  # noqa: E402
from qenivo.engines.native_ipm import status as native_status  # noqa: E402
from qenivo.model import ModelBuilder  # noqa: E402

# Published Netlib values, the same constants tests/test_native.py checks.
PUBLISHED = {
    "afiro": -4.6475314286e02,
    "blend": -3.0812149846e01,
    "sc50a": -6.4575077059e01,
    "sc105": -5.2202061212e01,
    "adlittle": 2.2549496316e05,
    "kb2": -1.7499001299e03,
}


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _base() -> dict:
    return {
        "suite": "bench_native_ipm",
        "commit": _commit(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native": native_status(),
    }


def _csc(A):
    C = sp.csc_matrix(A, dtype=float)
    C.sum_duplicates()
    return C


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    mask = rng.random((n, n)) < 0.15
    B = np.tril(B * mask, -1)
    return _csc(B @ B.T + np.eye(n))


def _kkt(q, m, seed):
    rng = np.random.default_rng(seed)
    C = (rng.random((m, q)) < 0.4) * rng.normal(scale=0.2, size=(m, q))
    n = q + m
    A = np.zeros((n, n))
    A[:q, :q] = 2.0 * np.eye(q)
    A[q:, :q] = C
    A[:q, q:] = C.T
    A[q:, q:] = -0.5 * np.eye(m)
    return _csc(A)


def _rel(A, x, b):
    return float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))


def _hand():
    built = ModelBuilder("hand")
    built.var("x", 0.0, np.inf, obj=2.0)
    built.var("y", 0.0, np.inf, obj=1.0)
    built.row("eq", {"x": 1.0, "y": 1.0}, 1.0, 1.0)
    return built.build()


def main() -> int:
    rows = []
    gate_failed = False
    rng = np.random.default_rng(6)

    for kind, matrix, spd, method in (
        ("factor_spd", _spd(22, 4), True, "amd"),
        ("factor_spd", _spd(22, 4), True, "nd"),
        ("factor_kkt", _kkt(8, 4, 5), False, "amd"),
    ):
        for impl in ("auto", "python"):
            b = rng.normal(size=matrix.shape[0])
            t0 = time.perf_counter()
            try:
                x = factor_solve(
                    matrix.shape[0],
                    matrix.indptr,
                    matrix.indices,
                    matrix.data,
                    b,
                    method=method,
                    spd=spd,
                    impl=impl,
                )
                residual = _rel(matrix, x, b)
                ok = residual <= 1e-12
                err = ""
            except Exception as exc:  # noqa: BLE001
                residual, ok, err = None, False, str(exc).splitlines()[0][:240]
            if not ok:
                gate_failed = True
            row = _base()
            row.update(
                kind=kind,
                method=method,
                impl=impl,
                n=int(matrix.shape[0]),
                residual=residual,
                seconds=round(time.perf_counter() - t0, 6),
                ok=ok,
                error=err,
            )
            rows.append(row)

    prob = _hand()
    for impl in ("auto", "python"):
        t0 = time.perf_counter()
        try:
            out = solve_ipm_native(prob, tol=1e-8, impl=impl, max_iter=80)
            k = kkt_residuals(prob, out["x"], out["y"])
            ok = (
                out["status"] == "optimal"
                and abs(out["x"][0]) <= 1e-5
                and abs(out["x"][1] - 1.0) <= 1e-5
                and k["max_rel"] <= 1e-7
            )
            err = ""
            status, obj, iters, max_rel = (
                out["status"],
                float(k["objective"]),
                int(out["iterations"]),
                float(k["max_rel"]),
            )
        except Exception as exc:  # noqa: BLE001
            ok, err = False, str(exc).splitlines()[0][:240]
            status, obj, iters, max_rel = "error", None, None, None
        if not ok:
            gate_failed = True
        row = _base()
        row.update(
            kind="hand_lp",
            impl=impl,
            status=status,
            objective=obj,
            iterations=iters,
            max_rel=max_rel,
            seconds=round(time.perf_counter() - t0, 6),
            ok=ok,
            error=err,
        )
        rows.append(row)

    data = ROOT / "tests" / "data"
    for name, published in PUBLISHED.items():
        path = data / f"{name}.mps"
        row = _base()
        row.update(kind="netlib", instance=name, published=published)
        if not path.exists():
            row.update(ok=False, status="missing", error=str(path))
            rows.append(row)
            continue
        t0 = time.perf_counter()
        try:
            model = read(path)
            out = solve_ipm_native(model, tol=1e-8, time_limit=60.0, max_iter=200)
            k = kkt_residuals(model, out["x"], out["y"]) if out["x"] is not None else {}
            obj = k.get("objective")
            rel = None
            if obj is not None:
                rel = abs(obj - published) / max(1.0, abs(published))
            ok = out["status"] == "optimal" and rel is not None and rel <= 1e-6
            row.update(
                status=out["status"],
                rows=int(model.m),
                cols=int(model.n),
                nnz=int(model.nnz),
                objective=None if obj is None else float(obj),
                rel_obj=rel,
                iterations=int(out["iterations"]),
                max_rel=None if "max_rel" not in k else float(k["max_rel"]),
                factor=out.get("factor"),
                seconds=round(time.perf_counter() - t0, 6),
                ok=ok,
                error="",
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                status="error",
                seconds=round(time.perf_counter() - t0, 6),
                ok=False,
                error=str(exc).splitlines()[0][:240],
            )
        rows.append(row)

    out = ROOT / "bench" / "results" / "native_ipm.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    passed = sum(1 for r in rows if r.get("ok"))
    print(f"wrote {out} ({len(rows)} rows, {passed} ok, gate_failed={gate_failed})")
    return 1 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
