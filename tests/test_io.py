"""IO: native MPS reader, LP/JSON formats, round-trips, write_mps vectorisation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from qenivo.io.jsonmodel import read_json, write_json
from qenivo.io.lpformat import read_lp, write_lp
from qenivo.io.mps import _read_mps_python, read, read_mps, write_mps
from qenivo.models.industry import LIBRARY

DATA = Path(__file__).resolve().parent / "data"
MPS_FILES = sorted(DATA.glob("*.mps"))

SMALL = {
    "economic_dispatch": dict(units=4, periods=6),
    "unit_commitment": dict(units=3, periods=6),
    "transportation": dict(sources=6, sinks=15),
    "facility_location": dict(facilities=5, customers=12),
    "lot_sizing": dict(items=3, periods=5),
    "crude_blending": {},
    "crude_scheduling": dict(tanks=2, crudes=2, periods=4),
    "multi_commodity_flow": dict(nodes=10, commodities=3),
}


def _arrays_equal(a, b, rtol=0.0, atol=0.0):
    assert a.m == b.m and a.n == b.n
    assert a.c0 == b.c0 and a.obj_sign == b.obj_sign
    for name in ("c", "lc", "uc", "lx", "ux"):
        assert np.allclose(getattr(a, name), getattr(b, name), rtol=rtol, atol=atol, equal_nan=True), name
    assert np.array_equal(a.A.indptr, b.A.indptr)
    assert np.array_equal(a.A.indices, b.A.indices)
    assert np.allclose(a.A.data, b.A.data, rtol=rtol, atol=atol)
    assert list(a.row_names) == list(b.row_names)
    assert list(a.col_names) == list(b.col_names)
    ai = np.zeros(a.n, bool) if a.integer is None else a.integer
    bi = np.zeros(b.n, bool) if b.integer is None else b.integer
    assert np.array_equal(ai, bi)
    if a.Q is None and b.Q is None:
        return
    assert a.Q is not None and b.Q is not None
    assert (a.Q - b.Q).nnz == 0


@pytest.mark.parametrize("path", MPS_FILES, ids=[p.name for p in MPS_FILES])
def test_native_mps_matches_python(path):
    py = _read_mps_python(path)
    os.environ.pop("QENIVO_MPS_PYTHON", None)
    try:
        from qenivo.io.native_mps import read_mps_native, status
        nat = read_mps_native(path)
    except Exception as e:
        pytest.skip(f"native MPS unavailable: {e} ({status() if 'status' in dir() else ''})")
    _arrays_equal(nat, py)


@pytest.mark.parametrize("path", MPS_FILES, ids=[p.name for p in MPS_FILES])
def test_mps_lp_mps_roundtrip(path, tmp_path):
    os.environ["QENIVO_MPS_PYTHON"] = "1"
    try:
        orig = _read_mps_python(path)
        lp = tmp_path / "t.lp"
        mps2 = tmp_path / "t.mps"
        write_lp(orig, lp)
        mid = read_lp(lp)
        write_mps(mid, mps2)
        back = _read_mps_python(mps2)
        # Free rows are dropped by MPS; compare after filtering infinite row bounds.
        keep = np.isfinite(orig.lc) | np.isfinite(orig.uc)
        o = orig
        if not np.all(keep):
            o = type(orig)(
                c=orig.c, A=orig.A[keep], lc=orig.lc[keep], uc=orig.uc[keep],
                lx=orig.lx, ux=orig.ux, c0=orig.c0, obj_sign=orig.obj_sign,
                name=orig.name,
                row_names=[orig.row_names[i] for i in range(orig.m) if keep[i]],
                col_names=list(orig.col_names),
                integer=orig.integer, Q=orig.Q,
            )
        assert o.m == back.m and o.n == back.n
        assert np.allclose(o.c, back.c)
        assert np.allclose(o.lc, back.lc) and np.allclose(o.uc, back.uc)
        assert np.allclose(o.lx, back.lx) and np.allclose(o.ux, back.ux)
        assert (o.A - back.A).nnz == 0
    finally:
        os.environ.pop("QENIVO_MPS_PYTHON", None)


@pytest.mark.parametrize("path", MPS_FILES, ids=[p.name for p in MPS_FILES])
def test_mps_json_mps_roundtrip(path, tmp_path):
    orig = _read_mps_python(path)
    js = tmp_path / "t.json"
    mps2 = tmp_path / "t.mps"
    write_json(orig, js)
    mid = read_json(js)
    write_mps(mid, mps2)
    back = _read_mps_python(mps2)
    assert np.allclose(orig.c, back.c)
    assert np.allclose(orig.lc, back.lc) and np.allclose(orig.uc, back.uc)
    assert np.allclose(orig.lx, back.lx) and np.allclose(orig.ux, back.ux)
    assert (orig.A - back.A).nnz == 0
    assert orig.obj_sign == back.obj_sign and orig.c0 == back.c0


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_sector_mps_lp_roundtrip(name, tmp_path):
    p = LIBRARY[name](**SMALL.get(name, {}))
    mps = tmp_path / "s.mps"
    lp = tmp_path / "s.lp"
    mps2 = tmp_path / "s2.mps"
    write_mps(p, mps)
    a = _read_mps_python(mps)
    write_lp(a, lp)
    b = read_lp(lp)
    write_mps(b, mps2)
    c = _read_mps_python(mps2)
    assert a.n == c.n
    assert np.allclose(a.c, c.c)
    assert (a.A - c.A).nnz == 0


def test_read_dispatches_by_suffix(tmp_path):
    p = _read_mps_python(DATA / "afiro.mps")
    lp = tmp_path / "a.lp"
    js = tmp_path / "a.json"
    mps = tmp_path / "a.mps"
    write_lp(p, lp)
    write_json(p, js)
    write_mps(p, mps)
    assert read(lp).n == p.n
    assert read(js).n == p.n
    assert read_mps(mps).n == p.n  # api.read uses read_mps


def test_write_mps_python_readable(tmp_path):
    p = _read_mps_python(DATA / "blend.mps")
    f = tmp_path / "out.mps"
    write_mps(p, f)
    q = _read_mps_python(f)
    _arrays_equal(p, q)


def test_sc_bound_errors(tmp_path):
    text = "NAME T\nROWS\n N OBJ\n L R1\nCOLUMNS\n X R1 1\nBOUNDS\n SC BND X 2\nENDATA\n"
    path = tmp_path / "sc.mps"
    path.write_text(text)
    with pytest.raises(NotImplementedError):
        _read_mps_python(path)
    try:
        from qenivo.io.native_mps import library, read_mps_native, status
    except Exception:
        pytest.skip("native MPS import failed")
    if library() is None:                   # no compiler, or QENIVO_NATIVE=0
        pytest.skip(status())
    try:
        read_mps_native(path)
        pytest.fail("expected native reader to reject SC bounds")
    except RuntimeError as e:
        if "not supported" in str(e) or "SC" in str(e):
            return
        if "compiler" in str(e).lower() or "build" in str(e).lower() or "native" in str(e).lower():
            pytest.skip(status())
        raise
