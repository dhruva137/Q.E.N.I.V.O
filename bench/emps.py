"""Decoder for netlib's compressed LP format ("emps"), line-for-line port of
David M. Gay's emps.c (https://www.netlib.org/lp/data/emps.c).

Used by netlib.org/lp/data, the netlib kennington set, and many files on
plato.asu.edu/ftp/lptestset (those without a .mps suffix). The port keeps the
per-line checksum verification of emps.c, so a decoding error raises instead of
silently producing wrong numbers. Output is fixed-format MPS exactly as emps.c
prints it (numbers are the same 12-character strings).
"""
from __future__ import annotations

TRTAB = ("!\"#$%&'()*+,-./0123456789;<=>?@"
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~")
assert len(TRTAB) == 92
INV = [92] * 256
for _i, _ch in enumerate(TRTAB):
    INV[ord(_ch)] = _i
BOUND_TYPES = ["UP", "LO", "FX", "FR", "MI", "PL"]


class EmpsError(ValueError):
    pass


def is_emps(text: str) -> bool:
    """True when the text looks like emps: a NAME line followed by two numeric
    statistics lines (8 and 3 integers) instead of a ROWS section."""
    lines = text.splitlines()[:40]
    for i, line in enumerate(lines):
        if line.startswith("NAME"):
            rest = lines[i + 1:i + 3]
            if len(rest) < 2:
                return False
            a, b = rest[0].split(), rest[1].split()
            return (len(a) == 8 and len(b) == 3
                    and all(t.lstrip("-").isdigit() for t in a + b))
    return False


class _Reader:
    def __init__(self, text: str):
        self.lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.pos = 0
        self.ncs = 1
        self.chk = [" "]

    def _next_raw(self):
        if self.pos >= len(self.lines):
            raise EmpsError("premature end of file")
        s = self.lines[self.pos]
        self.pos += 1
        return s

    def _checkchar(self, s: str):
        x = 0
        for ch in s:
            c = INV[ord(ch) & 0xFF]
            x = (x >> 1) + c + 16384 if x & 1 else (x >> 1) + c
        self.chk.append(TRTAB[x % 92])
        self.ncs += 1

    def _checkline(self):
        while True:
            expected = "".join(self.chk)
            line = self._next_raw()
            if line == expected:
                break
            if line.startswith(":") and self.ncs <= 72:   # mystery line inside checksum
                self.chk.pop()
                self.ncs -= 1
                self._checkchar(line)
                continue
            raise EmpsError(f"checksum mismatch near line {self.pos}: expected {expected!r}, got {line!r}")
        self.ncs = 1
        self.chk = [" "]

    def rdline(self) -> str:
        while True:
            s = self._next_raw()
            self._checkchar(s)
            if self.ncs >= 72:
                self._checkline()
            if s.startswith(":"):                            # mystery line: skip
                continue
            return s

    def reset(self):
        self.ncs = 1
        self.chk = [" "]

    def final_check(self):
        if self.ncs > 1:
            self._checkline()


def _exindx(z: str, p: int):
    k = INV[ord(z[p])]
    p += 1
    if k >= 46:
        raise EmpsError(f"bad index in {z!r}")
    if k >= 23:
        return k - 23, p
    x = k
    while True:
        k = INV[ord(z[p])]
        p += 1
        x = x * 46 + k
        if k >= 46:
            return x - 46, p


def _exform(z: str, p: int, table: list, kmax: int):
    """Expand one number starting at z[p]; returns (12-char string, new p)."""
    k = INV[ord(z[p])]
    if k < 46:                                    # supersparse index into the number table
        idx, p = _exindx(z, p)
        if idx > kmax:
            raise EmpsError(f"index {idx} > kmax {kmax}")
        return table[idx], p
    p += 1
    s: list[str] = []
    d: list[str] = []
    k -= 46
    if k >= 23:
        s.append("-")
        k -= 23
        nelim = 11
    else:
        nelim = 12
    if k >= 11:                                   # integer floating point
        k -= 11
        d.append(".")
        if k >= 6:
            x = k - 6
        else:
            x = k
            while True:
                k = INV[ord(z[p])]
                p += 1
                x = x * 46 + k
                if k >= 46:
                    x -= 46
                    break
        if not x:
            d.append("0")
        else:
            while True:
                d.append(chr(48 + x % 10))
                x //= 10
                if not x:
                    break
        while d:
            s.append(d.pop())
    else:                                         # general floating point
        ex = INV[ord(z[p])] - 50
        p += 1
        x = INV[ord(z[p])]
        p += 1
        y = 0
        for _ in range(k):
            if x >= 100000000:
                y = x
                x = INV[ord(z[p])]
            else:
                x = x * 92 + INV[ord(z[p])]
            p += 1
        if y:
            while x > 1:
                d.append(chr(48 + x % 10))
                x //= 10
            while True:
                d.append(chr(48 + y % 10))
                if y < 10:
                    break
                y //= 10
        elif x:
            while True:
                d.append(chr(48 + x % 10))
                if x < 10:
                    break
                x //= 10
        else:
            d.append("0")
        nd = len(d) + ex
        eout = False
        if ex > 0:
            if nd < nelim or ex < 3:
                while d:
                    s.append(d.pop())
                s.extend("0" * ex)
                s.append(".")
            else:
                eout = True
        elif nd >= 0:
            for _ in range(nd):
                s.append(d.pop())
            s.append(".")
            while d:
                s.append(d.pop())
        elif ex > -nelim:
            s.append(".")
            s.extend("0" * (-nd))
            while d:
                s.append(d.pop())
        else:
            eout = True
        if eout:
            ex += len(d) - 1
            if ex == -10:
                ex = -9
            else:
                if 9 < ex <= len(d) + 8:
                    while True:
                        s.append(d.pop())
                        ex -= 1
                        if not ex > 9:
                            break
                s.append(d.pop())
            s.append(".")
            while d:
                s.append(d.pop())
            s.append("E")
            if ex < 0:
                s.append("-")
                ex = -ex
            e = []
            while ex:
                e.append(chr(48 + ex % 10))
                ex //= 10
            while e:
                s.append(e.pop())
    out = "".join(s)
    return out.rjust(12), p


def decode_emps(text: str) -> str:
    r = _Reader(text)
    out: list[str] = []
    buf = r.rdline()
    while not buf.startswith("NAME"):
        buf = r.rdline()
    out.append(buf)
    r.reset()
    a = r.rdline().split()
    b = r.rdline().split()
    if len(a) != 8 or len(b) != 3:
        raise EmpsError("bad statistics lines")
    nrow, ncol, _colmx, nz, _nrhs, rhsnz, _nran, ranz = map(int, a)
    _nbd, bdnz, ns = map(int, b)
    r.reset()

    table = [None] * (ns + 1)                     # 1-based number table
    z, p = "", 0
    for i in range(1, ns + 1):
        if p >= len(z):
            z, p = r.rdline(), 0
        table[i], p = _exform(z, p, table, i - 1)
    kmax = ns

    names = [""] * (nrow + ncol + 1)              # 1-based: rows then columns
    out.append("ROWS")
    for i in range(1, nrow + 1):
        line = r.rdline()
        typ, nm = line[0], line[1:]
        out.append(f" {typ}  {nm}")
        names[i] = nm[:8]
    cn = nrow

    def colout(head, count, what):
        nonlocal cn
        if not count:
            if what <= 2:
                out.append(head)
            return
        out.append(head)
        z, p = "", 0
        curcol = ""
        remaining = count
        while remaining:
            remaining -= 1
            if p >= len(z):
                z, p = r.rdline(), 0
            while True:
                n, p = _exindx(z, p)
                if n:
                    break
                curcol = z[p:p + 8]
                if what == 1:
                    cn += 1
                    names[cn] = z[p:p + 8]
                z, p = r.rdline(), 0
            if what >= 4:
                if n >= 7:
                    raise EmpsError(f"bad bound type index {n}")
                if p >= len(z):
                    z, p = r.rdline(), 0
                j, p = _exindx(z, p)
                colname = names[nrow + j]
                if n >= 4:
                    out.append(f" {BOUND_TYPES[n - 1]} {curcol:<8.8}  {colname:.8}")
                    continue
                if p >= len(z):
                    z, p = r.rdline(), 0
                val, p = _exform(z, p, table, kmax)
                out.append(f" {BOUND_TYPES[n - 1]} {curcol:<8.8}  {colname:<8.8}  {val:.15}")
            else:
                rowname = names[n]
                if p >= len(z):
                    z, p = r.rdline(), 0
                val, p = _exform(z, p, table, kmax)
                out.append(f"    {curcol:<8.8}  {rowname:<8.8}  {val:.15}")

    colout("COLUMNS", nz, 1)
    colout("RHS", rhsnz, 2)
    colout("RANGES", ranz, 3)
    colout("BOUNDS", bdnz, 4)
    r.final_check()
    out.append("ENDATA")
    return "\n".join(out) + "\n"
