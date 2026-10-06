"""`qenivo-io`: a scriptable optimizer shell with the command layout planners already know.

    qenivo-io                               interactive (prompt on a terminal, else reads stdin)
    qenivo-io model.mps                     read a model, then take commands
    qenivo-io -c "read m.mps" "optimize" "display solution objective"
    qenivo-io -f commands.txt [-i]          run a command file (-i: stay interactive after it)

Commands (any unique prefix works, e.g. `disp sol var -`):
    read FILE [mps|lp]            optimize | primopt | tranopt | baropt | mipopt
    display solution objective|variables|dual|slacks|reduced|quality [ITEMS]
    display problem stats         display settings [changed|all]
    set timelimit S | threads N | mip tolerances mipgap G | tolerance T | certdir DIR | defaults
    write FILE [sol|mps|lp|cert]  help [COMMAND]   quit

The command names, the -c / -f / -i invocation and the parameter paths follow the public user
manual of the IBM ILOG interactive optimizer (sources and the legal note in docs/COMPATIBILITY.md).
This is an independent implementation: no text or code of that product is used, and every
answer goes through QENIVO's certificate path. A certificate JSON is written after every solve.
Options without a QENIVO equivalent are refused with a message rather than silently accepted.
"""
from __future__ import annotations

import argparse
import fnmatch
import shlex
import sys
import time
from pathlib import Path

import numpy as np

PROMPT = "QENIVO> "
_PRIMAL_PYTHON_MAX = 3000          # rows + columns up to which primopt runs the true primal method

COMMANDS = {
    "read": "read FILE [mps|lp]: load a model (MPS free/fixed, .gz/.bz2, or LP format).",
    "optimize": "optimize: solve with QENIVO's router (MILP: branch and cut; LP: routed engine).",
    "primopt": "primopt: primal simplex (two-phase bounded primal, Python engine, up to "
               f"{_PRIMAL_PYTHON_MAX} rows+columns; larger models use the native core, which runs "
               "its own dual method, and the shell says so).",
    "tranopt": "tranopt: dual simplex (native C++ dual simplex; the Python fallback uses the dual "
               "method when the slack basis is dual feasible, else primal two-phase).",
    "baropt": "baropt: interior point (Mehrotra predictor-corrector; native IPM for QP).",
    "mipopt": "mipopt: branch and cut, stopping at the relative gap set by 'set mip tolerances mipgap'.",
    "display": "display solution objective|variables|dual|slacks|reduced|quality [ITEMS], "
               "display problem stats, display settings [changed|all]. ITEMS: - or * for all, "
               "names, glob patterns (x*), name1-name2 or 1-based positions i-j.",
    "set": "set timelimit SECONDS | threads N | mip tolerances mipgap G | tolerance T | certdir DIR | defaults",
    "write": "write FILE [sol|mps|lp|cert]: type from the word or the extension (.sol, .mps, .lp, .json).",
    "help": "help [COMMAND]: this text.",
    "quit": "quit: leave the shell.",
    "exit": "exit: same as quit.",
}

DEFAULTS = {"timelimit": 3600.0, "threads": 0, "mipgap": 1e-6, "tolerance": 1e-6, "certdir": "."}

NOTES = [
    "Notes on parameters:",
    "  timelimit  wall-clock seconds per solve (QENIVO default 3600).",
    "  threads    accepted and recorded in the certificate; QENIVO's single-model engines run one",
    "             solve thread, so the value does not change optimize. Case stacks use 'qenivo cases'.",
    "  mipgap     relative gap (incumbent - bound) / max(1, |incumbent|); QENIVO default 1e-6.",
    "  tolerance  QENIVO only: relative KKT tolerance an answer must pass to be called optimal.",
    "  certdir    QENIVO only: where the certificate JSON of each solve is written.",
    "Not available (refused with a message): other algorithm, emphasis, cut and heuristic switches,",
    "absolute MIP gap, basis files (.bas), parameter files (.prm), netopt, feasopt, add/change/delete.",
    ".sol files are QENIVO's plain-text solution format, not an XML solution format.",
]


class ShellError(Exception):
    """A command could not run; the shell reports it and goes on."""


def _match(word: str, options, what: str, first: bool = False) -> str:
    """Unique-prefix match; with first=True an ambiguous prefix takes the first option listed."""
    w = word.lower()
    if w in options:
        return w
    hits = [o for o in options if o.startswith(w)]
    if len(hits) == 1 or (hits and first):
        return hits[0]
    if not hits:
        raise ShellError(f"unknown {what} '{word}' (one of: {', '.join(options)})")
    raise ShellError(f"'{word}' is ambiguous: {', '.join(hits)}")


def _unquote(w: str) -> str:
    """Windows-safe word split keeps backslashes; drop one pair of surrounding quotes."""
    return w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in "\"'" else w


def _fmt(v: float) -> str:
    return f"{v:.12g}"


class Shell:
    def __init__(self, out=None, certdir: str | None = None):
        self.out = out or sys.stdout
        self.prob = None
        self.source: str | None = None
        self.sol = None
        self.cert_path: Path | None = None
        self.defaults = dict(DEFAULTS, certdir=certdir or DEFAULTS["certdir"])
        self.settings = dict(self.defaults)
        self.errors = 0

    # ------------------------------------------------------------------ plumbing
    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def run_line(self, line: str) -> bool:
        """Run one command line. Returns False when the shell should stop."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        try:
            words = [_unquote(w) for w in shlex.split(line, posix=False)]
        except ValueError as e:
            self.errors += 1
            self.say(f"Error: {e}")
            return True
        try:
            cmd = _match(words[0], COMMANDS, "command")
            if cmd in ("quit", "exit"):
                return False
            getattr(self, "cmd_" + cmd)(words[1:])
        except ShellError as e:
            self.errors += 1
            self.say(f"Error: {e}")
        return True

    def run_lines(self, lines) -> bool:
        for line in lines:
            if not self.run_line(line):
                return False
        return True

    def interact(self, stream, prompt: bool) -> None:
        while True:
            if prompt:
                self.out.write(PROMPT)
                self.out.flush()
            line = stream.readline()
            if not line or not self.run_line(line):
                return

    def _need_problem(self):
        if self.prob is None:
            raise ShellError("no problem loaded; use 'read FILE'")
        return self.prob

    def _need_solution(self):
        self._need_problem()
        if self.sol is None:
            raise ShellError("no solution; run optimize first")
        return self.sol

    # ------------------------------------------------------------------ commands
    def cmd_help(self, args):
        if args:
            self.say(COMMANDS[_match(args[0], COMMANDS, "command")])
            return
        for k, v in COMMANDS.items():
            self.say(f"{k:9s} {v}")
        self.say()
        for line in NOTES:
            self.say(line)

    def cmd_read(self, args):
        if not args:
            raise ShellError("read needs a file name")
        # An unquoted Windows path may contain spaces, so later words are part of the
        # name unless the last word is an explicit mps/lp type and the path before it exists.
        kind = None
        parts = args
        if len(args) >= 2 and args[-1].lower() in ("mps", "lp"):
            without = Path(" ".join(args[:-1]))
            if without.exists() or not Path(" ".join(args)).exists():
                kind = args[-1].lower()
                parts = args[:-1]
        path = Path(parts[0] if len(parts) == 1 else " ".join(parts))
        if kind is None:
            kind = "lp" if path.name.lower().endswith((".lp", ".lp.gz")) else "mps"
        if not path.exists():
            raise ShellError(f"no such file {path}")
        try:
            if kind == "lp":
                from ..io.lpformat import read_lp
                prob = read_lp(path)
            else:
                from ..api import read
                prob = read(path)
        except Exception as e:  # noqa: BLE001  (any parse failure is a user-facing message)
            raise ShellError(f"cannot read {path}: {e}") from None
        self.prob, self.source, self.sol, self.cert_path = prob, str(path), None, None
        kinds = "MILP" if prob.is_mip else ("QP" if prob.is_qp else "LP")
        self.say(f"Problem '{prob.name or path.stem}' read from {path}: {kinds}, {prob.size_str()}.")

    def cmd_optimize(self, args):
        self._solve("optimize")

    def cmd_primopt(self, args):
        self._solve("primopt")

    def cmd_tranopt(self, args):
        self._solve("tranopt")

    def cmd_baropt(self, args):
        self._solve("baropt")

    def cmd_mipopt(self, args):
        self._solve("mipopt")

    def cmd_display(self, args):
        if not args:
            raise ShellError("display what? solution, problem or settings")
        what = _match(args[0], ("solution", "problem", "settings"), "display option", first=True)
        if what == "settings":
            mode = _match(args[1], ("changed", "all"), "settings option") if len(args) > 1 else "changed"
            self._show_settings(mode)
        elif what == "problem":
            opt = _match(args[1], ("stats",), "problem option") if len(args) > 1 else "stats"
            if opt == "stats":
                self._show_stats()
        else:
            if len(args) < 2:
                raise ShellError("display solution objective|variables|dual|slacks|reduced|quality")
            opt = _match(args[1], ("objective", "variables", "dual", "slacks", "reduced", "quality"),
                         "solution option")
            getattr(self, "_show_" + opt)(args[2:])

    def cmd_set(self, args):
        if not args:
            raise ShellError(COMMANDS["set"])
        key = args[0].lower()
        if key == "defaults":
            self.settings = dict(self.defaults)
            self.say("All parameters reset to their defaults.")
            return
        known = ("timelimit", "threads", "mip", "tolerance", "certdir")
        hits = [k for k in known if k.startswith(key)]
        if len(hits) != 1:
            self.say(f"Note: parameter '{' '.join(args)}' has no QENIVO equivalent; nothing was set "
                     "(see 'help').")
            return
        key, rest = hits[0], args[1:]
        if key == "mip":
            if len(rest) >= 2 and "tolerances".startswith(rest[0].lower()) and rest[1].lower() == "mipgap":
                key, rest = "mipgap", rest[2:]
            else:
                self.say(f"Note: parameter 'mip {' '.join(rest)}' has no QENIVO equivalent; only "
                         "'mip tolerances mipgap' is available.")
                return
        if len(rest) != 1:
            raise ShellError(f"set {key} needs one value")
        raw = rest[0]
        try:
            if key == "certdir":
                val = raw
            elif key == "threads":
                val = int(raw)
                if val < 0:
                    raise ValueError
            else:
                val = float(raw)
                if not np.isfinite(val) or val < 0 or (key == "mipgap" and val > 1) or \
                        (key in ("timelimit", "tolerance") and val == 0):
                    raise ValueError
        except ValueError:
            raise ShellError(f"invalid value '{raw}' for {key}") from None
        self.settings[key] = val
        self.say(f"New value for {key}: {val}")
        if key == "threads" and val != 1:
            self.say("Note: QENIVO's single-model engines run one solve thread; the value is recorded "
                     "in the certificate only.")

    def cmd_write(self, args):
        if not args:
            raise ShellError("write needs a file name")
        path = Path(args[0])
        if len(args) > 1:
            kind = _match(args[1], ("sol", "mps", "lp", "cert"), "file type")
        else:
            low = path.name.lower()
            kind = ("sol" if low.endswith(".sol") else "lp" if low.endswith(".lp")
                    else "cert" if low.endswith(".json") else "mps" if low.endswith(".mps") else None)
            if kind is None:
                raise ShellError("cannot tell the file type; add sol, mps, lp or cert")
        prob = self._need_problem()
        if kind == "mps":
            from ..io.mps import write_mps
            write_mps(prob, path, name=prob.name or path.stem)
        elif kind == "lp":
            from ..io.lpformat import write_lp
            write_lp(prob, path)
        else:
            sol = self._need_solution()
            if kind == "cert":
                sol.save_certificate(path)
            else:
                if sol.x is None:
                    raise ShellError(f"no primal values to write (verdict {sol.verdict})")
                from ..io.mps import write_sol
                y_int = None if sol.y is None else prob.obj_sign * sol.y
                write_sol(path, prob, sol.x, y_int, status=sol.verdict,
                          kkt={k: v for k, v in sol.residuals.items() if k.startswith("rel_")},
                          col_statuses=(sol.extra.get("basis") or {}).get("col_statuses"),
                          row_statuses=(sol.extra.get("basis") or {}).get("row_statuses"))
        self.say(f"{kind.upper() if kind != 'cert' else 'Certificate'} written to {path}.")

    # ------------------------------------------------------------------ solving
    def _solve(self, how: str):
        from .. import api
        prob = self._need_problem()
        tl, tol = float(self.settings["timelimit"]), float(self.settings["tolerance"])
        if how in ("primopt", "tranopt", "baropt") and prob.is_mip:
            raise ShellError(f"the model has integer variables; {how} solves continuous models "
                             "(use mipopt or optimize)")
        if how in ("primopt", "tranopt") and prob.is_qp:
            raise ShellError("simplex is for LP; use baropt or optimize for a QP")
        if how == "mipopt" and not prob.is_mip:
            self.say("No integer variables: solving as a continuous model (optimize).")
            how = "optimize"
        t0 = time.perf_counter()
        if how == "primopt":
            sol = self._primal(prob, tol, tl)
        elif how == "tranopt":
            sol = api.solve(prob, engine="simplex", tol=tol, time_limit=tl)
        elif how == "baropt":
            sol = api.solve(prob, engine="ipm-native" if prob.is_qp else "ipm", tol=tol, time_limit=tl)
        elif prob.is_mip and not prob.is_qp:
            from ..engines.milp import solve_milp
            from ..workload.router import route
            sol = solve_milp(prob, tol=tol, time_limit=tl, gap_tol=float(self.settings["mipgap"]),
                             meta=route(prob, 1, "milp", "numpy"), source=self.source)
        else:
            sol = api.solve(prob, engine="auto", tol=tol, time_limit=tl)
        wall = time.perf_counter() - t0
        sol.source = self.source
        sol.extra["shell"] = {"command": how, "settings": dict(self.settings), "wall_time": wall}
        self.sol = sol
        self.cert_path = self._cert_file()
        sol.save_certificate(self.cert_path)
        self._report(sol, how, wall)

    def _primal(self, prob, tol, tl):
        from .. import api
        from ..engines.simplex import solve_simplex, solve_simplex_python
        from ..workload.router import route
        meta = route(prob, 1, "simplex", "numpy")
        if prob.m + prob.n <= _PRIMAL_PYTHON_MAX:
            out = solve_simplex_python(prob, tol=min(tol, 1e-9), time_limit=tl, dual=False)
            meta.update(method="primal simplex (two-phase, bounded)", engine_impl="python")
        else:
            self.say(f"Note: {prob.m + prob.n} rows+columns exceed the primal engine's size "
                     f"({_PRIMAL_PYTHON_MAX}); the native core runs its dual method instead.")
            out = solve_simplex(prob, tol=min(tol, 1e-9), time_limit=tl, dual=False)
            meta.update(method="native simplex core (dual method)", engine_impl=out.get("engine_impl"))
        meta.update(iterations=out["iterations"], time=out["time"], reason="primopt requested by the user")
        sol = api._finish(prob, out["status"], out["x"], out["y"], tol, meta, source=self.source)
        if out["status"] == "infeasible" and sol.verdict != "infeasible":
            sol = api._certify_infeasible(prob, tol, tl, meta, self.source) or sol
        if "col_statuses" in out:
            sol.extra["basis"] = {"col_statuses": out["col_statuses"], "row_statuses": out["row_statuses"]}
        return sol

    def _cert_file(self) -> Path:
        d = Path(self.settings["certdir"])
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(self.source).name.split(".")[0] if self.source else (self.prob.name or "model")
        return d / f"{stem}.cert.json"

    def _report(self, sol, how, wall):
        e = sol.engine
        if sol.verdict == "optimal":
            mip = sol.extra.get("mip")
            tag = (f"Proven optimal within relative gap {mip['gap']:.2e}" if mip else "Proven optimal")
            self.say(f"{tag}.  Objective = {sol.objective:.10e}")
        elif sol.verdict == "infeasible":
            self.say("Proven infeasible (Farkas certificate).")
        elif sol.verdict == "unbounded":
            self.say("Proven unbounded (improving ray and a feasible point).")
        else:
            why = e.get("note") or e.get("reason") or sol.status
            self.say(f"Not proven ({sol.status}): {why}")
            if sol.x is not None and sol.objective is not None:
                self.say(f"  best point found, NOT proven: objective {sol.objective:.10e}")
        self.say(f"{how}: engine {e.get('engine')}"
                 + (f" ({e['method']})" if e.get("method") else "")
                 + f", iterations {e.get('iterations')}, {wall:.3f} s")
        self.say(f"Certificate written to {self.cert_path}.")

    # ------------------------------------------------------------------ display
    def _select(self, names, tokens) -> list[int]:
        if not tokens or tokens in (["-"], ["*"]):
            return list(range(len(names)))
        pos = {nm: i for i, nm in enumerate(names)}
        picked: list[int] = []
        for tok in tokens:
            if tok in pos:
                picked.append(pos[tok])
            elif any(ch in tok for ch in "*?["):
                picked.extend(i for i, nm in enumerate(names) if fnmatch.fnmatchcase(nm, tok))
            elif "-" in tok:
                a, b = tok.split("-", 1)
                if a.isdigit() and b.isdigit():
                    lo, hi = int(a) - 1, int(b) - 1
                elif a in pos and b in pos:
                    lo, hi = pos[a], pos[b]
                else:
                    raise ShellError(f"no such item or range '{tok}'")
                if not (0 <= lo <= hi < len(names)):
                    raise ShellError(f"range '{tok}' is out of order or out of bounds")
                picked.extend(range(lo, hi + 1))
            elif tok.isdigit() and 1 <= int(tok) <= len(names):
                picked.append(int(tok) - 1)
            else:
                raise ShellError(f"no such item '{tok}'")
        return sorted(set(picked))

    def _table(self, title, names, values, tokens, unproven: bool):
        idx = self._select(names, tokens)
        width = max([len(title)] + [len(names[i]) for i in idx]) + 2
        if unproven:
            self.say("Warning: these values are NOT proven (see display solution quality).")
        self.say(f"{title:<{width}s}Value")
        zero = 0
        for i in idx:
            v = float(values[i])
            if abs(v) <= 1e-12:
                zero += 1
                continue
            self.say(f"{names[i]:<{width}s}{_fmt(v)}")
        if zero:
            self.say(f"({zero} of {len(idx)} selected values are zero and not shown)")

    def _show_objective(self, args):
        sol = self._need_solution()
        if sol.objective is None:
            self.say(f"No objective value: verdict {sol.verdict}.")
            return
        mip = sol.extra.get("mip")
        label = "Proven optimal" if sol.verdict == "optimal" else "NOT proven"
        self.say(f"Objective = {sol.objective:.10e}   ({label}; certificate {self.cert_path})")
        if mip:
            self.say(f"  best bound {mip['bound']:.10e}, relative gap {mip['gap']:.2e}, nodes {mip['nodes']}")

    def _show_variables(self, args):
        sol = self._need_solution()
        if sol.x is None:
            raise ShellError(f"no primal values (verdict {sol.verdict})")
        _, cn = sol.problem.names()
        self._table("Variable", cn, sol.x, args, sol.verdict != "optimal")

    def _show_dual(self, args):
        sol = self._need_solution()
        if sol.y is None:
            raise ShellError("no dual values (a MILP has none, or no solution was proven)")
        rn, _ = sol.problem.names()
        self._table("Constraint", rn, sol.y, args, sol.verdict != "optimal")

    def _show_reduced(self, args):
        sol = self._need_solution()
        rc = sol.reduced_costs
        if rc is None:
            raise ShellError("no reduced costs (a MILP has none, or no solution was proven)")
        _, cn = sol.problem.names()
        self._table("Variable", cn, rc, args, sol.verdict != "optimal")

    def _show_slacks(self, args):
        sol = self._need_solution()
        if sol.x is None:
            raise ShellError(f"no primal values (verdict {sol.verdict})")
        rn, _ = sol.problem.names()
        self._table("Constraint", rn, slacks(sol.problem, sol.x), args, sol.verdict != "optimal")

    def _show_quality(self, args):
        sol = self._need_solution()
        self.say(f"Verdict {sol.verdict} (engine status {sol.status}), tolerance {sol.tol:g}")
        for k in ("rel_primal", "rel_dual", "rel_gap", "max_rel", "max_row_violation", "max_integrality", "gap"):
            if k in sol.residuals:
                self.say(f"  {k:18s} {float(sol.residuals[k]):.3e}")
        self.say(f"  certificate        {self.cert_path}")
        src = self.source or ""
        if sol.is_proven and ".mps" in src.lower() and not sol.problem.is_mip:
            from ..certify.certificate import load
            from ..certify.verify import verify
            rep = verify(src, load(self.cert_path))
            self.say("  independent check  " + ("passed" if rep["passed"] else "FAILED") +
                     " (separate MPS reader, compensated sums)")

    def _show_settings(self, mode):
        shown = 0
        for k, v in self.settings.items():
            d = self.defaults[k]
            if mode == "all" or v != d:
                self.say(f"{k:10s} {v}" + ("" if v == d else f"   (default {d})"))
                shown += 1
        if not shown:
            self.say("All parameters are at their defaults.")

    def _show_stats(self):
        p = self._need_problem()
        st = problem_stats(p)
        self.say(f"Problem name   : {p.name or '(none)'}   sense: {st['sense']}   source: {self.source}")
        self.say(f"Variables      : {p.n:>9d}  [continuous {st['continuous']}, integer {st['integer']}, "
                 f"binary {st['binary']}]")
        b = st["bounds"]
        self.say(f"  bounds       : lower only {b['lower']}, upper only {b['upper']}, boxed {b['boxed']}, "
                 f"fixed {b['fixed']}, free {b['free']}")
        r = st["rows"]
        self.say(f"Constraints    : {p.m:>9d}  [less {r['L']}, greater {r['G']}, equal {r['E']}, "
                 f"ranged {r['R']}, free {r['N']}]")
        self.say(f"Nonzeros       : {p.nnz:>9d}   objective nonzeros {st['obj_nnz']}"
                 + (f"   quadratic nonzeros {st['q_nnz']}" if st["q_nnz"] else ""))
        for label, key in (("matrix", "matrix"), ("objective", "objective"), ("bounds", "bound"),
                           ("rhs", "rhs")):
            lo, hi = st["ranges"][key]
            self.say(f"  |{label}| range{'':<{10 - len(label)}}: " +
                     ("all zero or none" if lo is None else f"[{lo:.3e}, {hi:.3e}]"))


# ---------------------------------------------------------------------- helpers
def _row_kinds(p) -> np.ndarray:
    lo, hi = np.isfinite(p.lc), np.isfinite(p.uc)
    kind = np.full(p.m, "N", dtype="<U1")
    kind[hi & ~lo] = "L"
    kind[lo & ~hi] = "G"
    both = lo & hi
    kind[both & (p.lc == p.uc)] = "E"
    kind[both & (p.lc != p.uc)] = "R"
    return kind


def slacks(p, x) -> np.ndarray:
    """Row slack: right-hand side minus activity for <=, >= and = rows; activity minus the lower
    end for a ranged row; NaN for a free row (the convention of the classic callable libraries)."""
    act = p.A @ np.asarray(x, dtype=float)
    kind = _row_kinds(p)
    out = np.full(p.m, np.nan)
    m = kind == "L"
    out[m] = p.uc[m] - act[m]
    m = (kind == "G") | (kind == "E")
    out[m] = p.lc[m] - act[m]
    m = kind == "R"
    out[m] = act[m] - p.lc[m]
    return out


def _abs_range(v):
    v = np.abs(np.asarray(v, dtype=float))
    v = v[np.isfinite(v) & (v > 0)]
    return (None, None) if v.size == 0 else (float(v.min()), float(v.max()))


def problem_stats(p) -> dict:
    integer = p.integer if p.integer is not None else np.zeros(p.n, bool)
    binary = integer & (p.lx >= 0) & (p.ux <= 1) & (p.lx > -1) & (p.ux < 2)
    lo, hi = np.isfinite(p.lx), np.isfinite(p.ux)
    kind = _row_kinds(p)
    return {
        "sense": "maximize" if p.obj_sign < 0 else "minimize",
        "continuous": int((~integer).sum()), "integer": int((integer & ~binary).sum()), "binary": int(binary.sum()),
        "bounds": {"lower": int((lo & ~hi).sum()), "upper": int((hi & ~lo).sum()),
                   "boxed": int((lo & hi & (p.lx != p.ux)).sum()), "fixed": int((lo & hi & (p.lx == p.ux)).sum()),
                   "free": int((~lo & ~hi).sum())},
        "rows": {k: int((kind == k).sum()) for k in "LGERN"},
        "obj_nnz": int(np.count_nonzero(p.c)), "q_nnz": int(p.Q.nnz) if p.is_qp else 0,
        "ranges": {"matrix": _abs_range(p.A.data), "objective": _abs_range(p.c),
                   "bound": _abs_range(np.concatenate([p.lx, p.ux])),
                   "rhs": _abs_range(np.concatenate([p.lc, p.uc]))},
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qenivo-io", description="QENIVO optimizer shell (certified answers).")
    ap.add_argument("model", nargs="?", help="model file to read first")
    ap.add_argument("-c", dest="commands", nargs="+", metavar="CMD", help="commands to run, each quoted")
    ap.add_argument("-f", dest="file", help="file with one command per line")
    ap.add_argument("-i", dest="interactive", action="store_true", help="stay interactive after -c / -f")
    ap.add_argument("--certdir", help="directory for certificates (default: current directory)")
    return ap


def main(argv=None) -> int:
    """Exit code 0 when every command ran, 1 when any command failed, 2 on a usage error."""
    a = build_parser().parse_args(argv)
    sh = Shell(certdir=a.certdir)
    go = True
    if a.model:
        go = sh.run_line(f'read "{a.model}"')
    if go and a.file:
        try:
            lines = Path(a.file).read_text(encoding="utf-8").splitlines()
        except OSError as e:
            print(f"Error: cannot read command file {a.file}: {e}", file=sys.stderr)
            return 2
        go = sh.run_lines(lines)
    if go and a.commands:
        go = sh.run_lines(a.commands)
    if go and (a.interactive or not (a.file or a.commands)):
        sh.interact(sys.stdin, prompt=sys.stdin.isatty())
    return 1 if sh.errors else 0


if __name__ == "__main__":
    sys.exit(main())
