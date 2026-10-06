# Compatibility: the `qenivo-io` shell, the QNV C interface, and shadow mode

Planners at a refinery already script a commercial optimizer: a command file that reads an MPS
matrix, optimizes, prints the objective and duals, and writes a solution. QENIVO offers three
familiar front doors so that this habit carries over. All three are independent implementations
written from public documentation (sources below). Every answer still goes through QENIVO's
certificate path: a result that fails its KKT or Farkas check is reported as not proven.

| Part | Where | Entry point |
|---|---|---|
| Command shell | `src/qenivo/compat/shell.py` | `qenivo-io`, `qenivo io` (once the lead wires them), `python -m qenivo.compat.shell` |
| C interface subset (QNV) | `capi/compat/` | `qnv_compat.h`, `qnv_compat.c`; test program `test_qnv_compat.c` |
| Shadow mode | `src/qenivo/compat/shadow.py` | `python -m qenivo.compat.shadow` |

## 1. The `qenivo-io` shell

Invocation:

```text
qenivo-io                                   interactive; a prompt on a terminal, else commands from stdin
qenivo-io plan.mps                          read the model first
qenivo-io -c "read plan.mps" "optimize" "display solution objective"
qenivo-io -f commands.txt [-i]              run a command file; -i stays interactive afterwards
qenivo-io ... --certdir certs/              where the certificate of each solve goes (default: here)
```

Exit code: 0 when every command ran, 1 when any command failed (unknown command, missing model,
bad value), 2 on a usage error. Any unique prefix of a command word works (`disp sol var -`).
After `display`, an ambiguous single letter takes the first option in the order solution, problem,
settings, so `d s o` means `display solution objective`.

### Supported commands

| Command | QENIVO behaviour |
|---|---|
| `read FILE [mps\|lp]` | MPS (free or fixed, `.gz`/`.bz2`) or LP format; type from the word or the extension |
| `optimize` | MILP: branch and cut with the `mipgap` setting; LP / QP: QENIVO's router (`qenivo.api.solve`, engine auto) |
| `primopt` | primal simplex: the two-phase bounded primal method (Python engine) for models up to 3,000 rows + columns; above that the native C++ core runs its own dual method and the shell prints a note saying so |
| `tranopt` | dual simplex: the native C++ dual simplex (`engine="simplex"`); its Python fallback uses the dual method when the slack basis is dual feasible, else primal two-phase |
| `baropt` | interior point: Mehrotra predictor-corrector (`ipm`; `ipm-native` for a QP) |
| `mipopt` | branch and cut; on a model without integer columns it says so and runs `optimize` |
| `display solution objective` | objective, the verdict, and for a MILP the best bound, gap and node count |
| `display solution variables\|dual\|slacks\|reduced [ITEMS]` | values over a selection; zero values are counted, not listed |
| `display solution quality` | verdict, KKT residuals, certificate path, and for an MPS LP the independent verifier's result |
| `display problem stats` | sizes, variable types, bound kinds, row kinds, nonzeros, coefficient ranges |
| `display settings [changed\|all]` | the parameters below |
| `set timelimit S` | wall-clock seconds per solve (default 3600) |
| `set threads N` | accepted and recorded in the certificate; single-model engines run one solve thread, so it does not change a solve (stacks of cases use `qenivo cases`) |
| `set mip tolerances mipgap G` | relative gap passed to branch and cut (default 1e-6, range 0 to 1) |
| `set tolerance T` | QENIVO only: the relative KKT tolerance an answer must pass to be called optimal |
| `set certdir DIR` | QENIVO only: certificate directory |
| `set defaults` | reset all parameters |
| `write FILE [sol\|mps\|lp\|cert]` | QENIVO `.sol` text, MPS, LP, or the certificate JSON |
| `help [COMMAND]`, `quit`, `exit` | |

ITEMS: `-` or `*` for all, names, glob patterns (`X0*`), a name range `X01-X04`, or 1-based
positions `1-4`. Slack convention: right-hand side minus activity for `<=`, `>=` and `=` rows,
activity minus the lower end for a ranged row, undefined for a free row. Duals and reduced costs
are in the user's objective sense (`c - A'y` are the reduced costs).

After every solve the shell writes `<model>.cert.json` (in `certdir`), the same schema as
`qenivo solve --cert`, plus a `shell` block with the command and every setting. `qenivo verify`
re-checks it.

### Differences and things that are not there

* The MIP gap formula is QENIVO's: `(incumbent - bound) / max(1, |incumbent|)`. The reference
  manual's relative gap divides by `1e-10 + |incumbent|` and its default is 1e-4; QENIVO's default
  is 1e-6. Results within the gap are labelled "Proven optimal within relative gap g".
* `.sol` files are QENIVO's plain-text format (status, objective, values, duals, basis status),
  not an XML solution file. Basis (`.bas`) and parameter (`.prm`) files are not read or written.
* Refused with a message, not silently accepted: other `set` parameters (emphasis, cuts,
  heuristics, absolute gap, algorithm switches, ...); `set` messages say "has no QENIVO equivalent".
* Not implemented: `add`, `change`, `delete`, `enter`, `netopt`, `feasopt` (QENIVO's infeasibility
  explanation is `qenivo explain`), `xecute`, solution pools, display of the model's rows.
* `primopt`, `tranopt` and `baropt` refuse a model with integer columns (use `mipopt` or `optimize`);
  the simplex commands refuse a QP.

## 2. The QNV C interface (`capi/compat/`)

`qnv_compat.h` gives the environment / problem-object shape that callable libraries use, with
QENIVO's own names, constants and status codes, over the existing QENIVO C ABI (`capi/qenivo.h`):

* `QNVopenenv`, `QNVcloseenv`, `QNVerrorstring`, `QNVset/getdblparam`, `QNVset/getintparam`
* `QNVcreateprob`, `QNVfreeprob`, `QNVcopylp` (column-major matrix with row senses `L G E R`, right-hand
  sides and range values), `QNVcopyctype` (`C I B`), `QNVchgobjsen`, `QNVgetnumrows`, `QNVgetnumcols`
* `QNVlpopt`, `QNVprimopt`, `QNVdualopt`, `QNVbaropt`, `QNVmipopt`
* `QNVgetstat`, `QNVgetstatstring`, `QNVgetobjval`, `QNVgetx`, `QNVgetpi`, `QNVgetslack`, `QNVgetdj`,
  `QNVsolution`, `QNVgetcertificate`, `QNVgetengine`

Certification: `QNV_STAT_OPTIMAL / INFEASIBLE / UNBOUNDED` only when proven. A native-simplex optimum
must pass the C ABI's KKT residual check; any other native outcome is re-solved through QENIVO's
Python certificate path (an infeasible model then carries a checked Farkas ray). Maximisation is
solved as the negated minimisation and objective, duals and reduced costs are returned in the
user's sense; the certificate JSON describes the minimisation form the ABI solved.

Limits: `QNV_PARAM_MIPGAP` returns `QNV_ERR_UNSUPPORTED` (the C ABI has no gap argument; the shell
has one). `QNV_PARAM_THREADS` accepts 0 or 1 only. `QNVprimopt` runs the native core, which follows
its own dual method (the ABI has no primal-only switch). There is no file reader in the C subset:
models are passed as arrays. Values of `|v| >= 1e20` are infinite.

Build and test: `python capi/compat/build_qnv.py` (GCC or Clang, static on Windows, cached by
source hash) or `capi/compat/CMakeLists.txt`. The test program runs 70 checks: argument errors,
a maximisation LP with `L`, `G` and ranged rows through dual, primal and barrier (objective,
primal, duals, slacks, reduced costs), the sense flipped to minimise, a certified infeasible model,
and a binary knapsack. `tests/test_compat_shell.py::test_qnv_c_program_passes_every_check` builds
and runs it, and skips when no compiler is found.

No callable-library-style vendor names are mapped (for example no `#define` from a vendor routine
name to a QNV function). Such a mapping header would put a vendor's product prefix in our source;
the familiar shape is kept, the names are not. If a customer needs a source-level shim, it should be
written against the customer's own licensed headers, outside this repository.

## 3. Shadow mode

```text
python -m qenivo.compat.shadow plan.mps --external "wrapper {model} {solution}" [--report diff.json]
                                        [--cert plan.cert.json] [--engine auto] [--tol 1e-6] [--obj-tol 1e-6]
```

The external solver runs as a separate command configured by the user (`--external` or the
`QENIVO_SHADOW_CMD` environment variable); nothing of it is imported into QENIVO. `{model}` is the
model path; the command writes `{solution}` (or prints to stdout) as JSON:
`{"status", "objective", "x": {column: value}, "y": {row: value}}`, duals in the user's sense.

QENIVO solves and writes its certificate. The external answer is then checked by QENIVO's residual
code on the same model: claimed objective against `c'x`, primal feasibility, integrality for a
MILP, and with duals the dual feasibility and duality gap. Verdicts, objectives, primal and dual
values are compared; different vectors with equal checked objectives are reported as an
alternative optimum (or several dual optima), not as a mismatch. Exit 0 agree, 1 mismatch,
2 usage, 3 the external command failed.

`tests/test_compat_shadow.py` uses HiGHS (bench-only `highspy`) in a child process as the
external solver when installed, and a fixed-answer command otherwise: agreement on afiro and a
binary knapsack, and mismatches for a wrong objective, an infeasible `x`, a wrong status, wrong
duals and missing columns.

## 4. Public sources used

All pages are public IBM Documentation for IBM ILOG CPLEX Optimization Studio 22.1, read on
1 to 2 October 2026. They were used only to learn which commands, options and conventions exist
and what they mean; the descriptions above are our own wording and nothing was copied.

1. Interactive Optimizer, table of commands: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/InteractiveOptimizer/topics/commands.html`
   (command names: read, optimize, primopt, tranopt, baropt, mipopt, display, set, write, quit, help)
2. Interactive Optimizer overview page: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/homepages/interactiveoptimizercommands.html`
3. Using Interactive Optimizer commands from the operating system: `https://www.ibm.com/docs/en/SSSA5P_22.1.2/ilog.odms.cplex.help/CPLEX/InteractiveOptimizer/topics/useCmdsFileFromOS.html`
   (the `-c`, `-f` and `-i` invocation)
4. Managing parameters in the Interactive Optimizer: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/InteractiveOptimizer/topics/manageParameters.html`
   (`display settings changed / all`)
5. Relative MIP gap tolerance: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html`
   (path `mip tolerances mipgap`, its formula, default and range, used to document our difference)
6. Optimizer time limit: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/TiLim.html` (path `timelimit`; known from the search-result summary of this page, not re-read in full)
7. Global thread count: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Threads.html` (path `threads`, 0 = automatic)
8. Callable library, copying an LP: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/refcallablelibrary/cpxapi/copylp.html`
   (the column-major layout, row sense letters and the range-value rule, which are a data format)
9. Callable library, slacks: `https://www.ibm.com/docs/zh/SSSA5P_22.1.0/ilog.odms.cplex.help/refcallablelibrary/cpxapi/getslack.html`
   (slack sign convention for ordinary and ranged rows)

No FICO Xpress material was used; the shell does not attempt Xpress console compatibility.

## 5. Legal-review note (for the owner and any counsel; not legal advice)

* **What was taken:** only facts about behaviour and interfaces: command words, parameter paths,
  the `-c/-f/-i` invocation, a matrix layout and sign conventions. These are functional elements a
  user needs in order to reuse their own scripts. No source code, header file, documentation text,
  sample, output format sample or screenshot of any vendor product was copied into this repository;
  help texts, messages and output layouts are our own.
* **Trademarks:** no product, file, module, function, constant or command of ours carries a vendor
  trademark. Our names are `qenivo-io`, `QNV*` and `qenivo.compat`. The documentation names the
  vendor products only to say what we are compatible with (nominative use), for example in this
  file and in the shell's module docstring. Marketing text should say "command-compatible with
  common optimizer shells" or name the product only with "compatible with" and a trademark notice.
* **Vendor-style function names** were deliberately not mapped (see section 2).
* **Licences:** the IBM pages were read as public web documentation; we did not accept any product
  licence, download any vendor software, or use a vendor installation to observe behaviour. Points
  where our behaviour may differ from the vendor's (gap formula, `.sol` format, abbreviation rules)
  are listed above and were not reverse-engineered.
* **Shadow mode** runs a solver the user has installed and licensed, as a separate process chosen
  by the user. QENIVO neither ships nor links it. HiGHS (MIT licence) is used only in tests, in a
  child process, and only when present.
* **Before external release:** a reviewer should confirm the nominative-use wording in the README,
  deck and portal text, and re-read sections 1 to 3 against the sources above for any phrase that
  tracks the vendor's wording too closely.
