/* python_bridge.cpp — this file embeds Python.
 *
 * Engines other than the native simplex are solved by loading the CPython C API
 * at runtime and calling qenivo.solve inside that interpreter. The simplex path
 * never enters this translation unit. Symbols are resolved with GetProcAddress
 * or dlsym so a MinGW build can embed an MSVC CPython (Anaconda, python.org)
 * without linking its import library. The solver still runs in-process.
 *
 * The text protocol below is internal. Callers see only the C ABI in qenivo.h.
 * Set QENIVO_PYTHON_DLL to the shared library if it is not already loaded, and
 * QENIVO_SRC to the directory that contains the qenivo package.
 */

#include "python_bridge.h"

#include "qenivo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <dlfcn.h>
#endif

namespace {

struct PyObject;

using Py_ssize_t = intptr_t;

struct Api {
    int (*is_initialized)(void) = nullptr;
    void (*initialize_ex)(int) = nullptr;
    int (*gil_ensure)(void) = nullptr;
    void (*gil_release)(int) = nullptr;
    int (*run_string)(const char*) = nullptr;
    PyObject* (*add_module)(const char*) = nullptr;
    PyObject* (*module_dict)(PyObject*) = nullptr;
    PyObject* (*dict_get_string)(PyObject*, const char*) = nullptr;
    PyObject* (*unicode_from_string)(const char*) = nullptr;
    const char* (*unicode_as_utf8)(PyObject*) = nullptr;
    PyObject* (*tuple_new)(Py_ssize_t) = nullptr;
    int (*tuple_set)(PyObject*, Py_ssize_t, PyObject*) = nullptr;
    PyObject* (*call_object)(PyObject*, PyObject*) = nullptr;
    void (*decref)(PyObject*) = nullptr;
    PyObject* (*err_occurred)(void) = nullptr;
    void (*err_clear)(void) = nullptr;
    void (*err_print)(void) = nullptr;
    void (*err_fetch)(PyObject**, PyObject**, PyObject**) = nullptr;
    PyObject* (*object_str)(PyObject*) = nullptr;
};

std::mutex g_py_mu;
Api g_api;
bool g_api_ready = false;
bool g_script_ready = false;
std::string g_load_error;

#if defined(_WIN32)
using Module = HMODULE;
#else
using Module = void*;
#endif

void* symbol_of(Module mod, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(mod, name));
#else
    return dlsym(mod, name);
#endif
}

Module load_python_module() {
#if defined(_WIN32)
    if (const char* env = std::getenv("QENIVO_PYTHON_DLL")) {
        if (env[0] != '\0') {
            HMODULE mod = LoadLibraryA(env);
            if (mod) return mod;
        }
    }
    const char* names[] = {"python313.dll", "python312.dll", "python311.dll",
                           "python314.dll", "python310.dll", nullptr};
    for (int i = 0; names[i]; ++i) {
        HMODULE mod = GetModuleHandleA(names[i]);
        if (mod) return mod;
    }
    for (int i = 0; names[i]; ++i) {
        HMODULE mod = LoadLibraryA(names[i]);
        if (mod) return mod;
    }
    char python_exe[MAX_PATH];
    if (SearchPathA(nullptr, "python.exe", nullptr, MAX_PATH, python_exe, nullptr)) {
        char* slash = std::strrchr(python_exe, '\\');
        if (slash) {
            *slash = '\0';
            for (int i = 0; names[i]; ++i) {
                char full[MAX_PATH];
                std::snprintf(full, sizeof(full), "%s\\%s", python_exe, names[i]);
                HMODULE mod = LoadLibraryA(full);
                if (mod) return mod;
            }
        }
    }
    return nullptr;
#else
    if (const char* env = std::getenv("QENIVO_PYTHON_DLL")) {
        if (env[0] != '\0') {
            void* mod = dlopen(env, RTLD_NOW | RTLD_GLOBAL);
            if (mod) return mod;
        }
    }
    if (dlsym(RTLD_DEFAULT, "Py_IsInitialized")) return RTLD_DEFAULT;
    const char* names[] = {"libpython3.13.so", "libpython3.12.so", "libpython3.11.so",
                           "libpython3.14.so", "libpython3.10.so", "libpython3.so", nullptr};
    for (int i = 0; names[i]; ++i) {
        void* mod = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (mod) return mod;
    }
    return nullptr;
#endif
}

template <class Fn>
bool bind(Module mod, Fn& slot, const char* name) {
    void* p = symbol_of(mod, name);
    if (!p) {
        g_load_error = std::string("Python is embedded but the export is missing: ") + name;
        return false;
    }
    std::memcpy(&slot, &p, sizeof(slot));
    return true;
}

bool load_api() {
    if (g_api_ready) return true;
    Module mod = load_python_module();
    if (!mod) {
        g_load_error = "Python is embedded but python3xx.dll / libpython was not found "
                       "(set QENIVO_PYTHON_DLL)";
        return false;
    }
    Api api;
    bool ok = true;
    ok = bind(mod, api.is_initialized, "Py_IsInitialized") && ok;
    ok = bind(mod, api.initialize_ex, "Py_InitializeEx") && ok;
    ok = bind(mod, api.gil_ensure, "PyGILState_Ensure") && ok;
    ok = bind(mod, api.gil_release, "PyGILState_Release") && ok;
    ok = bind(mod, api.run_string, "PyRun_SimpleString") && ok;
    ok = bind(mod, api.add_module, "PyImport_AddModule") && ok;
    ok = bind(mod, api.module_dict, "PyModule_GetDict") && ok;
    ok = bind(mod, api.dict_get_string, "PyDict_GetItemString") && ok;
    ok = bind(mod, api.unicode_from_string, "PyUnicode_FromString") && ok;
    ok = bind(mod, api.unicode_as_utf8, "PyUnicode_AsUTF8") && ok;
    ok = bind(mod, api.tuple_new, "PyTuple_New") && ok;
    ok = bind(mod, api.tuple_set, "PyTuple_SetItem") && ok;
    ok = bind(mod, api.call_object, "PyObject_CallObject") && ok;
    ok = bind(mod, api.decref, "Py_DecRef") && ok;
    ok = bind(mod, api.err_occurred, "PyErr_Occurred") && ok;
    ok = bind(mod, api.err_clear, "PyErr_Clear") && ok;
    ok = bind(mod, api.err_print, "PyErr_Print") && ok;
    ok = bind(mod, api.err_fetch, "PyErr_Fetch") && ok;
    ok = bind(mod, api.object_str, "PyObject_Str") && ok;
    if (!ok) return false;
    g_api = api;
    g_api_ready = true;
    return true;
}

std::string python_exception_text() {
    if (!g_api.err_occurred || !g_api.err_occurred()) return "Python error";
    PyObject *typ = nullptr, *val = nullptr, *tb = nullptr;
    g_api.err_fetch(&typ, &val, &tb);
    std::string text = "Python error";
    PyObject* as_str = val ? g_api.object_str(val) : nullptr;
    if (as_str) {
        const char* utf = g_api.unicode_as_utf8(as_str);
        if (utf && utf[0]) text = utf;
        g_api.decref(as_str);
    }
    if (typ) g_api.decref(typ);
    if (val) g_api.decref(val);
    if (tb) g_api.decref(tb);
    g_api.err_clear();
    return text;
}

constexpr const char* SCRIPT = R"PYEND(
def capi_entry(payload):
    try:
        return _capi_solve(payload)
    except Exception as exc:
        return "ERR\n" + str(exc).replace("\n", " ")

def _floats(tokens, n):
    import math
    out = []
    for _ in range(n):
        t = next(tokens)
        if t == "inf":
            out.append(math.inf)
        elif t == "-inf":
            out.append(-math.inf)
        else:
            out.append(float(t))
    return out

def _ints(tokens, n):
    return [int(next(tokens)) for _ in range(n)]

def _json_default(o):
    import math
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(o, np.floating):
        v = float(o)
        return None if not math.isfinite(v) else v
    if np is not None and isinstance(o, np.integer):
        return int(o)
    if np is not None and isinstance(o, np.ndarray):
        return o.tolist()
    if np is not None and isinstance(o, np.bool_):
        return bool(o)
    return str(o)

def _fmt_vec(v):
    import numpy as np
    if v is None:
        return "none"
    arr = np.asarray(v, dtype=float).reshape(-1)
    body = " ".join("%.17g" % float(a) for a in arr)
    return str(int(arr.size)) + (" " + body if arr.size else "")

def _capi_solve(payload):
    import json
    import os
    import sys
    src = os.environ.get("QENIVO_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    import numpy as np
    import scipy.sparse as sp
    import qenivo
    tok = iter(payload.split())
    m = int(next(tok))
    n = int(next(tok))
    nnz = int(next(tok))
    ap = _ints(tok, m + 1)
    ai = _ints(tok, nnz)
    ax = _floats(tok, nnz)
    c = _floats(tok, n)
    lc = _floats(tok, m)
    uc = _floats(tok, m)
    lx = _floats(tok, n)
    ux = _floats(tok, n)
    has_int = int(next(tok))
    integer = np.asarray(_ints(tok, n), dtype=bool) if has_int else None
    has_q = int(next(tok))
    Q = None
    if has_q:
        qn = int(next(tok))
        qp = _ints(tok, n + 1)
        qi = _ints(tok, qn)
        qx = _floats(tok, qn)
        Q = sp.csr_matrix((qx, qi, qp), shape=(n, n)) if qn else sp.csr_matrix((n, n))
    engine = next(tok)
    tol = float(next(tok))
    time_limit = float(next(tok))
    name = next(tok)
    if nnz:
        A = sp.csr_matrix((ax, ai, ap), shape=(m, n))
    else:
        A = sp.csr_matrix((m, n), dtype=float)
    prob = qenivo.Problem(c=np.asarray(c, dtype=float), A=A,
                          lc=np.asarray(lc, dtype=float), uc=np.asarray(uc, dtype=float),
                          lx=np.asarray(lx, dtype=float), ux=np.asarray(ux, dtype=float),
                          integer=integer, Q=Q, name=name)
    sol = qenivo.solve(prob, engine=engine, tol=tol, time_limit=time_limit)
    basis = {}
    if sol.extra:
        basis = sol.extra.get("basis") or {}
    code = {"basic": 0, "at_lower": 1, "at_upper": 2, "free": 3}
    def basis_line(key):
        seq = basis.get(key) if isinstance(basis, dict) else None
        if not seq:
            return "none"
        nums = [code.get(s, 1) for s in seq]
        return str(len(nums)) + " " + " ".join(str(k) for k in nums)
    obj = sol.objective
    obj_s = "none" if obj is None else "%.17g" % float(obj)
    iters = int((sol.engine or {}).get("iterations") or 0)
    cert = json.dumps(sol.certificate(), default=_json_default, allow_nan=False)
    rc = sol.reduced_costs
    return "\n".join([
        "OK",
        str(sol.status),
        str(sol.verdict),
        obj_s,
        str(iters),
        "X " + _fmt_vec(sol.x),
        "Y " + _fmt_vec(sol.y),
        "RC " + _fmt_vec(rc),
        "BC " + basis_line("col_statuses"),
        "BR " + basis_line("row_statuses"),
        "CERT " + cert,
    ])
)PYEND";

void emit_double(std::ostream& out, double v) {
    if (std::isinf(v)) {
        out << (v > 0 ? "inf" : "-inf");
        return;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    out << buf;
}

std::string build_payload(const PythonRequest& req) {
    std::ostringstream out;
    const int32_t nnz = req.a_ptr.empty() ? 0 : req.a_ptr.back();
    out << req.rows << ' ' << req.cols << '\n' << nnz << '\n';
    for (int32_t v : req.a_ptr) out << v << ' ';
    out << '\n';
    for (int32_t v : req.a_idx) out << v << ' ';
    out << '\n';
    for (double v : req.a_val) { emit_double(out, v); out << ' '; }
    out << '\n';
    for (double v : req.cost) { emit_double(out, v); out << ' '; }
    out << '\n';
    for (double v : req.row_lower) { emit_double(out, v); out << ' '; }
    out << '\n';
    for (double v : req.row_upper) { emit_double(out, v); out << ' '; }
    out << '\n';
    for (double v : req.col_lower) { emit_double(out, v); out << ' '; }
    out << '\n';
    for (double v : req.col_upper) { emit_double(out, v); out << ' '; }
    out << '\n' << (req.has_integer ? 1 : 0) << '\n';
    if (req.has_integer) {
        for (int32_t v : req.integer) out << v << ' ';
        out << '\n';
    }
    out << (req.has_q ? 1 : 0) << '\n';
    if (req.has_q) {
        const int32_t qn = req.q_ptr.empty() ? 0 : req.q_ptr.back();
        out << qn << '\n';
        for (int32_t v : req.q_ptr) out << v << ' ';
        out << '\n';
        for (int32_t v : req.q_idx) out << v << ' ';
        out << '\n';
        for (double v : req.q_val) { emit_double(out, v); out << ' '; }
        out << '\n';
    }
    out << req.engine << '\n';
    emit_double(out, req.tolerance);
    out << '\n';
    emit_double(out, req.time_limit);
    out << '\n' << req.name << '\n';
    return out.str();
}

double parse_number(const std::string& tok) {
    if (tok == "inf") return HUGE_VAL;
    if (tok == "-inf") return -HUGE_VAL;
    return std::strtod(tok.c_str(), nullptr);
}

bool parse_vec_line(const std::string& line, const char* key, std::vector<double>& out, bool& present,
                    std::string& error) {
    std::istringstream in(line);
    std::string got, mark;
    if (!(in >> got) || got != key) {
        error = std::string("bridge reply missing ") + key;
        return false;
    }
    if (!(in >> mark)) {
        error = std::string("bridge reply truncated at ") + key;
        return false;
    }
    if (mark == "none") {
        present = false;
        out.clear();
        return true;
    }
    int n = std::atoi(mark.c_str());
    if (n < 0) {
        error = "negative vector length from Python";
        return false;
    }
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::string num;
        if (!(in >> num)) {
            error = std::string("short vector from Python for ") + key;
            return false;
        }
        out[static_cast<size_t>(i)] = parse_number(num);
    }
    present = true;
    return true;
}

bool parse_basis_line(const std::string& line, const char* key, std::vector<int32_t>& out, bool& present,
                      std::string& error) {
    std::istringstream in(line);
    std::string got, mark;
    if (!(in >> got) || got != key) {
        error = std::string("bridge reply missing ") + key;
        return false;
    }
    if (!(in >> mark)) {
        error = std::string("bridge reply truncated at ") + key;
        return false;
    }
    if (mark == "none") {
        present = false;
        out.clear();
        return true;
    }
    int n = std::atoi(mark.c_str());
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        int v = 0;
        if (!(in >> v)) {
            error = "short basis from Python";
            return false;
        }
        out[static_cast<size_t>(i)] = v;
    }
    present = true;
    return true;
}

int32_t status_from_verdict(const std::string& verdict) {
    if (verdict == "optimal") return 0;
    if (verdict == "infeasible") return 1;
    if (verdict == "unbounded") return 2;
    return 6;
}

bool parse_reply(const std::string& text, PythonResponse& response) {
    std::istringstream in(text);
    std::string line;
    if (!std::getline(in, line)) {
        response.error = "empty reply from embedded Python";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "OK") {
        std::ostringstream rest;
        rest << line;
        std::string more;
        while (std::getline(in, more)) rest << ' ' << more;
        response.error = rest.str();
        return false;
    }
    auto take = [&](std::string& dest) -> bool {
        if (!std::getline(in, dest)) return false;
        if (!dest.empty() && dest.back() == '\r') dest.pop_back();
        return true;
    };
    if (!take(response.status_name) || !take(response.verdict)) {
        response.error = "truncated status from embedded Python";
        return false;
    }
    std::string obj_line, iter_line;
    if (!take(obj_line) || !take(iter_line)) {
        response.error = "truncated objective from embedded Python";
        return false;
    }
    if (obj_line == "none") {
        response.has_objective = false;
    } else {
        response.has_objective = true;
        response.objective = parse_number(obj_line);
    }
    response.iterations = std::strtoll(iter_line.c_str(), nullptr, 10);
    response.status = status_from_verdict(response.verdict);
    std::string xline, yline, rcline, bcline, brline, certline;
    if (!take(xline) || !take(yline) || !take(rcline) || !take(bcline) || !take(brline) || !take(certline)) {
        response.error = "truncated solution from embedded Python";
        return false;
    }
    bool x_ok = false, y_ok = false, rc_ok = false, bc_ok = false, br_ok = false;
    if (!parse_vec_line(xline, "X", response.x, x_ok, response.error)) return false;
    if (!parse_vec_line(yline, "Y", response.y, y_ok, response.error)) return false;
    if (!parse_vec_line(rcline, "RC", response.reduced, rc_ok, response.error)) return false;
    if (!parse_basis_line(bcline, "BC", response.column_basis, bc_ok, response.error)) return false;
    if (!parse_basis_line(brline, "BR", response.row_basis, br_ok, response.error)) return false;
    response.has_x = x_ok;
    response.has_y = y_ok;
    response.has_reduced = rc_ok;
    response.has_basis = bc_ok && br_ok;
    const std::string prefix = "CERT ";
    if (certline.compare(0, prefix.size(), prefix) != 0) {
        response.error = "certificate missing from embedded Python";
        return false;
    }
    response.certificate = certline.substr(prefix.size());
    return true;
}

}  // namespace

int python_solve(const PythonRequest& request, PythonResponse& response) {
    std::lock_guard<std::mutex> lock(g_py_mu);
    if (!load_api()) {
        response.error = g_load_error;
        return QENIVO_ERR_PYTHON;
    }
    if (!g_api.is_initialized()) g_api.initialize_ex(0);
    const int gil = g_api.gil_ensure();
    int code = QENIVO_OK;
    if (!g_script_ready) {
        if (g_api.run_string(SCRIPT) != 0) {
            response.error = python_exception_text();
            code = QENIVO_ERR_PYTHON;
        } else {
            g_script_ready = true;
        }
    }
    if (code == 0) {
        g_api.err_clear();
        PyObject* main = g_api.add_module("__main__");
        PyObject* dict = main ? g_api.module_dict(main) : nullptr;
        PyObject* fn = dict ? g_api.dict_get_string(dict, "capi_entry") : nullptr;
        if (!fn) {
            response.error = "embedded Python did not define capi_entry";
            code = QENIVO_ERR_PYTHON;
        } else {
            const std::string payload = build_payload(request);
            PyObject* text = g_api.unicode_from_string(payload.c_str());
            PyObject* args = g_api.tuple_new(1);
            if (!text || !args || g_api.tuple_set(args, 0, text) != 0) {
                if (text && !args) g_api.decref(text);
                if (args) g_api.decref(args);
                response.error = "could not build the Python call";
                code = QENIVO_ERR_PYTHON;
            } else {
                PyObject* result = g_api.call_object(fn, args);
                g_api.decref(args);
                if (!result) {
                    response.error = python_exception_text();
                    code = QENIVO_ERR_PYTHON;
                } else {
                    const char* utf = g_api.unicode_as_utf8(result);
                    std::string reply = utf ? utf : "";
                    g_api.decref(result);
                    if (utf == nullptr) {
                        response.error = "embedded Python returned a non-string";
                        code = QENIVO_ERR_PYTHON;
                    } else if (!parse_reply(reply, response)) {
                        code = QENIVO_ERR_PYTHON;
                    }
                }
            }
        }
    }
    g_api.gil_release(gil);
    if (const char* debug = std::getenv("QENIVO_CAPI_DEBUG")) {
        if (debug[0] == '1' && !response.error.empty()) {
            std::fprintf(stderr, "qenivo python bridge: %s\n", response.error.c_str());
        }
    }
    return code;
}
