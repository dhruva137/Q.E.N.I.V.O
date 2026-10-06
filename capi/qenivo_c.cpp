/* qenivo_c.cpp — C ABI implementation.
 *
 * Simplex, native, and native-simplex call nr_simplex in native/simplex_core.cpp
 * (Maros; Vanderbei). The revised simplex itself is not reimplemented here.
 * Every other engine is handed to python_bridge.cpp, which embeds Python.
 * The ABI, the CSR to CSC conversion, the KKT residuals on the simplex answer,
 * and the certificate JSON are original.
 */

#include "qenivo.h"

#include "python_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern "C" int nr_simplex(int m, int n, const int* Ap, const int* Ai, const double* Ax, const double* c,
                          const double* lx, const double* ux, const double* lc, const double* uc, double tol,
                          double time_limit, long max_iter, int* status, double* x_out, double* y_out,
                          long* iters_out, int* refactors_out);

static_assert(sizeof(int) == sizeof(int32_t), "the native simplex uses 32-bit int indices");

namespace {

struct Csr {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> ptr;
    std::vector<int32_t> idx;
    std::vector<double> val;
};

struct Residuals {
    double primal = 0;
    double dual = 0;
    double gap = 0;
    double primal_obj = 0;
    double dual_obj = 0;
    double max_rel = 0;
};

}  // namespace

struct qenivo_model {
    mutable std::mutex mu;
    int32_t rows = 0;
    int32_t cols = 0;
    Csr A;
    Csr Q;
    bool has_q = false;
    bool has_integer = false;
    std::vector<double> cost, row_lower, row_upper, col_lower, col_upper;
    std::vector<int32_t> integer;
    std::string name = "capi";
    double tolerance = 1e-8;
    double time_limit = 3600.0;
    qenivo_progress_fn progress = nullptr;
    void* progress_user = nullptr;

    bool solved = false;
    int32_t status = -1;
    std::string status_name;
    bool has_objective = false;
    double objective = 0;
    int64_t iterations = 0;
    bool has_x = false;
    bool has_y = false;
    bool has_reduced = false;
    bool has_basis = false;
    std::vector<double> x, y, reduced;
    std::vector<int32_t> column_basis, row_basis;
    std::string certificate;
    std::string impl;
    int32_t last_error = QENIVO_OK;
    std::string error;
};

namespace {

thread_local std::string g_create_error;

const char* error_text(int32_t code) {
    switch (code) {
    case QENIVO_OK: return "ok";
    case QENIVO_ERR_NULL: return "null";
    case QENIVO_ERR_ARGUMENT: return "argument";
    case QENIVO_ERR_STATE: return "state";
    case QENIVO_ERR_ENGINE: return "engine";
    case QENIVO_ERR_SOLVE: return "solve";
    case QENIVO_ERR_PYTHON: return "python";
    case QENIVO_ERR_NOMEM: return "nomem";
    case QENIVO_ERR_INTERNAL: return "internal";
    default: return "unknown";
    }
}

const char* status_text(int32_t code) {
    switch (code) {
    case QENIVO_OPTIMAL: return "optimal";
    case QENIVO_INFEASIBLE: return "infeasible";
    case QENIVO_UNBOUNDED: return "unbounded";
    case QENIVO_ITERATION_LIMIT: return "iteration_limit";
    case QENIVO_TIME_LIMIT: return "time_limit";
    case QENIVO_NUMERICAL: return "numerical_error";
    case QENIVO_NOT_PROVEN: return "not_proven";
    default: return "unknown";
    }
}

const char* basis_text(int32_t code) {
    switch (code) {
    case QENIVO_BASIC: return "basic";
    case QENIVO_AT_LOWER: return "at_lower";
    case QENIVO_AT_UPPER: return "at_upper";
    case QENIVO_NONBASIC_FREE: return "free";
    default: return "unknown";
    }
}

bool token_ok(const char* text) {
    if (text == nullptr || text[0] == '\0') return false;
    size_t n = std::strlen(text);
    if (n > 64) return false;
    for (size_t i = 0; i < n; ++i) {
        char c = text[i];
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                  c == '-';
        if (!ok) return false;
    }
    return true;
}

bool is_simplex_name(const char* engine) {
    return std::strcmp(engine, "simplex") == 0 || std::strcmp(engine, "native") == 0 ||
           std::strcmp(engine, "native-simplex") == 0;
}

void create_fail(const char* msg) { g_create_error = msg; }

bool finite_cost(const double* cost, int32_t n, const char** why) {
    if (n == 0) return true;
    if (cost == nullptr) {
        *why = "cost is null";
        return false;
    }
    for (int32_t j = 0; j < n; ++j) {
        if (!std::isfinite(cost[j])) {
            *why = "cost is not finite";
            return false;
        }
    }
    return true;
}

bool bounds_ok(const double* lo, const double* hi, int32_t n, const char* what, const char** why) {
    if (n == 0) return true;
    if (lo == nullptr || hi == nullptr) {
        *why = what;
        return false;
    }
    for (int32_t i = 0; i < n; ++i) {
        if (std::isnan(lo[i]) || std::isnan(hi[i])) {
            *why = "a bound is NaN";
            return false;
        }
        if (std::isfinite(lo[i]) && std::isfinite(hi[i]) && lo[i] > hi[i]) {
            *why = "a lower bound exceeds its upper bound";
            return false;
        }
    }
    return true;
}

bool copy_csr(int32_t rows, int32_t cols, const int32_t* ptr, const int32_t* idx, const double* val, Csr& out,
              const char** why) {
    if (rows < 0 || cols < 0) {
        *why = "negative dimension";
        return false;
    }
    if (ptr == nullptr || ptr[0] != 0) {
        *why = "CSR row pointer must be non-null and start at 0";
        return false;
    }
    for (int32_t i = 0; i < rows; ++i) {
        if (ptr[i + 1] < ptr[i]) {
            *why = "CSR row pointer is not nondecreasing";
            return false;
        }
    }
    const int32_t nnz = ptr[rows];
    if (nnz < 0) {
        *why = "negative CSR length";
        return false;
    }
    if (nnz > 0 && (idx == nullptr || val == nullptr)) {
        *why = "CSR index or value array is null";
        return false;
    }
    for (int32_t k = 0; k < nnz; ++k) {
        if (idx[k] < 0 || idx[k] >= cols) {
            *why = "CSR column index out of range";
            return false;
        }
        if (!std::isfinite(val[k])) {
            *why = "CSR value is not finite";
            return false;
        }
    }
    out.rows = rows;
    out.cols = cols;
    out.ptr.assign(ptr, ptr + rows + 1);
    out.idx.assign(idx, idx + nnz);
    out.val.assign(val, val + nnz);
    return true;
}

int32_t fail(qenivo_model& model, int32_t code, const std::string& msg) {
    model.last_error = code;
    model.error = msg;
    model.solved = false;
    return code;
}

void notify(qenivo_model& model, int64_t iteration, double objective, int32_t status) {
    if (model.progress != nullptr) model.progress(model.progress_user, iteration, objective, status);
}

void json_number(std::ostream& out, double v) {
    if (!std::isfinite(v)) {
        out << "null";
        return;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.16g", v);
    out << buf;
}

void json_escape(std::ostream& out, const std::string& text) {
    out << '"';
    for (char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    out << '"';
}

Residuals residuals_of(qenivo_model& model) {
    const int32_t m = model.rows;
    const int32_t n = model.cols;
    std::vector<double> ax(static_cast<size_t>(m), 0.0);
    std::vector<double> aty(static_cast<size_t>(n), 0.0);
    for (int32_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int32_t k = model.A.ptr[static_cast<size_t>(i)]; k < model.A.ptr[static_cast<size_t>(i) + 1]; ++k) {
            sum += model.A.val[static_cast<size_t>(k)] * model.x[static_cast<size_t>(model.A.idx[static_cast<size_t>(k)])];
        }
        ax[static_cast<size_t>(i)] = sum;
        if (!model.has_y) continue;
        const double yi = model.y[static_cast<size_t>(i)];
        for (int32_t k = model.A.ptr[static_cast<size_t>(i)]; k < model.A.ptr[static_cast<size_t>(i) + 1]; ++k) {
            aty[static_cast<size_t>(model.A.idx[static_cast<size_t>(k)])] += model.A.val[static_cast<size_t>(k)] * yi;
        }
    }
    model.reduced.assign(static_cast<size_t>(n), 0.0);
    double primal2 = 0.0;
    double dual2 = 0.0;
    double pobj = 0.0;
    double dobj = 0.0;
    double b2 = 0.0;
    double c2 = 0.0;
    for (int32_t i = 0; i < m; ++i) {
        const double lo = model.row_lower[static_cast<size_t>(i)];
        const double hi = model.row_upper[static_cast<size_t>(i)];
        double clipped = ax[static_cast<size_t>(i)];
        if (clipped < lo) clipped = lo;
        if (clipped > hi) clipped = hi;
        const double d = ax[static_cast<size_t>(i)] - clipped;
        primal2 += d * d;
        const double y = model.has_y ? model.y[static_cast<size_t>(i)] : 0.0;
        const double yp = y > 0.0 ? y : 0.0;
        const double yn = y < 0.0 ? -y : 0.0;
        double rd = 0.0;
        if (!std::isfinite(lo)) rd += yp;
        if (!std::isfinite(hi)) rd -= yn;
        dual2 += rd * rd;
        dobj += yp * (std::isfinite(lo) ? lo : 0.0) - yn * (std::isfinite(hi) ? hi : 0.0);
        const double scale = std::max(std::isfinite(lo) ? std::fabs(lo) : 0.0, std::isfinite(hi) ? std::fabs(hi) : 0.0);
        b2 += scale * scale;
    }
    for (int32_t j = 0; j < n; ++j) {
        const double lo = model.col_lower[static_cast<size_t>(j)];
        const double hi = model.col_upper[static_cast<size_t>(j)];
        double clipped = model.x[static_cast<size_t>(j)];
        if (clipped < lo) clipped = lo;
        if (clipped > hi) clipped = hi;
        const double d = model.x[static_cast<size_t>(j)] - clipped;
        primal2 += d * d;
        const double lam = model.cost[static_cast<size_t>(j)] - aty[static_cast<size_t>(j)];
        model.reduced[static_cast<size_t>(j)] = lam;
        const double lp = lam > 0.0 ? lam : 0.0;
        const double ln = lam < 0.0 ? -lam : 0.0;
        double rd = 0.0;
        if (!std::isfinite(lo)) rd += lp;
        if (!std::isfinite(hi)) rd -= ln;
        dual2 += rd * rd;
        dobj += lp * (std::isfinite(lo) ? lo : 0.0) - ln * (std::isfinite(hi) ? hi : 0.0);
        pobj += model.cost[static_cast<size_t>(j)] * model.x[static_cast<size_t>(j)];
        c2 += model.cost[static_cast<size_t>(j)] * model.cost[static_cast<size_t>(j)];
    }
    Residuals r;
    r.primal = std::sqrt(primal2);
    r.dual = std::sqrt(dual2);
    r.gap = std::fabs(pobj - dobj);
    r.primal_obj = pobj;
    r.dual_obj = dobj;
    const double rel_p = r.primal / (1.0 + std::sqrt(b2));
    const double rel_d = r.dual / (1.0 + std::sqrt(c2));
    const double rel_g = r.gap / (1.0 + std::fabs(pobj) + std::fabs(dobj));
    r.max_rel = std::max(rel_p, std::max(rel_d, rel_g));
    return r;
}

std::string simplex_certificate(const qenivo_model& model, const Residuals& r, int refactors) {
    std::ostringstream out;
    out << "{\"schema\":\"qenivo.capi.certificate/1\",\"abi\":" << QENIVO_ABI_VERSION
        << ",\"impl\":\"native simplex\",\"engine\":\"simplex\",\"name\":";
    json_escape(out, model.name);
    out << ",\"status\":";
    json_escape(out, model.status_name);
    out << ",\"objective\":";
    json_number(out, model.objective);
    out << ",\"iterations\":" << model.iterations << ",\"refactorizations\":" << refactors
        << ",\"residuals\":{\"primal\":";
    json_number(out, r.primal);
    out << ",\"dual\":";
    json_number(out, r.dual);
    out << ",\"gap\":";
    json_number(out, r.gap);
    out << ",\"max_rel\":";
    json_number(out, r.max_rel);
    out << ",\"dual_objective\":";
    json_number(out, r.dual_obj);
    out << "},\"x\":[";
    for (int32_t j = 0; j < model.cols; ++j) {
        if (j) out << ',';
        json_number(out, model.x[static_cast<size_t>(j)]);
    }
    out << "],\"y\":[";
    for (int32_t i = 0; i < model.rows; ++i) {
        if (i) out << ',';
        json_number(out, model.y[static_cast<size_t>(i)]);
    }
    out << "],\"reduced_costs\":[";
    for (int32_t j = 0; j < model.cols; ++j) {
        if (j) out << ',';
        json_number(out, model.reduced[static_cast<size_t>(j)]);
    }
    out << "],\"basis\":{\"columns\":[";
    for (int32_t j = 0; j < model.cols; ++j) {
        if (j) out << ',';
        json_escape(out, basis_text(model.column_basis[static_cast<size_t>(j)]));
    }
    out << "],\"rows\":[";
    for (int32_t i = 0; i < model.rows; ++i) {
        if (i) out << ',';
        json_escape(out, basis_text(model.row_basis[static_cast<size_t>(i)]));
    }
    out << "]}}";
    return out.str();
}

void csr_to_csc(const Csr& A, std::vector<int>& colptr, std::vector<int>& rowidx, std::vector<double>& values) {
    const int n = A.cols;
    const int nnz = A.ptr.empty() ? 0 : A.ptr.back();
    colptr.assign(static_cast<size_t>(n) + 1, 0);
    for (int k = 0; k < nnz; ++k) colptr[static_cast<size_t>(A.idx[static_cast<size_t>(k)]) + 1] += 1;
    for (int j = 0; j < n; ++j) colptr[static_cast<size_t>(j) + 1] += colptr[static_cast<size_t>(j)];
    rowidx.resize(static_cast<size_t>(nnz));
    values.resize(static_cast<size_t>(nnz));
    std::vector<int> next = colptr;
    for (int i = 0; i < A.rows; ++i) {
        for (int k = A.ptr[static_cast<size_t>(i)]; k < A.ptr[static_cast<size_t>(i) + 1]; ++k) {
            const int dest = next[static_cast<size_t>(A.idx[static_cast<size_t>(k)])]++;
            rowidx[static_cast<size_t>(dest)] = i;
            values[static_cast<size_t>(dest)] = A.val[static_cast<size_t>(k)];
        }
    }
}

int32_t solve_simplex(qenivo_model& model) {
    if (model.has_integer) {
        return fail(model, QENIVO_ERR_ENGINE, "engine simplex does not enforce integrality; use milp");
    }
    if (model.has_q) {
        return fail(model, QENIVO_ERR_ENGINE, "engine simplex does not solve a quadratic term; use pdqp");
    }
    model.impl = "native simplex";
    if (model.cols == 0) {
        model.status = QENIVO_OPTIMAL;
        model.status_name = status_text(model.status);
        model.x.clear();
        model.y.assign(static_cast<size_t>(model.rows), 0.0);
        model.reduced.clear();
        model.column_basis.clear();
        model.row_basis.assign(static_cast<size_t>(model.rows), QENIVO_BASIC);
        model.has_x = model.has_y = model.has_reduced = model.has_basis = true;
        model.has_objective = true;
        model.objective = 0.0;
        model.iterations = 0;
        model.solved = true;
        Residuals empty;
        model.certificate = simplex_certificate(model, empty, 0);
        model.last_error = QENIVO_OK;
        model.error.clear();
        return QENIVO_OK;
    }
    std::vector<int> colptr, rowidx, basis(static_cast<size_t>(model.cols + model.rows), -1);
    std::vector<double> values, x(static_cast<size_t>(model.cols), 0.0);
    std::vector<double> y(static_cast<size_t>(std::max(model.rows, static_cast<int32_t>(1))), 0.0);
    csr_to_csc(model.A, colptr, rowidx, values);
    long iters = 0;
    int refactors = 0;
    const long max_iter = 20000L + 50L * (static_cast<long>(model.rows) + static_cast<long>(model.cols));
    static const int k_zero_i = 0;
    static const double k_zero_d = 0.0;
    const int* ai = rowidx.empty() ? &k_zero_i : rowidx.data();
    const double* ax = values.empty() ? &k_zero_d : values.data();
    const double* lc = model.rows ? model.row_lower.data() : &k_zero_d;
    const double* uc = model.rows ? model.row_upper.data() : &k_zero_d;
    const int code = nr_simplex(model.rows, model.cols, colptr.data(), ai, ax, model.cost.data(),
                                model.col_lower.data(), model.col_upper.data(), lc, uc, model.tolerance,
                                model.time_limit, max_iter, basis.data(), x.data(), y.data(), &iters, &refactors);
    if (code < 0 || code > 5) return fail(model, QENIVO_ERR_SOLVE, "native simplex returned an unknown status");
    model.x = std::move(x);
    model.y.assign(y.begin(), y.begin() + model.rows);
    model.has_x = true;
    model.has_y = true;
    model.column_basis.assign(basis.begin(), basis.begin() + model.cols);
    model.row_basis.assign(basis.begin() + model.cols, basis.end());
    model.has_basis = true;
    model.iterations = iters;
    model.objective = 0.0;
    for (int32_t j = 0; j < model.cols; ++j) model.objective += model.cost[static_cast<size_t>(j)] * model.x[static_cast<size_t>(j)];
    model.has_objective = true;
    Residuals r = residuals_of(model);
    model.has_reduced = true;
    int32_t status = code;
    if (status == QENIVO_OPTIMAL && !(r.max_rel <= 1e-5)) status = QENIVO_NUMERICAL;
    model.status = status;
    model.status_name = status_text(status);
    model.certificate = simplex_certificate(model, r, refactors);
    model.solved = true;
    model.last_error = QENIVO_OK;
    model.error.clear();
    return QENIVO_OK;
}

int32_t solve_python(qenivo_model& model, const char* engine) {
    PythonRequest req;
    req.rows = model.rows;
    req.cols = model.cols;
    req.a_ptr = model.A.ptr;
    req.a_idx = model.A.idx;
    req.a_val = model.A.val;
    req.cost = model.cost;
    req.row_lower = model.row_lower;
    req.row_upper = model.row_upper;
    req.col_lower = model.col_lower;
    req.col_upper = model.col_upper;
    req.has_integer = model.has_integer;
    req.integer = model.integer;
    req.has_q = model.has_q;
    req.q_ptr = model.Q.ptr;
    req.q_idx = model.Q.idx;
    req.q_val = model.Q.val;
    req.engine = engine;
    req.tolerance = model.tolerance;
    req.time_limit = model.time_limit;
    req.name = model.name;
    PythonResponse response;
    const int rc = python_solve(req, response);
    if (rc != QENIVO_OK) return fail(model, rc, response.error.empty() ? "embedded Python failed" : response.error);
    if (response.has_x && static_cast<int32_t>(response.x.size()) != model.cols) {
        return fail(model, QENIVO_ERR_SOLVE, "embedded Python returned the wrong number of primal values");
    }
    if (response.has_y && static_cast<int32_t>(response.y.size()) != model.rows) {
        return fail(model, QENIVO_ERR_SOLVE, "embedded Python returned the wrong number of duals");
    }
    model.impl = "embedded python";
    model.status = response.status;
    model.status_name = response.verdict;
    model.has_objective = response.has_objective;
    model.objective = response.objective;
    model.iterations = response.iterations;
    model.has_x = response.has_x;
    model.x = std::move(response.x);
    model.has_y = response.has_y;
    model.y = std::move(response.y);
    model.has_reduced = response.has_reduced;
    model.reduced = std::move(response.reduced);
    model.has_basis = response.has_basis && static_cast<int32_t>(response.column_basis.size()) == model.cols &&
                      static_cast<int32_t>(response.row_basis.size()) == model.rows;
    if (model.has_basis) {
        model.column_basis = std::move(response.column_basis);
        model.row_basis = std::move(response.row_basis);
    }
    model.certificate = std::move(response.certificate);
    model.solved = true;
    model.last_error = QENIVO_OK;
    model.error.clear();
    return QENIVO_OK;
}

template <class T>
int32_t copy_out(const qenivo_model& model, bool ready, const std::vector<T>& src, T* dest, int32_t n) {
    if (!model.solved) return QENIVO_ERR_STATE;
    if (!ready) return QENIVO_ERR_STATE;
    if (n != static_cast<int32_t>(src.size())) return QENIVO_ERR_ARGUMENT;
    if (n > 0 && dest == nullptr) return QENIVO_ERR_NULL;
    if (n > 0) std::memcpy(dest, src.data(), sizeof(T) * static_cast<size_t>(n));
    return QENIVO_OK;
}

}  // namespace

extern "C" {

int32_t qenivo_abi_version(void) { return QENIVO_ABI_VERSION; }

const char* qenivo_error_name(int32_t code) { return error_text(code); }

const char* qenivo_create_error_message(void) { return g_create_error.c_str(); }

qenivo_model* qenivo_model_create(int32_t rows, int32_t cols, const int32_t* a_rowptr, const int32_t* a_colidx,
                                  const double* a_values, const double* cost, const double* row_lower,
                                  const double* row_upper, const double* col_lower, const double* col_upper,
                                  const int32_t* integrality, const int32_t* q_rowptr, const int32_t* q_colidx,
                                  const double* q_values, int32_t* error) {
    g_create_error.clear();
    auto set_error = [&](int32_t code) -> qenivo_model* {
        if (error) *error = code;
        return nullptr;
    };
    try {
        const char* why = "invalid model";
        if (!finite_cost(cost, cols, &why) || !bounds_ok(row_lower, row_upper, rows, "row bounds are null", &why) ||
            !bounds_ok(col_lower, col_upper, cols, "column bounds are null", &why)) {
            create_fail(why);
            return set_error(QENIVO_ERR_ARGUMENT);
        }
        Csr A;
        if (!copy_csr(rows, cols, a_rowptr, a_colidx, a_values, A, &why)) {
            create_fail(why);
            return set_error(QENIVO_ERR_ARGUMENT);
        }
        Csr Q;
        bool has_q = q_rowptr != nullptr;
        if (has_q) {
            if (!copy_csr(cols, cols, q_rowptr, q_colidx, q_values, Q, &why)) {
                create_fail(why);
                return set_error(QENIVO_ERR_ARGUMENT);
            }
            has_q = Q.ptr.back() > 0;
        }
        bool has_integer = false;
        std::vector<int32_t> integer;
        if (integrality != nullptr && cols > 0) {
            integer.assign(integrality, integrality + cols);
            for (int32_t v : integer) {
                if (v) has_integer = true;
            }
        }
        auto* model = new qenivo_model();
        model->rows = rows;
        model->cols = cols;
        model->A = std::move(A);
        model->Q = std::move(Q);
        model->has_q = has_q;
        model->has_integer = has_integer;
        model->integer = std::move(integer);
        if (cols) model->cost.assign(cost, cost + cols);
        if (rows) {
            model->row_lower.assign(row_lower, row_lower + rows);
            model->row_upper.assign(row_upper, row_upper + rows);
        }
        if (cols) {
            model->col_lower.assign(col_lower, col_lower + cols);
            model->col_upper.assign(col_upper, col_upper + cols);
        }
        if (error) *error = QENIVO_OK;
        return model;
    } catch (const std::bad_alloc&) {
        create_fail("out of memory");
        return set_error(QENIVO_ERR_NOMEM);
    } catch (const std::exception& ex) {
        create_fail(ex.what());
        return set_error(QENIVO_ERR_INTERNAL);
    } catch (...) {
        create_fail("unknown exception");
        return set_error(QENIVO_ERR_INTERNAL);
    }
}

void qenivo_model_free(qenivo_model* model) { delete model; }

int32_t qenivo_set_name(qenivo_model* model, const char* name) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    if (!token_ok(name)) return QENIVO_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(model->mu);
    model->name = name;
    return QENIVO_OK;
}

int32_t qenivo_set_tolerance(qenivo_model* model, double absolute) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    if (!(absolute > 0.0) || !std::isfinite(absolute)) return QENIVO_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(model->mu);
    model->tolerance = absolute;
    return QENIVO_OK;
}

int32_t qenivo_set_time_limit(qenivo_model* model, double seconds) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    if (!(seconds > 0.0) || !std::isfinite(seconds)) return QENIVO_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(model->mu);
    model->time_limit = seconds;
    return QENIVO_OK;
}

int32_t qenivo_set_progress(qenivo_model* model, qenivo_progress_fn fn, void* user) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    model->progress = fn;
    model->progress_user = user;
    return QENIVO_OK;
}

int32_t qenivo_model_rows(const qenivo_model* model) {
    if (model == nullptr) return -1;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->rows;
}

int32_t qenivo_model_cols(const qenivo_model* model) {
    if (model == nullptr) return -1;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->cols;
}

int32_t qenivo_solve(qenivo_model* model, const char* engine) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    if (!token_ok(engine)) return QENIVO_ERR_ARGUMENT;
    try {
        std::lock_guard<std::mutex> lock(model->mu);
        model->solved = false;
        model->certificate.clear();
        notify(*model, 0, std::numeric_limits<double>::quiet_NaN(), -1);
        const int32_t rc = is_simplex_name(engine) ? solve_simplex(*model) : solve_python(*model, engine);
        if (rc == QENIVO_OK) {
            const double obj = model->has_objective ? model->objective : std::numeric_limits<double>::quiet_NaN();
            notify(*model, model->iterations, obj, model->status);
        }
        return rc;
    } catch (const std::bad_alloc&) {
        return fail(*model, QENIVO_ERR_NOMEM, "out of memory");
    } catch (const std::exception& ex) {
        return fail(*model, QENIVO_ERR_INTERNAL, ex.what());
    } catch (...) {
        return fail(*model, QENIVO_ERR_INTERNAL, "unknown exception");
    }
}

int32_t qenivo_last_error(const qenivo_model* model) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->last_error;
}

const char* qenivo_last_error_message(const qenivo_model* model) {
    if (model == nullptr) return "null handle";
    std::lock_guard<std::mutex> lock(model->mu);
    return model->error.c_str();
}

int32_t qenivo_result_status(const qenivo_model* model) {
    if (model == nullptr) return -1;
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved) return -1;
    return model->status;
}

const char* qenivo_result_status_name(const qenivo_model* model) {
    if (model == nullptr) return "null";
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved) return "unsolved";
    return model->status_name.c_str();
}

int32_t qenivo_result_objective(const qenivo_model* model, double* objective) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    if (objective == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved || !model->has_objective) return QENIVO_ERR_STATE;
    *objective = model->objective;
    return QENIVO_OK;
}

int64_t qenivo_result_iterations(const qenivo_model* model) {
    if (model == nullptr) return -1;
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved) return -1;
    return model->iterations;
}

int32_t qenivo_result_has_duals(const qenivo_model* model) {
    if (model == nullptr) return 0;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->solved && model->has_y ? 1 : 0;
}

int32_t qenivo_result_has_basis(const qenivo_model* model) {
    if (model == nullptr) return 0;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->solved && model->has_basis ? 1 : 0;
}

int32_t qenivo_result_has_reduced_costs(const qenivo_model* model) {
    if (model == nullptr) return 0;
    std::lock_guard<std::mutex> lock(model->mu);
    return model->solved && model->has_reduced ? 1 : 0;
}

int32_t qenivo_result_x(const qenivo_model* model, double* x, int32_t cols) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    return copy_out(*model, model->has_x, model->x, x, cols);
}

int32_t qenivo_result_y(const qenivo_model* model, double* y, int32_t rows) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    return copy_out(*model, model->has_y, model->y, y, rows);
}

int32_t qenivo_result_reduced_costs(const qenivo_model* model, double* reduced, int32_t cols) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    return copy_out(*model, model->has_reduced, model->reduced, reduced, cols);
}

int32_t qenivo_result_basis(const qenivo_model* model, int32_t* column_status, int32_t cols, int32_t* row_status,
                           int32_t rows) {
    if (model == nullptr) return QENIVO_ERR_NULL;
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved || !model->has_basis) return QENIVO_ERR_STATE;
    const int32_t col_rc = copy_out(*model, true, model->column_basis, column_status, cols);
    if (col_rc != QENIVO_OK) return col_rc;
    return copy_out(*model, true, model->row_basis, row_status, rows);
}

const char* qenivo_result_certificate_json(const qenivo_model* model) {
    if (model == nullptr) return "";
    std::lock_guard<std::mutex> lock(model->mu);
    if (!model->solved) return "";
    return model->certificate.c_str();
}

}  // extern "C"
