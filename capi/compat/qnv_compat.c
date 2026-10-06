/* qnv_compat.c - QNV environment/problem interface over the QENIVO C ABI (see qnv_compat.h).
 *
 * The problem object keeps the model in the ABI's form (CSR rows, minimisation cost, row and
 * column intervals) and creates a qenivo_model only for the duration of a solve, so senses,
 * types and objective sense can change between solves. Maximisation is solved as the negated
 * minimisation; objective, duals and reduced costs are turned back to the user's sense.
 * Original code.
 */
#include "qnv_compat.h"

#include "qenivo.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct qnv_env {
    double time_limit;
    double tolerance;
    int threads;
};

struct qnv_lp {
    char name[128];
    int rows, cols, sense;          /* sense: QNV_MIN or QNV_MAX */
    int* rowptr;                    /* CSR, rows + 1 */
    int* colidx;
    double* val;
    double* obj;                    /* user's objective (not negated) */
    double* lc;
    double* uc;
    double* lx;
    double* ux;
    int* integer;                   /* NULL when every column is continuous */
    /* last solution, user's sense */
    int stat;
    int has_x, has_y, has_dj;
    double objval;
    double* x;
    double* y;
    double* dj;
    char* cert;
    char engine[64];
};

static double clip_inf(double v) {
    if (v >= QNV_INFBOUND) return INFINITY;
    if (v <= -QNV_INFBOUND) return -INFINITY;
    return v;
}

static void drop_solution(qnv_lp* lp) {
    free(lp->x);
    free(lp->y);
    free(lp->dj);
    free(lp->cert);
    lp->x = lp->y = lp->dj = NULL;
    lp->cert = NULL;
    lp->has_x = lp->has_y = lp->has_dj = 0;
    lp->stat = QNV_STAT_NONE;
    lp->objval = NAN;
    lp->engine[0] = '\0';
}

static void drop_model(qnv_lp* lp) {
    free(lp->rowptr);
    free(lp->colidx);
    free(lp->val);
    free(lp->obj);
    free(lp->lc);
    free(lp->uc);
    free(lp->lx);
    free(lp->ux);
    free(lp->integer);
    lp->rowptr = lp->colidx = lp->integer = NULL;
    lp->val = lp->obj = lp->lc = lp->uc = lp->lx = lp->ux = NULL;
    lp->rows = lp->cols = 0;
}

const char* QNVerrorstring(int code) {
    switch (code) {
    case QNV_OK: return "ok";
    case QNV_ERR_NULL: return "null environment, problem or required pointer";
    case QNV_ERR_ARGUMENT: return "invalid argument";
    case QNV_ERR_NO_PROBLEM: return "no problem data has been copied";
    case QNV_ERR_NO_SOLUTION: return "no solution is available";
    case QNV_ERR_UNSUPPORTED: return "not available in QENIVO through the C interface";
    case QNV_ERR_NOMEM: return "out of memory";
    case QNV_ERR_ENGINE: return "the QENIVO engine reported an error";
    default: return "unknown error code";
    }
}

const char* QNVgetstatstring(int stat) {
    switch (stat) {
    case QNV_STAT_NONE: return "no solution";
    case QNV_STAT_OPTIMAL: return "optimal (certified)";
    case QNV_STAT_INFEASIBLE: return "infeasible (certified)";
    case QNV_STAT_UNBOUNDED: return "unbounded (certified)";
    case QNV_STAT_NOT_PROVEN: return "not proven";
    case QNV_STAT_TIME_LIMIT: return "time limit, not proven";
    case QNV_STAT_ITERATION_LIMIT: return "iteration limit, not proven";
    default: return "unknown status";
    }
}

qnv_env* QNVopenenv(int* status) {
    qnv_env* env = (qnv_env*)calloc(1, sizeof(qnv_env));
    if (status) *status = env ? QNV_OK : QNV_ERR_NOMEM;
    if (!env) return NULL;
    env->time_limit = 3600.0;
    env->tolerance = 1e-6;
    env->threads = 0;
    return env;
}

int QNVcloseenv(qnv_env** env) {
    if (!env || !*env) return QNV_ERR_NULL;
    free(*env);
    *env = NULL;
    return QNV_OK;
}

int QNVsetdblparam(qnv_env* env, int which, double value) {
    if (!env) return QNV_ERR_NULL;
    if (which == QNV_PARAM_MIPGAP) return QNV_ERR_UNSUPPORTED;
    if (!(value > 0.0) || !isfinite(value)) return QNV_ERR_ARGUMENT;
    if (which == QNV_PARAM_TIMELIMIT) env->time_limit = value;
    else if (which == QNV_PARAM_TOLERANCE) env->tolerance = value;
    else return QNV_ERR_UNSUPPORTED;
    return QNV_OK;
}

int QNVgetdblparam(qnv_env* env, int which, double* value) {
    if (!env || !value) return QNV_ERR_NULL;
    if (which == QNV_PARAM_TIMELIMIT) *value = env->time_limit;
    else if (which == QNV_PARAM_TOLERANCE) *value = env->tolerance;
    else return QNV_ERR_UNSUPPORTED;
    return QNV_OK;
}

int QNVsetintparam(qnv_env* env, int which, int value) {
    if (!env) return QNV_ERR_NULL;
    if (which != QNV_PARAM_THREADS) return QNV_ERR_UNSUPPORTED;
    if (value != 0 && value != 1) return QNV_ERR_UNSUPPORTED;
    env->threads = value;
    return QNV_OK;
}

int QNVgetintparam(qnv_env* env, int which, int* value) {
    if (!env || !value) return QNV_ERR_NULL;
    if (which != QNV_PARAM_THREADS) return QNV_ERR_UNSUPPORTED;
    *value = env->threads;
    return QNV_OK;
}

qnv_lp* QNVcreateprob(qnv_env* env, int* status, const char* name) {
    if (!env) {
        if (status) *status = QNV_ERR_NULL;
        return NULL;
    }
    qnv_lp* lp = (qnv_lp*)calloc(1, sizeof(qnv_lp));
    if (status) *status = lp ? QNV_OK : QNV_ERR_NOMEM;
    if (!lp) return NULL;
    lp->sense = QNV_MIN;
    lp->rows = lp->cols = -1;
    lp->objval = NAN;
    strncpy(lp->name, (name && name[0]) ? name : "qnv", sizeof(lp->name) - 1);
    for (char* p = lp->name; *p; ++p) {            /* the ABI takes the name as one word */
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') *p = '_';
    }
    return lp;
}

int QNVfreeprob(qnv_env* env, qnv_lp** lp) {
    if (!env || !lp || !*lp) return QNV_ERR_NULL;
    drop_solution(*lp);
    drop_model(*lp);
    free(*lp);
    *lp = NULL;
    return QNV_OK;
}

int QNVcopylp(qnv_env* env, qnv_lp* lp, int numcols, int numrows, int objsense, const double* obj,
              const double* rhs, const char* sense, const int* colstart, const int* colcount,
              const int* rowindex, const double* value, const double* lb, const double* ub,
              const double* rngval) {
    if (!env || !lp) return QNV_ERR_NULL;
    if (numcols < 0 || numrows < 0 || (objsense != QNV_MIN && objsense != QNV_MAX)) return QNV_ERR_ARGUMENT;
    if (numrows > 0 && !sense) return QNV_ERR_NULL;
    if (numcols > 0 && (!colstart || !colcount)) return QNV_ERR_NULL;
    long nnz = 0;
    for (int j = 0; j < numcols; ++j) {
        if (colcount[j] < 0 || colstart[j] < 0) return QNV_ERR_ARGUMENT;
        nnz += colcount[j];
        for (int k = colstart[j]; k < colstart[j] + colcount[j]; ++k) {
            if (!rowindex || !value) return QNV_ERR_NULL;
            if (rowindex[k] < 0 || rowindex[k] >= numrows || !isfinite(value[k])) return QNV_ERR_ARGUMENT;
        }
    }
    for (int i = 0; i < numrows; ++i) {
        const char s = sense[i];
        if (s != 'L' && s != 'G' && s != 'E' && s != 'R') return QNV_ERR_ARGUMENT;
        if (s == 'R' && !rngval) return QNV_ERR_NULL;
    }
    drop_solution(lp);
    drop_model(lp);
    const size_t m = (size_t)numrows;
    const size_t n = (size_t)numcols;
    lp->rowptr = (int*)calloc(m + 1, sizeof(int));
    lp->colidx = (int*)malloc((size_t)(nnz ? nnz : 1) * sizeof(int));
    lp->val = (double*)malloc((size_t)(nnz ? nnz : 1) * sizeof(double));
    lp->obj = (double*)calloc(n ? n : 1, sizeof(double));
    lp->lc = (double*)malloc((m ? m : 1) * sizeof(double));
    lp->uc = (double*)malloc((m ? m : 1) * sizeof(double));
    lp->lx = (double*)malloc((n ? n : 1) * sizeof(double));
    lp->ux = (double*)malloc((n ? n : 1) * sizeof(double));
    int* next = (int*)calloc(m + 1, sizeof(int));
    if (!lp->rowptr || !lp->colidx || !lp->val || !lp->obj || !lp->lc || !lp->uc || !lp->lx || !lp->ux || !next) {
        free(next);
        drop_model(lp);
        return QNV_ERR_NOMEM;
    }
    for (int j = 0; j < numcols; ++j) {
        for (int k = colstart[j]; k < colstart[j] + colcount[j]; ++k) lp->rowptr[rowindex[k] + 1] += 1;
    }
    for (size_t i = 0; i < m; ++i) lp->rowptr[i + 1] += lp->rowptr[i];
    memcpy(next, lp->rowptr, (m + 1) * sizeof(int));
    for (int j = 0; j < numcols; ++j) {
        for (int k = colstart[j]; k < colstart[j] + colcount[j]; ++k) {
            const int dest = next[rowindex[k]]++;
            lp->colidx[dest] = j;
            lp->val[dest] = value[k];
        }
    }
    free(next);
    for (size_t i = 0; i < m; ++i) {
        const double r = rhs ? clip_inf(rhs[i]) : 0.0;
        switch (sense[i]) {
        case 'L': lp->lc[i] = -INFINITY; lp->uc[i] = r; break;
        case 'G': lp->lc[i] = r; lp->uc[i] = INFINITY; break;
        case 'E': lp->lc[i] = r; lp->uc[i] = r; break;
        default: {
            const double w = rngval[i];
            lp->lc[i] = w >= 0 ? r : r + w;
            lp->uc[i] = w >= 0 ? r + w : r;
        }
        }
    }
    for (size_t j = 0; j < n; ++j) {
        lp->obj[j] = obj ? obj[j] : 0.0;
        lp->lx[j] = lb ? clip_inf(lb[j]) : 0.0;
        lp->ux[j] = ub ? clip_inf(ub[j]) : INFINITY;
    }
    lp->rows = numrows;
    lp->cols = numcols;
    lp->sense = objsense;
    return QNV_OK;
}

int QNVcopyctype(qnv_env* env, qnv_lp* lp, const char* ctype) {
    if (!env || !lp || !ctype) return QNV_ERR_NULL;
    if (lp->cols < 0) return QNV_ERR_NO_PROBLEM;
    for (int j = 0; j < lp->cols; ++j) {
        if (ctype[j] != 'C' && ctype[j] != 'I' && ctype[j] != 'B') return QNV_ERR_ARGUMENT;
    }
    free(lp->integer);
    lp->integer = (int*)calloc((size_t)(lp->cols ? lp->cols : 1), sizeof(int));
    if (!lp->integer) return QNV_ERR_NOMEM;
    int any = 0;
    for (int j = 0; j < lp->cols; ++j) {
        lp->integer[j] = ctype[j] != 'C';
        any |= lp->integer[j];
        if (ctype[j] == 'B') {
            if (lp->lx[j] < 0.0) lp->lx[j] = 0.0;
            if (lp->ux[j] > 1.0) lp->ux[j] = 1.0;
        }
    }
    if (!any) {
        free(lp->integer);
        lp->integer = NULL;
    }
    drop_solution(lp);
    return QNV_OK;
}

int QNVchgobjsen(qnv_env* env, qnv_lp* lp, int objsense) {
    if (!env || !lp) return QNV_ERR_NULL;
    if (objsense != QNV_MIN && objsense != QNV_MAX) return QNV_ERR_ARGUMENT;
    lp->sense = objsense;
    drop_solution(lp);
    return QNV_OK;
}

int QNVgetnumrows(qnv_env* env, qnv_lp* lp) { return (env && lp && lp->rows > 0) ? lp->rows : 0; }
int QNVgetnumcols(qnv_env* env, qnv_lp* lp) { return (env && lp && lp->cols > 0) ? lp->cols : 0; }

static int stat_from_abi(int32_t s) {
    switch (s) {
    case QENIVO_OPTIMAL: return QNV_STAT_OPTIMAL;
    case QENIVO_INFEASIBLE: return QNV_STAT_INFEASIBLE;
    case QENIVO_UNBOUNDED: return QNV_STAT_UNBOUNDED;
    case QENIVO_TIME_LIMIT: return QNV_STAT_TIME_LIMIT;
    case QENIVO_ITERATION_LIMIT: return QNV_STAT_ITERATION_LIMIT;
    default: return QNV_STAT_NOT_PROVEN;
    }
}

static char* dup_text(const char* s) {
    const size_t k = s ? strlen(s) : 0;
    char* out = (char*)malloc(k + 1);
    if (out) {
        if (k) memcpy(out, s, k);
        out[k] = '\0';
    }
    return out;
}

/* One solve through the ABI with `engine`; fills lp's solution in the user's sense. */
static int run_engine(qnv_env* env, qnv_lp* lp, const char* engine) {
    const size_t m = (size_t)lp->rows;
    const size_t n = (size_t)lp->cols;
    const double sg = (double)lp->sense;
    double* cost = (double*)malloc((n ? n : 1) * sizeof(double));
    if (!cost) return QNV_ERR_NOMEM;
    for (size_t j = 0; j < n; ++j) cost[j] = sg * lp->obj[j];
    int32_t err = 0;
    qenivo_model* mdl = qenivo_model_create(lp->rows, lp->cols, lp->rowptr, lp->colidx, lp->val, cost,
                                            m ? lp->lc : NULL, m ? lp->uc : NULL, lp->lx, lp->ux,
                                            lp->integer, NULL, NULL, NULL, &err);
    free(cost);
    if (!mdl) return err == QENIVO_ERR_NOMEM ? QNV_ERR_NOMEM : QNV_ERR_ARGUMENT;
    int rc = QNV_OK;
    if (qenivo_set_name(mdl, lp->name) != QENIVO_OK || qenivo_set_tolerance(mdl, env->tolerance) != QENIVO_OK ||
        qenivo_set_time_limit(mdl, env->time_limit) != QENIVO_OK || qenivo_solve(mdl, engine) != QENIVO_OK) {
        rc = QNV_ERR_ENGINE;
    }
    drop_solution(lp);
    if (rc == QNV_OK) {
        lp->stat = stat_from_abi(qenivo_result_status(mdl));
        double v = NAN;
        if (qenivo_result_objective(mdl, &v) == QENIVO_OK && lp->stat == QNV_STAT_OPTIMAL) lp->objval = sg * v;
        lp->x = (double*)malloc((n ? n : 1) * sizeof(double));
        lp->y = (double*)malloc((m ? m : 1) * sizeof(double));
        lp->dj = (double*)malloc((n ? n : 1) * sizeof(double));
        if (!lp->x || !lp->y || !lp->dj) {
            rc = QNV_ERR_NOMEM;
        } else {
            lp->has_x = qenivo_result_x(mdl, lp->x, lp->cols) == QENIVO_OK;
            lp->has_y = qenivo_result_has_duals(mdl) && qenivo_result_y(mdl, lp->y, lp->rows) == QENIVO_OK;
            lp->has_dj = qenivo_result_has_reduced_costs(mdl) &&
                         qenivo_result_reduced_costs(mdl, lp->dj, lp->cols) == QENIVO_OK;
            for (size_t i = 0; lp->has_y && i < m; ++i) lp->y[i] *= sg;
            for (size_t j = 0; lp->has_dj && j < n; ++j) lp->dj[j] *= sg;
        }
        lp->cert = dup_text(qenivo_result_certificate_json(mdl));
        strncpy(lp->engine, strcmp(engine, "simplex") == 0 ? "native simplex" : "embedded python",
                sizeof(lp->engine) - 1);
    }
    qenivo_model_free(mdl);
    return rc;
}

static int ready(qnv_env* env, qnv_lp* lp) {
    if (!env || !lp) return QNV_ERR_NULL;
    if (lp->cols < 0) return QNV_ERR_NO_PROBLEM;
    return QNV_OK;
}

/* Native simplex first; anything but a KKT-checked optimum goes to the certified Python path. */
static int simplex_then_certify(qnv_env* env, qnv_lp* lp) {
    int rc = ready(env, lp);
    if (rc != QNV_OK) return rc;
    if (lp->integer) return QNV_ERR_UNSUPPORTED;
    rc = run_engine(env, lp, "simplex");
    if (rc == QNV_OK && lp->stat == QNV_STAT_OPTIMAL) return QNV_OK;
    return run_engine(env, lp, "auto") == QNV_OK ? QNV_OK : rc;
}

int QNVprimopt(qnv_env* env, qnv_lp* lp) { return simplex_then_certify(env, lp); }
int QNVdualopt(qnv_env* env, qnv_lp* lp) { return simplex_then_certify(env, lp); }

int QNVbaropt(qnv_env* env, qnv_lp* lp) {
    const int rc = ready(env, lp);
    if (rc != QNV_OK) return rc;
    if (lp->integer) return QNV_ERR_UNSUPPORTED;
    return run_engine(env, lp, "ipm");
}

int QNVlpopt(qnv_env* env, qnv_lp* lp) {
    const int rc = ready(env, lp);
    if (rc != QNV_OK) return rc;
    if (lp->integer) return QNV_ERR_UNSUPPORTED;
    return run_engine(env, lp, "auto");
}

int QNVmipopt(qnv_env* env, qnv_lp* lp) {
    const int rc = ready(env, lp);
    if (rc != QNV_OK) return rc;
    return run_engine(env, lp, lp->integer ? "milp" : "auto");
}

int QNVgetstat(qnv_env* env, qnv_lp* lp) { return (env && lp) ? lp->stat : QNV_STAT_NONE; }

int QNVgetobjval(qnv_env* env, qnv_lp* lp, double* objval) {
    if (!env || !lp || !objval) return QNV_ERR_NULL;
    if (lp->stat != QNV_STAT_OPTIMAL) return QNV_ERR_NO_SOLUTION;
    *objval = lp->objval;
    return QNV_OK;
}

static int range_copy(const double* src, int have, int count, double* out, int begin, int end) {
    if (!out) return QNV_ERR_NULL;
    if (!have) return QNV_ERR_NO_SOLUTION;
    if (begin < 0 || end >= count || begin > end) return QNV_ERR_ARGUMENT;
    memcpy(out, src + begin, (size_t)(end - begin + 1) * sizeof(double));
    return QNV_OK;
}

int QNVgetx(qnv_env* env, qnv_lp* lp, double* x, int begin, int end) {
    if (!env || !lp) return QNV_ERR_NULL;
    return range_copy(lp->x, lp->has_x, lp->cols, x, begin, end);
}

int QNVgetpi(qnv_env* env, qnv_lp* lp, double* pi, int begin, int end) {
    if (!env || !lp) return QNV_ERR_NULL;
    return range_copy(lp->y, lp->has_y, lp->rows, pi, begin, end);
}

int QNVgetdj(qnv_env* env, qnv_lp* lp, double* dj, int begin, int end) {
    if (!env || !lp) return QNV_ERR_NULL;
    return range_copy(lp->dj, lp->has_dj, lp->cols, dj, begin, end);
}

int QNVgetslack(qnv_env* env, qnv_lp* lp, double* slack, int begin, int end) {
    if (!env || !lp || !slack) return QNV_ERR_NULL;
    if (!lp->has_x) return QNV_ERR_NO_SOLUTION;
    if (begin < 0 || end >= lp->rows || begin > end) return QNV_ERR_ARGUMENT;
    for (int i = begin; i <= end; ++i) {
        double act = 0.0;
        for (int k = lp->rowptr[i]; k < lp->rowptr[i + 1]; ++k) act += lp->val[k] * lp->x[lp->colidx[k]];
        const double lo = lp->lc[i];
        const double hi = lp->uc[i];
        double s;
        if (isfinite(lo) && isfinite(hi) && lo != hi) s = act - lo;     /* ranged row */
        else if (isfinite(hi) && !isfinite(lo)) s = hi - act;          /* <= */
        else if (isfinite(lo)) s = lo - act;                           /* >= and = */
        else s = NAN;                                                  /* free row */
        slack[i - begin] = s;
    }
    return QNV_OK;
}

int QNVsolution(qnv_env* env, qnv_lp* lp, int* stat, double* objval, double* x, double* pi,
                double* slack, double* dj) {
    if (!env || !lp) return QNV_ERR_NULL;
    if (stat) *stat = lp->stat;
    if (!lp->has_x) return QNV_ERR_NO_SOLUTION;
    if (objval) *objval = lp->objval;
    int rc = QNV_OK;
    if (x && lp->cols) rc = QNVgetx(env, lp, x, 0, lp->cols - 1);
    if (rc == QNV_OK && pi && lp->rows) rc = QNVgetpi(env, lp, pi, 0, lp->rows - 1);
    if (rc == QNV_OK && slack && lp->rows) rc = QNVgetslack(env, lp, slack, 0, lp->rows - 1);
    if (rc == QNV_OK && dj && lp->cols) rc = QNVgetdj(env, lp, dj, 0, lp->cols - 1);
    return rc;
}

const char* QNVgetcertificate(qnv_env* env, qnv_lp* lp) {
    return (env && lp && lp->cert) ? lp->cert : "";
}

const char* QNVgetengine(qnv_env* env, qnv_lp* lp) {
    return (env && lp) ? lp->engine : "";
}
