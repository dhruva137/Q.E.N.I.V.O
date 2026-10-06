/* Tiny C harness for the Qenivo ABI. No GoogleTest.
 *
 * Proves the bound LP  min -x  s.t.  0 <= x <= 1  has optimum -1, then a
 * two-binary MILP checked by hand: min -x-2y s.t. x+y <= 1, x,y in {0,1},
 * optimum -2 at (0, 1). The MILP goes through the embedded Python bridge, so
 * pass the package directory as argv[1] (the directory that contains qenivo/).
 */
#include "qenivo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_progress = 0;

static void expect(int cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", msg);
        g_failures += 1;
    }
}

static void on_progress(void* user, int64_t iteration, double objective, int32_t status) {
    int* calls = (int*)user;
    (void)objective;
    *calls += 1;
    if (status < 0) expect(iteration == 0, "start callback is iteration 0");
}

static void set_src(const char* src) {
    if (src == NULL || src[0] == '\0') return;
#if defined(_WIN32)
    static char buf[4096];
    snprintf(buf, sizeof(buf), "QENIVO_SRC=%s", src);
    _putenv(buf);
#else
    setenv("QENIVO_SRC", src, 1);
#endif
}

static qenivo_model* must_create(int32_t rows, int32_t cols, const int32_t* rp, const int32_t* ci, const double* ax,
                                 const double* c, const double* lc, const double* uc, const double* lx,
                                 const double* ux, const int32_t* integer) {
    int32_t err = -1;
    qenivo_model* model = qenivo_model_create(rows, cols, rp, ci, ax, c, lc, uc, lx, ux, integer, NULL, NULL, NULL, &err);
    if (model == NULL) {
        fprintf(stderr, "FAIL create (%s) %s\n", qenivo_error_name(err), qenivo_create_error_message());
        g_failures += 1;
    }
    return model;
}

static void check_bad_arguments(void) {
    int32_t err = 0;
    int32_t rp[1] = {0};
    double cost = -1.0;
    double lx = 0.0;
    double ux = 1.0;
    qenivo_model* missing = qenivo_model_create(0, 1, NULL, NULL, NULL, &cost, NULL, NULL, &lx, &ux, NULL, NULL, NULL,
                                                NULL, &err);
    expect(missing == NULL && err == QENIVO_ERR_ARGUMENT, "null CSR is an argument error");
    expect(qenivo_solve(NULL, "simplex") == QENIVO_ERR_NULL, "null handle is an error");
    expect(qenivo_abi_version() == QENIVO_ABI_VERSION, "abi version");

    qenivo_model* model = must_create(0, 1, rp, NULL, NULL, &cost, NULL, NULL, &lx, &ux, NULL);
    if (model == NULL) return;
    int32_t integer[1] = {1};
    qenivo_model_free(model);
    model = qenivo_model_create(0, 1, rp, NULL, NULL, &cost, NULL, NULL, &lx, &ux, integer, NULL, NULL, NULL, &err);
    expect(model != NULL, "integer model allocates");
    if (model != NULL) {
        expect(qenivo_solve(model, "simplex") == QENIVO_ERR_ENGINE, "simplex refuses integrality");
        qenivo_model_free(model);
    }
}

static void check_bound_lp(void) {
    /* min -x s.t. 0 <= x <= 1. Optimum -1 at x = 1, reduced cost -1, at upper bound. */
    int32_t rp[1] = {0};
    double cost[1] = {-1.0};
    double lx[1] = {0.0};
    double ux[1] = {1.0};
    qenivo_model* model = must_create(0, 1, rp, NULL, NULL, cost, NULL, NULL, lx, ux, NULL);
    if (model == NULL) return;
    expect(qenivo_set_progress(model, on_progress, &g_progress) == QENIVO_OK, "progress callback installs");
    expect(qenivo_set_tolerance(model, 1e-9) == QENIVO_OK, "tolerance");
    int32_t rc = qenivo_solve(model, "simplex");
    if (rc != QENIVO_OK) {
        fprintf(stderr, "FAIL simplex solve (%s) %s\n", qenivo_error_name(rc), qenivo_last_error_message(model));
        g_failures += 1;
        qenivo_model_free(model);
        return;
    }
    double obj = 0.0;
    double x = 0.0;
    double reduced = 0.0;
    int32_t basis = -1;
    expect(qenivo_result_objective(model, &obj) == QENIVO_OK, "objective readable");
    expect(qenivo_result_x(model, &x, 1) == QENIVO_OK, "x readable");
    expect(qenivo_result_reduced_costs(model, &reduced, 1) == QENIVO_OK, "reduced cost readable");
    expect(qenivo_result_basis(model, &basis, 1, NULL, 0) == QENIVO_OK, "basis readable");
    expect(qenivo_result_status(model) == QENIVO_OPTIMAL, "status optimal");
    expect(fabs(obj + 1.0) < 1e-8, "optimum is -1");
    expect(fabs(x - 1.0) < 1e-8, "x is 1");
    expect(fabs(reduced + 1.0) < 1e-8, "reduced cost is -1");
    expect(basis == QENIVO_AT_UPPER, "x is at its upper bound");
    expect(g_progress >= 2, "progress callback ran at start and finish");
    const char* cert = qenivo_result_certificate_json(model);
    expect(cert != NULL && strstr(cert, "native simplex") != NULL, "certificate names the native simplex");
    expect(cert != NULL && strstr(cert, "qenivo.capi.certificate/1") != NULL, "certificate schema");
    printf("optimum %.17g\n", obj);
    qenivo_model_free(model);
}

static void check_vertex(void) {
    /* min -x - y s.t. x + y <= 1, x,y >= 0. A vertex has objective -1. */
    int32_t rp[2] = {0, 2};
    int32_t ci[2] = {0, 1};
    double ax[2] = {1.0, 1.0};
    double cost[2] = {-1.0, -1.0};
    double lc[1] = {-INFINITY};
    double uc[1] = {1.0};
    double lx[2] = {0.0, 0.0};
    double ux[2] = {INFINITY, INFINITY};
    qenivo_model* model = must_create(1, 2, rp, ci, ax, cost, lc, uc, lx, ux, NULL);
    if (model == NULL) return;
    int32_t rc = qenivo_solve(model, "native-simplex");
    if (rc != QENIVO_OK) {
        fprintf(stderr, "FAIL vertex solve (%s) %s\n", qenivo_error_name(rc), qenivo_last_error_message(model));
        g_failures += 1;
        qenivo_model_free(model);
        return;
    }
    double obj = 0.0;
    double x[2] = {0.0, 0.0};
    double y = 0.0;
    int32_t col[2] = {0, 0};
    int32_t row[1] = {0};
    expect(qenivo_result_objective(model, &obj) == QENIVO_OK, "vertex objective");
    expect(qenivo_result_x(model, x, 2) == QENIVO_OK, "vertex x");
    expect(qenivo_result_y(model, &y, 1) == QENIVO_OK, "vertex dual");
    expect(qenivo_result_basis(model, col, 2, row, 1) == QENIVO_OK, "vertex basis");
    expect(fabs(obj + 1.0) < 1e-7, "vertex optimum is -1");
    expect(x[0] >= -1e-8 && x[1] >= -1e-8, "vertex is nonnegative");
    expect(fabs(x[0] + x[1] - 1.0) < 1e-6, "vertex meets x+y = 1");
    expect(isfinite(y), "row dual is finite");
    qenivo_model_free(model);
}

static void check_milp(void) {
    /* min -x - 2y s.t. x + y <= 1, x and y binary. Optimum -2 at (0, 1). */
    int32_t rp[2] = {0, 2};
    int32_t ci[2] = {0, 1};
    double ax[2] = {1.0, 1.0};
    double cost[2] = {-1.0, -2.0};
    double lc[1] = {-INFINITY};
    double uc[1] = {1.0};
    double lx[2] = {0.0, 0.0};
    double ux[2] = {1.0, 1.0};
    int32_t integer[2] = {1, 1};
    int32_t err = 0;
    qenivo_model* model = qenivo_model_create(1, 2, rp, ci, ax, cost, lc, uc, lx, ux, integer, NULL, NULL, NULL, &err);
    if (model == NULL) {
        fprintf(stderr, "FAIL milp create (%s) %s\n", qenivo_error_name(err), qenivo_create_error_message());
        g_failures += 1;
        return;
    }
    int32_t rc = qenivo_solve(model, "milp");
    if (rc != QENIVO_OK) {
        fprintf(stderr, "FAIL milp solve (%s) %s\n", qenivo_error_name(rc), qenivo_last_error_message(model));
        g_failures += 1;
        qenivo_model_free(model);
        return;
    }
    double obj = 0.0;
    double x[2] = {0.0, 0.0};
    expect(qenivo_result_status(model) == QENIVO_OPTIMAL, "milp status optimal");
    expect(qenivo_result_objective(model, &obj) == QENIVO_OK, "milp objective readable");
    expect(qenivo_result_x(model, x, 2) == QENIVO_OK, "milp x readable");
    expect(fabs(obj + 2.0) < 1e-6, "milp optimum is -2");
    expect(fabs(x[0]) < 1e-6 && fabs(x[1] - 1.0) < 1e-6, "milp point is (0, 1)");
    const char* cert = qenivo_result_certificate_json(model);
    expect(cert != NULL && strstr(cert, "qenivo.certificate/1") != NULL, "milp certificate is the Python certificate");
    printf("milp %.17g\n", obj);
    qenivo_model_free(model);
}

int main(int argc, char** argv) {
    set_src(argc > 1 ? argv[1] : getenv("QENIVO_SRC"));
    check_bad_arguments();
    check_bound_lp();
    check_vertex();
    check_milp();
    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("capi harness ok\n");
    return 0;
}
