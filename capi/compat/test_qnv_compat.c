/* Test program for the QNV interface. Exit status 0 only when every check passes.
 *
 * Needs QENIVO_PYTHON_DLL and QENIVO_SRC in the environment for the checks that go through the
 * embedded Python certificate path (barrier, MILP, infeasibility proof); capi/compat/build_qnv.py
 * builds this program and runs it with them set.
 *
 * Model A (maximize 3x + 2y), optimum 11 at x = 3 (upper bound), y = 1:
 *   cap:   x +  y <= 4      binding, dual 2
 *   room:  x + 3y <= 7      slack 1
 *   band:  x -  y in [0, 5] ranged (rhs 0, range 5), activity 2, slack 2
 *   floor:      y >= 0.5    slack 0.5 - 1 = -0.5
 *   0 <= x <= 3, y >= 0; reduced cost of x is 3 - 2 = 1.
 * Model B: x + y >= 3 with 0 <= x, y <= 1 is infeasible.
 * Model C: binary knapsack max 8a + 11b + 6c + 4d s.t. 5a + 7b + 4c + 3d <= 14, optimum 21.
 */
#include "qnv_compat.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;

static void expect(int cond, const char* what) {
    g_checks += 1;
    if (!cond) {
        g_failures += 1;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static int near(double a, double b, double tol) { return fabs(a - b) <= tol * (1.0 + fabs(b)); }

static void load_a(qnv_env* env, qnv_lp* lp) {
    /* columns x, y; rows cap, room, band, floor */
    const double obj[2] = {3.0, 2.0};
    const double rhs[4] = {4.0, 7.0, 0.0, 0.5};
    const char sense[4] = {'L', 'L', 'R', 'G'};
    const int start[2] = {0, 3};
    const int count[2] = {3, 4};
    const int row[7] = {0, 1, 2, 0, 1, 2, 3};
    const double val[7] = {1.0, 1.0, 1.0, 1.0, 3.0, -1.0, 1.0};
    const double lb[2] = {0.0, 0.0};
    const double ub[2] = {3.0, QNV_INFBOUND};
    const double rng[4] = {0.0, 0.0, 5.0, 0.0};
    expect(QNVcopylp(env, lp, 2, 4, QNV_MAX, obj, rhs, sense, start, count, row, val, lb, ub, rng) == QNV_OK,
           "copylp model A");
}

static void check_model_a(qnv_env* env, qnv_lp* lp, const char* label, double tol) {
    char what[160];
    int stat = -1;
    double obj = 0.0, x[2], pi[4], slack[4], dj[2];
    expect(QNVsolution(env, lp, &stat, &obj, x, pi, slack, dj) == QNV_OK, label);
    snprintf(what, sizeof(what), "%s: certified optimal (stat %d)", label, stat);
    expect(stat == QNV_STAT_OPTIMAL, what);
    snprintf(what, sizeof(what), "%s: objective 11 (got %.12g)", label, obj);
    expect(near(obj, 11.0, tol), what);
    snprintf(what, sizeof(what), "%s: x = (3, 1)", label);
    expect(near(x[0], 3.0, tol) && near(x[1], 1.0, tol), what);
    snprintf(what, sizeof(what), "%s: duals (2, 0, 0, 0) (got %g %g %g %g)", label, pi[0], pi[1], pi[2], pi[3]);
    expect(near(pi[0], 2.0, tol) && fabs(pi[1]) < 1e-5 && fabs(pi[2]) < 1e-5 && fabs(pi[3]) < 1e-5, what);
    snprintf(what, sizeof(what), "%s: slacks (0, 1, 2, -0.5) (got %g %g %g %g)", label, slack[0], slack[1],
             slack[2], slack[3]);
    expect(fabs(slack[0]) < 1e-5 && near(slack[1], 1.0, tol) && near(slack[2], 2.0, tol) &&
               near(slack[3], -0.5, tol), what);
    snprintf(what, sizeof(what), "%s: reduced costs (1, 0) (got %g %g)", label, dj[0], dj[1]);
    expect(near(dj[0], 1.0, tol) && fabs(dj[1]) < 1e-5, what);
    double one = 0.0;
    expect(QNVgetx(env, lp, &one, 1, 1) == QNV_OK && near(one, 1.0, tol), "range query of one column");
    expect(QNVgetx(env, lp, &one, 1, 2) == QNV_ERR_ARGUMENT, "range past the last column is refused");
    expect(strstr(QNVgetcertificate(env, lp), "optimal") != NULL, "certificate names the verdict");
}

static void check_arguments(qnv_env* env) {
    int status = -1;
    expect(QNVcreateprob(NULL, &status, "x") == NULL && status == QNV_ERR_NULL, "null env refused");
    qnv_lp* lp = QNVcreateprob(env, &status, "bad model");
    expect(lp != NULL && status == QNV_OK, "createprob");
    const double obj[1] = {1.0};
    const char sense[1] = {'Q'};
    const int start[1] = {0}, count[1] = {1}, row[1] = {0};
    const double val[1] = {1.0}, rhs[1] = {1.0};
    expect(QNVcopylp(env, lp, 1, 1, QNV_MIN, obj, rhs, sense, start, count, row, val, NULL, NULL, NULL) ==
               QNV_ERR_ARGUMENT, "unknown row sense refused");
    const int bad_row[1] = {3};
    const char ok_sense[1] = {'L'};
    expect(QNVcopylp(env, lp, 1, 1, QNV_MIN, obj, rhs, ok_sense, start, count, bad_row, val, NULL, NULL, NULL) ==
               QNV_ERR_ARGUMENT, "row index out of range refused");
    expect(QNVlpopt(env, lp) == QNV_ERR_NO_PROBLEM, "optimize without data refused");
    expect(QNVgetstat(env, lp) == QNV_STAT_NONE, "no status before a solve");
    double v = 0.0;
    expect(QNVgetobjval(env, lp, &v) == QNV_ERR_NO_SOLUTION, "no objective before a solve");
    expect(QNVsetdblparam(env, QNV_PARAM_MIPGAP, 1e-4) == QNV_ERR_UNSUPPORTED, "mipgap is reported unsupported");
    expect(QNVsetintparam(env, QNV_PARAM_THREADS, 4) == QNV_ERR_UNSUPPORTED, "threads > 1 is reported unsupported");
    expect(QNVsetintparam(env, QNV_PARAM_THREADS, 1) == QNV_OK, "threads 1 accepted");
    expect(QNVsetdblparam(env, QNV_PARAM_TIMELIMIT, -1.0) == QNV_ERR_ARGUMENT, "negative time limit refused");
    expect(QNVsetdblparam(env, QNV_PARAM_TIMELIMIT, 60.0) == QNV_OK, "time limit set");
    expect(QNVgetdblparam(env, QNV_PARAM_TIMELIMIT, &v) == QNV_OK && v == 60.0, "time limit read back");
    expect(strcmp(QNVerrorstring(QNV_ERR_UNSUPPORTED), "not available in QENIVO through the C interface") == 0,
           "error string");
    QNVfreeprob(env, &lp);
    expect(lp == NULL, "freeprob clears the pointer");
}

int main(void) {
    int status = -1;
    qnv_env* env = QNVopenenv(&status);
    if (!env || status != QNV_OK) {
        fprintf(stderr, "FAIL openenv\n");
        return 1;
    }
    check_arguments(env);

    qnv_lp* lp = QNVcreateprob(env, &status, "model_a");
    load_a(env, lp);
    expect(QNVgetnumrows(env, lp) == 4 && QNVgetnumcols(env, lp) == 2, "sizes");
    expect(QNVdualopt(env, lp) == QNV_OK, "dualopt runs");
    expect(strcmp(QNVgetengine(env, lp), "native simplex") == 0, "dualopt uses the native simplex");
    check_model_a(env, lp, "dualopt", 1e-9);
    expect(QNVprimopt(env, lp) == QNV_OK, "primopt runs");
    check_model_a(env, lp, "primopt", 1e-9);
    expect(QNVbaropt(env, lp) == QNV_OK, "baropt runs");
    expect(strcmp(QNVgetengine(env, lp), "embedded python") == 0, "baropt goes through the Python path");
    check_model_a(env, lp, "baropt", 1e-5);

    /* Minimize the same rows: x >= y (band) and y >= 0.5 give x = y = 0.5, objective 2.5. */
    expect(QNVchgobjsen(env, lp, QNV_MIN) == QNV_OK, "chgobjsen");
    expect(QNVgetstat(env, lp) == QNV_STAT_NONE, "changing the sense drops the old solution");
    expect(QNVlpopt(env, lp) == QNV_OK, "lpopt min");
    double obj = 0.0;
    expect(QNVgetobjval(env, lp, &obj) == QNV_OK && near(obj, 2.5, 1e-7), "min objective 2.5");
    QNVfreeprob(env, &lp);

    /* Infeasible: the native core says so; the QNV layer asks the certificate path to prove it. */
    lp = QNVcreateprob(env, &status, "model_b");
    {
        const double cost[2] = {1.0, 0.0};
        const double rhs[1] = {3.0};
        const char sense[1] = {'G'};
        const int start[2] = {0, 1}, count[2] = {1, 1}, row[2] = {0, 0};
        const double val[2] = {1.0, 1.0}, ub[2] = {1.0, 1.0};
        expect(QNVcopylp(env, lp, 2, 1, QNV_MIN, cost, rhs, sense, start, count, row, val, NULL, ub, NULL) == QNV_OK,
               "copylp model B");
    }
    expect(QNVdualopt(env, lp) == QNV_OK, "dualopt on an infeasible model runs");
    expect(QNVgetstat(env, lp) == QNV_STAT_INFEASIBLE, "infeasibility is certified");
    expect(strstr(QNVgetcertificate(env, lp), "farkas") != NULL, "certificate carries a Farkas ray");
    expect(QNVgetobjval(env, lp, &obj) == QNV_ERR_NO_SOLUTION, "no objective for an infeasible model");
    QNVfreeprob(env, &lp);

    /* Binary knapsack. */
    lp = QNVcreateprob(env, &status, "model_c");
    {
        const double cost[4] = {8.0, 11.0, 6.0, 4.0};
        const double rhs[1] = {14.0};
        const char sense[1] = {'L'};
        const int start[4] = {0, 1, 2, 3}, count[4] = {1, 1, 1, 1}, row[4] = {0, 0, 0, 0};
        const double val[4] = {5.0, 7.0, 4.0, 3.0};
        expect(QNVcopylp(env, lp, 4, 1, QNV_MAX, cost, rhs, sense, start, count, row, val, NULL, NULL, NULL) ==
                   QNV_OK, "copylp model C");
        expect(QNVcopyctype(env, lp, "BBBB") == QNV_OK, "copyctype");
    }
    expect(QNVlpopt(env, lp) == QNV_ERR_UNSUPPORTED, "lpopt refuses integer columns");
    expect(QNVmipopt(env, lp) == QNV_OK, "mipopt runs");
    double x[4] = {0, 0, 0, 0};
    expect(QNVgetstat(env, lp) == QNV_STAT_OPTIMAL, "knapsack certified optimal");
    expect(QNVgetobjval(env, lp, &obj) == QNV_OK && near(obj, 21.0, 1e-9), "knapsack objective 21");
    expect(QNVgetx(env, lp, x, 0, 3) == QNV_OK && fabs(x[0]) < 1e-6 && fabs(x[1] - 1) < 1e-6 &&
               fabs(x[2] - 1) < 1e-6 && fabs(x[3] - 1) < 1e-6, "knapsack picks b, c, d");
    double pi = 0.0;
    expect(QNVgetpi(env, lp, &pi, 0, 0) == QNV_ERR_NO_SOLUTION, "a MILP has no duals");
    QNVfreeprob(env, &lp);

    QNVcloseenv(&env);
    expect(env == NULL, "closeenv clears the pointer");
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
