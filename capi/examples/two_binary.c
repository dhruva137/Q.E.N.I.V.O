/* Standalone example: min -x - 2y s.t. x + y <= 1, x,y binary. Optimum -2 at (0, 1).
 * argv[1] is the directory that contains the qenivo Python package (embedded Python).
 */
#include "qenivo.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void set_src(const char* src) {
    if (src == NULL) return;
#if defined(_WIN32)
    static char buf[4096];
    snprintf(buf, sizeof(buf), "QENIVO_SRC=%s", src);
    _putenv(buf);
#else
    setenv("QENIVO_SRC", src, 1);
#endif
}

int main(int argc, char** argv) {
    set_src(argc > 1 ? argv[1] : getenv("QENIVO_SRC"));
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
        fprintf(stderr, "create failed (%s) %s\n", qenivo_error_name(err), qenivo_create_error_message());
        return 1;
    }
    int32_t rc = qenivo_solve(model, "milp");
    double obj = 0.0;
    double x[2] = {0.0, 0.0};
    if (rc != QENIVO_OK || qenivo_result_objective(model, &obj) != QENIVO_OK ||
        qenivo_result_x(model, x, 2) != QENIVO_OK) {
        fprintf(stderr, "solve failed (%s) %s\n", qenivo_error_name(rc), qenivo_last_error_message(model));
        qenivo_model_free(model);
        return 1;
    }
    printf("milp optimum %.17g x %.17g %.17g\n", obj, x[0], x[1]);
    qenivo_model_free(model);
    return (fabs(obj + 2.0) < 1e-6 && fabs(x[0]) < 1e-6 && fabs(x[1] - 1.0) < 1e-6) ? 0 : 1;
}
