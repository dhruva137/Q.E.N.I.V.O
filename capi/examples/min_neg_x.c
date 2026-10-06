/* Standalone example: min -x subject to 0 <= x <= 1, optimum -1.
 * Build with CMake and run with no arguments. Exit status is 0 only at that optimum.
 */
#include "qenivo.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    int32_t rp[1] = {0};
    double cost[1] = {-1.0};
    double lx[1] = {0.0};
    double ux[1] = {1.0};
    int32_t err = 0;
    qenivo_model* model = qenivo_model_create(0, 1, rp, NULL, NULL, cost, NULL, NULL, lx, ux, NULL, NULL, NULL, NULL,
                                              &err);
    if (model == NULL) {
        fprintf(stderr, "create failed (%s) %s\n", qenivo_error_name(err), qenivo_create_error_message());
        return 1;
    }
    int32_t rc = qenivo_solve(model, "simplex");
    double obj = 0.0;
    double x = 0.0;
    if (rc != QENIVO_OK || qenivo_result_objective(model, &obj) != QENIVO_OK || qenivo_result_x(model, &x, 1) != QENIVO_OK) {
        fprintf(stderr, "solve failed (%s) %s\n", qenivo_error_name(rc), qenivo_last_error_message(model));
        qenivo_model_free(model);
        return 1;
    }
    printf("optimum %.17g x %.17g\n", obj, x);
    qenivo_model_free(model);
    return (fabs(obj + 1.0) < 1e-8 && fabs(x - 1.0) < 1e-8) ? 0 : 1;
}
