/* Interface between the C ABI and the embedded CPython interpreter.
 * The simplex path does not include this header.
 */
#ifndef QENIVO_PYTHON_BRIDGE_H
#define QENIVO_PYTHON_BRIDGE_H

#include <stdint.h>

#include <string>
#include <vector>

struct PythonRequest {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int32_t> a_ptr, a_idx;
    std::vector<double> a_val;
    std::vector<double> cost, row_lower, row_upper, col_lower, col_upper;
    bool has_integer = false;
    std::vector<int32_t> integer;
    bool has_q = false;
    std::vector<int32_t> q_ptr, q_idx;
    std::vector<double> q_val;
    std::string engine;
    double tolerance = 1e-8;
    double time_limit = 3600.0;
    std::string name;
};

struct PythonResponse {
    int32_t status = 6;
    std::string status_name;
    std::string verdict;
    bool has_objective = false;
    double objective = 0.0;
    int64_t iterations = 0;
    bool has_x = false;
    bool has_y = false;
    bool has_reduced = false;
    bool has_basis = false;
    std::vector<double> x, y, reduced;
    std::vector<int32_t> column_basis, row_basis;
    std::string certificate;
    std::string error;
};

/* Returns QENIVO_OK or a QENIVO_ERR_* code. Fills response.error on failure. */
int python_solve(const PythonRequest& request, PythonResponse& response);

#endif
