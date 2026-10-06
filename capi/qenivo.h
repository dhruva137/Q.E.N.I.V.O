/* qenivo.h — versioned C ABI for the Qenivo solver.
 *
 * The simplex engine calls the native revised simplex (Maros, Computational
 * Techniques of the Simplex Method, Kluwer, 2003; Vanderbei, Linear Programming:
 * Foundations and Extensions, Springer) in native/simplex_core.cpp. Every other
 * engine name is solved by the embedded-Python bridge in python_bridge.cpp.
 * This header is original Qenivo ABI design. No solver source was copied.
 *
 * The ABI is minimize-only:  min c'x  s.t.  row_lower <= A x <= row_upper,
 * col_lower <= x <= col_upper, with an optional 0.5 x'Qx term and an optional
 * integrality mask. A maximisation model is the caller's to negate.
 * Bounds use IEEE infinity for "no bound".
 *
 * Handles are thread-safe against each other: each model has its own lock, and
 * the native simplex on two handles may run concurrently. Embedded-Python solves
 * are serialised, because CPython has one GIL. A progress callback must not
 * re-enter the same handle (the lock is held). Strings returned by a handle are
 * valid until the next call on that handle. No C++ exception crosses this ABI;
 * every function returns an integer code.
 */
#ifndef QENIVO_H
#define QENIVO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QENIVO_ABI_VERSION 1

#if defined(_WIN32)
#  if defined(QENIVO_BUILD)
#    define QENIVO_API __declspec(dllexport)
#  else
#    define QENIVO_API __declspec(dllimport)
#  endif
#else
#  if defined(QENIVO_BUILD)
#    define QENIVO_API __attribute__((visibility("default")))
#  else
#    define QENIVO_API
#  endif
#endif

/* Solver status. Distinct from the error codes below. */
#define QENIVO_OPTIMAL 0
#define QENIVO_INFEASIBLE 1
#define QENIVO_UNBOUNDED 2
#define QENIVO_ITERATION_LIMIT 3
#define QENIVO_TIME_LIMIT 4
#define QENIVO_NUMERICAL 5
#define QENIVO_NOT_PROVEN 6

/* Basis codes, the same integers the native simplex writes. */
#define QENIVO_BASIC 0
#define QENIVO_AT_LOWER 1
#define QENIVO_AT_UPPER 2
#define QENIVO_NONBASIC_FREE 3

/* Error codes. QENIVO_OK means the call ran; the model may still be infeasible. */
#define QENIVO_OK 0
#define QENIVO_ERR_NULL 1
#define QENIVO_ERR_ARGUMENT 2
#define QENIVO_ERR_STATE 3
#define QENIVO_ERR_ENGINE 4
#define QENIVO_ERR_SOLVE 5
#define QENIVO_ERR_PYTHON 6
#define QENIVO_ERR_NOMEM 7
#define QENIVO_ERR_INTERNAL 8

typedef struct qenivo_model qenivo_model;

/* status == -1 and iteration == 0 means the solve is starting (objective is NaN).
 * A later call reports the iteration count, the objective, and a QENIVO_* status. */
typedef void (*qenivo_progress_fn)(void* user, int64_t iteration, double objective, int32_t status);

QENIVO_API int32_t qenivo_abi_version(void);
QENIVO_API const char* qenivo_error_name(int32_t code);

/* CSR matrix A is rows x cols. a_rowptr has length rows+1 and a_rowptr[0] == 0,
 * including the no-row case (pointer to a single 0). Column indices and values
 * may be NULL when the matrix has no entries. q_* is the optional quadratic
 * term, CSR of order cols, or all NULL. integrality is NULL or length cols,
 * nonzero meaning an integer variable. Pointers need only live for this call;
 * the handle keeps its own copy. On failure returns NULL and writes *error
 * when error is not NULL. qenivo_create_error_message() is the detail, and it
 * is thread-local. */
QENIVO_API qenivo_model* qenivo_model_create(
    int32_t rows, int32_t cols,
    const int32_t* a_rowptr, const int32_t* a_colidx, const double* a_values,
    const double* cost,
    const double* row_lower, const double* row_upper,
    const double* col_lower, const double* col_upper,
    const int32_t* integrality,
    const int32_t* q_rowptr, const int32_t* q_colidx, const double* q_values,
    int32_t* error);

QENIVO_API void qenivo_model_free(qenivo_model* model);
QENIVO_API const char* qenivo_create_error_message(void);

QENIVO_API int32_t qenivo_set_name(qenivo_model* model, const char* name);
QENIVO_API int32_t qenivo_set_tolerance(qenivo_model* model, double absolute);
QENIVO_API int32_t qenivo_set_time_limit(qenivo_model* model, double seconds);
QENIVO_API int32_t qenivo_set_progress(qenivo_model* model, qenivo_progress_fn fn, void* user);

QENIVO_API int32_t qenivo_model_rows(const qenivo_model* model);
QENIVO_API int32_t qenivo_model_cols(const qenivo_model* model);

/* engine "simplex", "native" or "native-simplex" calls the native simplex and
 * refuses integrality or a quadratic term. Any other name (for example "milp")
 * goes through the embedded Python interpreter. */
/* Do not free a handle while another thread is still inside a call on it.
 * Distinct handles may be solved concurrently. */
QENIVO_API int32_t qenivo_solve(qenivo_model* model, const char* engine);

QENIVO_API int32_t qenivo_last_error(const qenivo_model* model);
QENIVO_API const char* qenivo_last_error_message(const qenivo_model* model);

/* -1 when solve has not produced a result. */
QENIVO_API int32_t qenivo_result_status(const qenivo_model* model);
QENIVO_API const char* qenivo_result_status_name(const qenivo_model* model);
QENIVO_API int32_t qenivo_result_objective(const qenivo_model* model, double* objective);
QENIVO_API int64_t qenivo_result_iterations(const qenivo_model* model);
QENIVO_API int32_t qenivo_result_has_duals(const qenivo_model* model);
QENIVO_API int32_t qenivo_result_has_basis(const qenivo_model* model);
QENIVO_API int32_t qenivo_result_has_reduced_costs(const qenivo_model* model);
QENIVO_API int32_t qenivo_result_x(const qenivo_model* model, double* x, int32_t cols);
QENIVO_API int32_t qenivo_result_y(const qenivo_model* model, double* y, int32_t rows);
QENIVO_API int32_t qenivo_result_reduced_costs(const qenivo_model* model, double* reduced, int32_t cols);
QENIVO_API int32_t qenivo_result_basis(const qenivo_model* model,
                                       int32_t* column_status, int32_t cols,
                                       int32_t* row_status, int32_t rows);
QENIVO_API const char* qenivo_result_certificate_json(const qenivo_model* model);

#ifdef __cplusplus
}
#endif

#endif
