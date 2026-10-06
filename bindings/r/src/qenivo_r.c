/* R binding. .Call entry points load the C ABI at runtime (QENIVO_C_DLL) and solve
 * the bound LP and the two-binary MILP. Build with R CMD SHLIB once R is installed;
 * this file is not part of the C ABI CMake target because it includes Rinternals.h.
 */
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE lib_t;
static lib_t open_lib(const char* path) { return LoadLibraryA(path); }
static void* sym(lib_t lib, const char* name) { return (void*)GetProcAddress(lib, name); }
#else
#include <dlfcn.h>
typedef void* lib_t;
static lib_t open_lib(const char* path) { return dlopen(path, RTLD_NOW); }
static void* sym(lib_t lib, const char* name) { return dlsym(lib, name); }
#endif

typedef void* model_t;
typedef model_t (*create_fn)(int, int, const int*, const int*, const double*, const double*, const double*,
                             const double*, const double*, const double*, const int*, const int*, const int*,
                             const double*, int*);
typedef int (*solve_fn)(model_t, const char*);
typedef int (*obj_fn)(model_t, double*);
typedef int (*x_fn)(model_t, double*, int);
typedef void (*free_fn)(model_t);
typedef const char* (*msg_fn)(void);

static lib_t library(void) {
    const char* path = getenv("QENIVO_C_DLL");
    if (!path || !path[0]) path = "qenivo_c";
    lib_t lib = open_lib(path);
    if (!lib) error("QENIVO_C_DLL did not load");
    return lib;
}

static SEXP solve_known(int milp) {
    lib_t lib = library();
    create_fn create = (create_fn)sym(lib, "qenivo_model_create");
    solve_fn solve = (solve_fn)sym(lib, "qenivo_solve");
    obj_fn objective = (obj_fn)sym(lib, "qenivo_result_objective");
    x_fn primal = (x_fn)sym(lib, "qenivo_result_x");
    free_fn release = (free_fn)sym(lib, "qenivo_model_free");
    msg_fn detail = (msg_fn)sym(lib, "qenivo_create_error_message");
    if (!create || !solve || !objective || !primal || !release) error("C ABI exports are missing");
    int err = 0;
    model_t model = NULL;
    int cols = milp ? 2 : 1;
    if (!milp) {
        int rp[1] = {0};
        double cost[1] = {-1.0}, lx[1] = {0.0}, ux[1] = {1.0};
        model = create(0, 1, rp, NULL, NULL, cost, NULL, NULL, lx, ux, NULL, NULL, NULL, NULL, &err);
    } else {
        int rp[2] = {0, 2}, ci[2] = {0, 1}, integer[2] = {1, 1};
        double ax[2] = {1.0, 1.0}, cost[2] = {-1.0, -2.0}, lc[1] = {R_NegInf}, uc[1] = {1.0};
        double lx[2] = {0.0, 0.0}, ux[2] = {1.0, 1.0};
        model = create(1, 2, rp, ci, ax, cost, lc, uc, lx, ux, integer, NULL, NULL, NULL, &err);
    }
    if (!model) error("qenivo_model_create failed: %s", detail ? detail() : "");
    const char* engine = milp ? "milp" : "simplex";
    if (solve(model, engine) != 0) {
        release(model);
        error("solve failed");
    }
    double obj = NA_REAL;
    objective(model, &obj);
    SEXP x = PROTECT(allocVector(REALSXP, cols));
    primal(model, REAL(x), cols);
    release(model);
    SEXP out = PROTECT(allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, ScalarReal(obj));
    SET_VECTOR_ELT(out, 1, x);
    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("objective"));
    SET_STRING_ELT(names, 1, mkChar("x"));
    setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(3);
    return out;
}

SEXP qenivo_bounded_lp(SEXP unused) {
    (void)unused;
    return solve_known(0);
}

SEXP qenivo_two_binary(SEXP unused) {
    (void)unused;
    return solve_known(1);
}

static const R_CallMethodDef calls[] = {
    {"qenivo_bounded_lp", (DL_FUNC)&qenivo_bounded_lp, 1},
    {"qenivo_two_binary", (DL_FUNC)&qenivo_two_binary, 1},
    {NULL, NULL, 0}
};

void R_init_qenivo(DllInfo* info) {
    R_registerRoutines(info, NULL, calls, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
