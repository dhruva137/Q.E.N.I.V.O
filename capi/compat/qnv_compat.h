/* qnv_compat.h - QNV: an environment/problem style C interface over the QENIVO C ABI (qenivo.h).
 *
 * Planners who call a commercial callable library are used to a shape: open an environment,
 * create a problem object, copy the LP in column-major form with row senses and right-hand
 * sides, optionally mark integer columns, optimize, then query status, objective, primal
 * values, row duals, slacks and reduced costs over an index range. QNV offers that shape with
 * its own names (all QNV-prefixed), its own constants and its own status codes. It was written
 * from the public description of that data layout only (see docs/COMPATIBILITY.md); no vendor
 * header, source or documentation text was used, and no vendor function names are mapped.
 *
 * Certified answers: QNV_STAT_OPTIMAL, QNV_STAT_INFEASIBLE and QNV_STAT_UNBOUNDED are reported
 * only when the QENIVO certificate path proved them. A native-simplex optimum must pass the C
 * ABI's KKT residual check; any other native outcome is re-solved through QENIVO's Python
 * certificate path (Farkas ray for infeasible). Anything unproven is QNV_STAT_NOT_PROVEN.
 *
 * Bounds and right-hand sides with |value| >= QNV_INFBOUND are infinite.
 * Index ranges [begin, end] are inclusive and zero-based.
 */
#ifndef QNV_COMPAT_H
#define QNV_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

#define QNV_INFBOUND 1.0e20

#define QNV_MIN 1
#define QNV_MAX -1

/* Return codes of every int function (0 = the call ran). */
#define QNV_OK 0
#define QNV_ERR_NULL 1001
#define QNV_ERR_ARGUMENT 1002
#define QNV_ERR_NO_PROBLEM 1003
#define QNV_ERR_NO_SOLUTION 1004
#define QNV_ERR_UNSUPPORTED 1005
#define QNV_ERR_NOMEM 1006
#define QNV_ERR_ENGINE 1007

/* Solution status (QNVgetstat). Codes are QENIVO's own. */
#define QNV_STAT_NONE 0
#define QNV_STAT_OPTIMAL 1
#define QNV_STAT_INFEASIBLE 2
#define QNV_STAT_UNBOUNDED 3
#define QNV_STAT_NOT_PROVEN 4
#define QNV_STAT_TIME_LIMIT 5
#define QNV_STAT_ITERATION_LIMIT 6

/* Parameters. */
#define QNV_PARAM_TIMELIMIT 1  /* double, seconds, > 0 */
#define QNV_PARAM_TOLERANCE 2  /* double, relative KKT tolerance of the certificate, > 0 */
#define QNV_PARAM_THREADS 3    /* int: 0 (automatic) or 1; QENIVO single-model solves use one thread */
#define QNV_PARAM_MIPGAP 4     /* not available through the C ABI: set/get return QNV_ERR_UNSUPPORTED */

typedef struct qnv_env qnv_env;
typedef struct qnv_lp qnv_lp;

qnv_env* QNVopenenv(int* status);
int QNVcloseenv(qnv_env** env);
const char* QNVerrorstring(int code);

int QNVsetdblparam(qnv_env* env, int which, double value);
int QNVgetdblparam(qnv_env* env, int which, double* value);
int QNVsetintparam(qnv_env* env, int which, int value);
int QNVgetintparam(qnv_env* env, int which, int* value);

qnv_lp* QNVcreateprob(qnv_env* env, int* status, const char* name);
int QNVfreeprob(qnv_env* env, qnv_lp** lp);

/* Column-major copy. For column j its entries are colstart[j] .. colstart[j]+colcount[j]-1 of
 * rowindex/value. sense[i] is 'L' (<=), 'G' (>=), 'E' (=) or 'R' (ranged: activity between
 * rhs[i] and rhs[i]+rngval[i], in whichever order). rhs NULL means 0, lb NULL means 0, ub NULL
 * means +infinity, rngval may be NULL when no row is ranged. Replaces any earlier data and
 * makes every column continuous. */
int QNVcopylp(qnv_env* env, qnv_lp* lp, int numcols, int numrows, int objsense, const double* obj,
              const double* rhs, const char* sense, const int* colstart, const int* colcount,
              const int* rowindex, const double* value, const double* lb, const double* ub,
              const double* rngval);
/* ctype[j] is 'C' continuous, 'I' integer or 'B' binary (bounds clipped to [0, 1]). */
int QNVcopyctype(qnv_env* env, qnv_lp* lp, const char* ctype);
int QNVchgobjsen(qnv_env* env, qnv_lp* lp, int objsense);
int QNVgetnumrows(qnv_env* env, qnv_lp* lp);
int QNVgetnumcols(qnv_env* env, qnv_lp* lp);

/* Solve. QNVlpopt and QNVmipopt use QENIVO's router, QNVdualopt the native dual simplex,
 * QNVbaropt the interior point method. QNVprimopt also runs the native simplex core, which
 * follows its own dual method: the C ABI exposes no primal-only switch (the qenivo-io shell's
 * primopt does). */
int QNVlpopt(qnv_env* env, qnv_lp* lp);
int QNVprimopt(qnv_env* env, qnv_lp* lp);
int QNVdualopt(qnv_env* env, qnv_lp* lp);
int QNVbaropt(qnv_env* env, qnv_lp* lp);
int QNVmipopt(qnv_env* env, qnv_lp* lp);

int QNVgetstat(qnv_env* env, qnv_lp* lp);
const char* QNVgetstatstring(int stat);
int QNVgetobjval(qnv_env* env, qnv_lp* lp, double* objval);
int QNVgetx(qnv_env* env, qnv_lp* lp, double* x, int begin, int end);
int QNVgetpi(qnv_env* env, qnv_lp* lp, double* pi, int begin, int end);    /* row duals, user's sense */
/* Slack: rhs - activity for 'L', 'G' and 'E' rows; activity - rhs for an 'R' row. */
int QNVgetslack(qnv_env* env, qnv_lp* lp, double* slack, int begin, int end);
int QNVgetdj(qnv_env* env, qnv_lp* lp, double* dj, int begin, int end);    /* reduced costs, user's sense */
/* Any output pointer may be NULL. Returns QNV_ERR_NO_SOLUTION when there is no primal point. */
int QNVsolution(qnv_env* env, qnv_lp* lp, int* stat, double* objval, double* x, double* pi,
                double* slack, double* dj);
/* Certificate JSON of the last solve (valid until the next solve or free); "" before one. */
const char* QNVgetcertificate(qnv_env* env, qnv_lp* lp);
/* Engine actually used for the last solve ("native simplex" or "embedded python"). */
const char* QNVgetengine(qnv_env* env, qnv_lp* lp);

#ifdef __cplusplus
}
#endif

#endif
