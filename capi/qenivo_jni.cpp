/* JNI glue compiled as C++. It calls the C ABI; the simplex path is the native core
 * and every other engine is the embedded Python bridge. JNIEXPORT names stay C linkage.
 */
#include "qenivo.h"

#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
static void set_env(const char* key, const char* value) { _putenv_s(key, value); }
#else
static void set_env(const char* key, const char* value) { setenv(key, value, 1); }
#endif

static void throw_code(JNIEnv* env, int32_t code, const char* detail) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s (%d)", detail ? detail : "qenivo", (int)code);
    if (cls) env->ThrowNew(cls, buf);
}

static int32_t* ints(JNIEnv* env, jintArray arr) {
    if (!arr) return nullptr;
    jsize n = env->GetArrayLength(arr);
    int32_t* out = (int32_t*)std::malloc(sizeof(int32_t) * (n ? (size_t)n : 1));
    if (!out) return nullptr;
    jint* src = env->GetIntArrayElements(arr, nullptr);
    for (jsize i = 0; i < n; ++i) out[i] = (int32_t)src[i];
    env->ReleaseIntArrayElements(arr, src, JNI_ABORT);
    return out;
}

static double* doubles(JNIEnv* env, jdoubleArray arr) {
    if (!arr) return nullptr;
    jsize n = env->GetArrayLength(arr);
    double* out = (double*)std::malloc(sizeof(double) * (n ? (size_t)n : 1));
    if (!out) return nullptr;
    env->GetDoubleArrayRegion(arr, 0, n, out);
    return out;
}

extern "C" {

JNIEXPORT void JNICALL Java_qenivo_QENIVO_setPackageDir(JNIEnv* env, jclass, jstring dir) {
    const char* text = env->GetStringUTFChars(dir, nullptr);
    if (text) {
        set_env("QENIVO_SRC", text);
        env->ReleaseStringUTFChars(dir, text);
    }
}

JNIEXPORT jlong JNICALL Java_qenivo_QENIVO_create(JNIEnv* env, jclass, jint rows, jint cols, jintArray row_ptr,
                                                  jintArray col_idx, jdoubleArray values, jdoubleArray cost,
                                                  jdoubleArray row_lower, jdoubleArray row_upper, jdoubleArray col_lower,
                                                  jdoubleArray col_upper, jintArray integrality) {
    int32_t* rp = ints(env, row_ptr);
    int32_t* ci = ints(env, col_idx);
    double* ax = doubles(env, values);
    double* c = doubles(env, cost);
    double* lc = doubles(env, row_lower);
    double* uc = doubles(env, row_upper);
    double* lx = doubles(env, col_lower);
    double* ux = doubles(env, col_upper);
    int32_t* integer = ints(env, integrality);
    int32_t err = 0;
    qenivo_model* model = qenivo_model_create(rows, cols, rp, ci, ax, c, lc, uc, lx, ux, integer, nullptr, nullptr,
                                              nullptr, &err);
    std::free(rp); std::free(ci); std::free(ax); std::free(c);
    std::free(lc); std::free(uc); std::free(lx); std::free(ux); std::free(integer);
    if (!model) {
        throw_code(env, err, qenivo_create_error_message());
        return 0;
    }
    return (jlong)model;
}

JNIEXPORT jint JNICALL Java_qenivo_QENIVO_solve(JNIEnv* env, jclass, jlong handle, jstring engine) {
    qenivo_model* model = (qenivo_model*)handle;
    const char* name = env->GetStringUTFChars(engine, nullptr);
    int32_t rc = qenivo_solve(model, name);
    env->ReleaseStringUTFChars(engine, name);
    if (rc != QENIVO_OK) {
        throw_code(env, rc, qenivo_last_error_message(model));
        return -1;
    }
    return qenivo_result_status(model);
}

JNIEXPORT jdouble JNICALL Java_qenivo_QENIVO_objective(JNIEnv*, jclass, jlong handle) {
    double obj = 0.0;
    qenivo_result_objective((qenivo_model*)handle, &obj);
    return obj;
}

JNIEXPORT jdoubleArray JNICALL Java_qenivo_QENIVO_primal(JNIEnv* env, jclass, jlong handle) {
    qenivo_model* model = (qenivo_model*)handle;
    int32_t cols = qenivo_model_cols(model);
    jdoubleArray out = env->NewDoubleArray(cols);
    if (!out || cols <= 0) return out;
    double* buf = (double*)std::malloc(sizeof(double) * (size_t)cols);
    if (!buf) return out;
    qenivo_result_x(model, buf, cols);
    env->SetDoubleArrayRegion(out, 0, cols, buf);
    std::free(buf);
    return out;
}

JNIEXPORT jstring JNICALL Java_qenivo_QENIVO_certificate(JNIEnv* env, jclass, jlong handle) {
    const char* json = qenivo_result_certificate_json((qenivo_model*)handle);
    return env->NewStringUTF(json ? json : "");
}

JNIEXPORT void JNICALL Java_qenivo_QENIVO_dispose(JNIEnv*, jclass, jlong handle) {
    qenivo_model_free((qenivo_model*)handle);
}

}
