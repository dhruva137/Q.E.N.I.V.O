package qenivo;

/** JNI binding. The native side is capi/qenivo_jni.cpp, which calls qenivo.h. */
public final class Qenivo {
    static {
        System.loadLibrary("qenivo_jni");
    }

    private Qenivo() {}

    public static native void setPackageDir(String dir);

    public static native long create(int rows, int cols, int[] rowPtr, int[] colIdx, double[] values,
                                     double[] cost, double[] rowLower, double[] rowUpper,
                                     double[] colLower, double[] colUpper, int[] integrality);

    public static native int solve(long handle, String engine);

    public static native double objective(long handle);

    public static native double[] primal(long handle);

    public static native String certificate(long handle);

    public static native void dispose(long handle);
}
