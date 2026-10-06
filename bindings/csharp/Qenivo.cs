// P/Invoke binding for the Qenivo C ABI (qenivo.h).
// The simplex engine is the native revised simplex. Other engine names are
// answered by the embedded Python bridge inside the same shared library.
using System.Runtime.InteropServices;

namespace Qenivo;

public static class Abi
{
    public const int Optimal = 0;
    public const int AtUpper = 2;
    public const int Ok = 0;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void Progress(IntPtr user, long iteration, double objective, int status);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_abi_version();

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qenivo_model_create(
        int rows, int cols,
        int[] rowPtr, int[]? colIdx, double[]? values,
        double[] cost, double[]? rowLower, double[]? rowUpper,
        double[] colLower, double[] colUpper, int[]? integrality,
        int[]? qRowPtr, int[]? qColIdx, double[]? qValues, out int error);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern void qenivo_model_free(IntPtr model);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qenivo_create_error_message();

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr qenivo_last_error_message(IntPtr model);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_solve(IntPtr model, string engine);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_result_status(IntPtr model);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_result_objective(IntPtr model, out double objective);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_result_x(IntPtr model, double[] x, int cols);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_result_reduced_costs(IntPtr model, double[] reduced, int cols);

    [DllImport("qenivo_c", CallingConvention = CallingConvention.Cdecl)]
    public static extern int qenivo_result_basis(IntPtr model, int[] columnStatus, int cols, int[]? rowStatus, int rows);

    public static string Message(IntPtr ptr) => Marshal.PtrToStringUTF8(ptr) ?? "";
}

public sealed class Model : IDisposable
{
    readonly IntPtr handle;

    Model(IntPtr handle) { this.handle = handle; }

    public static Model Create(int rows, int cols, int[] rowPtr, int[]? colIdx, double[]? values,
        double[] cost, double[]? rowLower, double[]? rowUpper, double[] colLower, double[] colUpper,
        int[]? integrality = null)
    {
        IntPtr ptr = Abi.qenivo_model_create(rows, cols, rowPtr, colIdx, values, cost, rowLower, rowUpper,
            colLower, colUpper, integrality, null, null, null, out int error);
        if (ptr == IntPtr.Zero)
            throw new InvalidOperationException("create failed: " + Abi.Message(Abi.qenivo_create_error_message()));
        return new Model(ptr);
    }

    public int Solve(string engine)
    {
        int rc = Abi.qenivo_solve(handle, engine);
        if (rc != Abi.Ok)
            throw new InvalidOperationException(Abi.Message(Abi.qenivo_last_error_message(handle)));
        return Abi.qenivo_result_status(handle);
    }

    public double Objective()
    {
        if (Abi.qenivo_result_objective(handle, out double obj) != Abi.Ok)
            throw new InvalidOperationException("no objective");
        return obj;
    }

    public double[] Primal(int cols)
    {
        double[] x = new double[cols];
        if (Abi.qenivo_result_x(handle, x, cols) != Abi.Ok)
            throw new InvalidOperationException("no primal");
        return x;
    }

    public void Dispose() => Abi.qenivo_model_free(handle);
}
