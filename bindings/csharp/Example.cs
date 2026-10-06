// Example: min -x subject to 0 <= x <= 1 (optimum -1), then the two-binary MILP
// min -x-2y s.t. x+y <= 1 (optimum -2 at (0,1)).
// argv[0] is the path of qenivo_c.dll / libqenivo_c.so. The .NET runtime must
// load that library before the P/Invoke calls; NativeLibrary.Load does it.
using System.Runtime.InteropServices;
using Qenivo;

string dll = args.Length > 0 ? args[0] : "qenivo_c";
NativeLibrary.SetDllImportResolver(typeof(Abi).Assembly, (name, assembly, path) =>
    name == "qenivo_c" ? NativeLibrary.Load(dll) : IntPtr.Zero);

using (Model bound = Model.Create(0, 1, new[] { 0 }, null, null, new[] { -1.0 }, null, null, new[] { 0.0 }, new[] { 1.0 }))
{
    int status = bound.Solve("simplex");
    double obj = bound.Objective();
    double[] x = bound.Primal(1);
    Console.WriteLine($"optimum {obj} x {x[0]} status {status}");
    if (status != Abi.Optimal || Math.Abs(obj + 1.0) > 1e-8 || Math.Abs(x[0] - 1.0) > 1e-8)
        return 1;
}

if (args.Length > 1)
    Environment.SetEnvironmentVariable("QENIVO_SRC", args[1]);

using (Model milp = Model.Create(1, 2, new[] { 0, 2 }, new[] { 0, 1 }, new[] { 1.0, 1.0 },
    new[] { -1.0, -2.0 }, new[] { double.NegativeInfinity }, new[] { 1.0 },
    new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }, new[] { 1, 1 }))
{
    int status = milp.Solve("milp");
    double obj = milp.Objective();
    double[] x = milp.Primal(2);
    Console.WriteLine($"milp {obj} x {x[0]} {x[1]} status {status}");
    if (status != Abi.Optimal || Math.Abs(obj + 2.0) > 1e-6 || Math.Abs(x[0]) > 1e-6 || Math.Abs(x[1] - 1.0) > 1e-6)
        return 1;
}
return 0;
