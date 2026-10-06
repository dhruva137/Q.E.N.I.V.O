package qenivo;

/** min -x on 0 <= x <= 1, then min -x-2y, x+y <= 1, both binary. */
public final class Example {
    public static void main(String[] args) {
        if (args.length > 0) Qenivo.setPackageDir(args[0]);
        int[] rp0 = {0};
        long bound = Qenivo.create(0, 1, rp0, null, null, new double[] {-1.0}, null, null,
                new double[] {0.0}, new double[] {1.0}, null);
        int status = Qenivo.solve(bound, "simplex");
        double obj = Qenivo.objective(bound);
        double[] x = Qenivo.primal(bound);
        System.out.println("optimum " + obj + " x " + x[0]);
        Qenivo.dispose(bound);
        if (status != 0 || Math.abs(obj + 1.0) > 1e-8 || Math.abs(x[0] - 1.0) > 1e-8)
            System.exit(1);

        long milp = Qenivo.create(1, 2, new int[] {0, 2}, new int[] {0, 1}, new double[] {1.0, 1.0},
                new double[] {-1.0, -2.0}, new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0},
                new double[] {0.0, 0.0}, new double[] {1.0, 1.0}, new int[] {1, 1});
        status = Qenivo.solve(milp, "milp");
        obj = Qenivo.objective(milp);
        x = Qenivo.primal(milp);
        System.out.println("milp " + obj + " x " + x[0] + " " + x[1]);
        Qenivo.dispose(milp);
        if (status != 0 || Math.abs(obj + 2.0) > 1e-6 || Math.abs(x[0]) > 1e-6 || Math.abs(x[1] - 1.0) > 1e-6)
            System.exit(1);
    }
}
