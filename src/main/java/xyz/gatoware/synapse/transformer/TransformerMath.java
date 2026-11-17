package xyz.gatoware.synapse.transformer;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.backend.CudaTransformerOps;
import xyz.gatoware.synapse.matrix.Matrix;

/** Matrix utilities shared by transformer layers; products use the selected Synapse backend. */
public final class TransformerMath {
    private TransformerMath() {}

    private static final ThreadLocal<Precision> PRECISION =
            ThreadLocal.withInitial(() -> Precision.FP32);

    public static Precision precision() {
        return PRECISION.get();
    }

    public static Precision setPrecision(Precision p) {
        Precision old = PRECISION.get();
        PRECISION.set(p);
        return old;
    }

    public static CudaTransformerOps gpu() {
        return Synapse.backend() instanceof CudaBackend c ? c.transformerOps() : null;
    }

    public static Matrix scale(Matrix m, float f) {
        return gpu() != null ? gpu().scale(m, f) : m.copy().multiply(f);
    }

    public static void host(Matrix m) {
        if (Synapse.backend() instanceof CudaBackend c) c.materialize(m);
    }

    public static Matrix transpose(Matrix x) {
        if (gpu() != null) return gpu().transpose(x);
        host(x);
        Matrix y = new Matrix(x.columns(), x.rows());
        for (int i = 0; i < x.rows(); i++)
            for (int j = 0; j < x.columns(); j++) y.values[j][i] = x.values[i][j];
        return y;
    }

    public static Matrix mm(Matrix a, Matrix b) {
        if (a.columns() != b.rows()) throw new IllegalArgumentException("Incompatible product");
        if (gpu() != null) return gpu().mm(a, b);
        return new Matrix(
                Synapse.backend().multiply(a.values, b.values, a.rows(), a.columns(), b.columns()));
    }

    public static Matrix add(Matrix a, Matrix b) {
        if (a.rows() != b.rows() || a.columns() != b.columns())
            throw new IllegalArgumentException();
        if (gpu() != null) return gpu().add(a, b);
        host(a);
        host(b);
        if (a.rows() != b.rows() || a.columns() != b.columns())
            throw new IllegalArgumentException("Incompatible sum");
        Matrix y = new Matrix(a.rows(), a.columns());
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < a.columns(); j++) y.values[i][j] = a.values[i][j] + b.values[i][j];
        return y;
    }

    public static Matrix random(int r, int c) {
        Matrix m = new Matrix(r, c);
        double s = 1 / Math.sqrt(c);
        for (float[] row : m.values)
            for (int j = 0; j < c; j++) row[j] = (float) ((Math.random() * 2 - 1) * s);
        return m;
    }

    public static void accumulate(Matrix a, Matrix b) {
        if (gpu() != null) {
            gpu().accumulate(a, b);
            return;
        }
        host(a);
        host(b);
        for (int i = 0; i < a.rows(); i++)
            for (int j = 0; j < a.columns(); j++) a.values[i][j] += b.values[i][j];
        a.markDirty();
    }

    public static void shape(Matrix g, Matrix y) {
        if (y == null) throw new IllegalStateException("Forward must precede backward");
        if (g.rows() != y.rows() || g.columns() != y.columns())
            throw new IllegalArgumentException("Gradient shape mismatch");
    }
}
