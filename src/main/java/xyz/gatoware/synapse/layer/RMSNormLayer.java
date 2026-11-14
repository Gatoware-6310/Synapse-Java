package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.TransformerMath;

/** Per-token RMSNorm, or LayerNorm when constructed with center=true. */
public class RMSNormLayer extends TrainableLayer {
    final Matrix scale, bias;
    final float epsilon;
    final boolean center;
    private Matrix normalized;
    private float[] inverse;
    private Matrix deviceInverse;

    public RMSNormLayer(int size) {
        this(size, 1e-5f, false);
    }

    public RMSNormLayer(int size, float epsilon, boolean center) {
        if (size < 1 || !(epsilon > 0) || !Float.isFinite(epsilon))
            throw new IllegalArgumentException();
        this.epsilon = epsilon;
        this.center = center;
        scale = parameter(new Matrix(size, 1));
        for (float[] r : scale.values) r[0] = 1;
        bias = center ? parameter(new Matrix(size, 1)) : null;
    }

    @Override
    public Matrix forward(Matrix x) {
        if (x.rows() != scale.rows() || x.columns() < 1) throw new IllegalArgumentException();
        if (TransformerMath.gpu() != null) {
            Matrix[] y = TransformerMath.gpu().norm(x, scale, bias, epsilon);
            normalized = y[1];
            deviceInverse = y[2];
            return y[0];
        }
        TransformerMath.host(x);
        TransformerMath.host(scale);
        if (center) TransformerMath.host(bias);
        normalized = new Matrix(x.rows(), x.columns());
        inverse = new float[x.columns()];
        Matrix y = new Matrix(x.rows(), x.columns());
        for (int t = 0; t < x.columns(); t++) {
            double mean = 0, ss = 0;
            if (center) {
                for (int i = 0; i < x.rows(); i++) mean += x.values[i][t];
                mean /= x.rows();
            }
            for (int i = 0; i < x.rows(); i++) {
                double v = x.values[i][t] - mean;
                ss += v * v;
            }
            inverse[t] = (float) (1 / Math.sqrt(ss / x.rows() + epsilon));
            for (int i = 0; i < x.rows(); i++) {
                float v = (float) ((x.values[i][t] - mean) * inverse[t]);
                normalized.values[i][t] = v;
                y.values[i][t] = v * scale.values[i][0] + (center ? bias.values[i][0] : 0);
            }
        }
        return y;
    }

    @Override
    public Matrix backwardGradient(Matrix g) {
        TransformerMath.shape(g, normalized);
        if (TransformerMath.gpu() != null)
            return TransformerMath.gpu()
                    .normBackward(
                            g,
                            normalized,
                            deviceInverse,
                            scale,
                            gradients.get(0),
                            center ? gradients.get(1) : null);
        Matrix dx = new Matrix(g.rows(), g.columns());
        for (int t = 0; t < g.columns(); t++) {
            double dot = 0, sum = 0;
            for (int i = 0; i < g.rows(); i++) {
                float v = g.values[i][t] * scale.values[i][0];
                dot += v * normalized.values[i][t];
                sum += v;
            }
            for (int i = 0; i < g.rows(); i++) {
                float n = normalized.values[i][t];
                dx.values[i][t] =
                        (float)
                                (inverse[t]
                                        * (g.values[i][t] * scale.values[i][0]
                                                - n * dot / g.rows()
                                                - (center ? sum / g.rows() : 0)));
                gradients.get(0).values[i][0] += g.values[i][t] * n;
                if (center) gradients.get(1).values[i][0] += g.values[i][t];
            }
        }
        for (Matrix d : gradients) d.markDirty();
        return dx;
    }

    @Override
    public void clearTrainingState() {
        normalized = null;
        inverse = null;
        deviceInverse = null;
    }
}
