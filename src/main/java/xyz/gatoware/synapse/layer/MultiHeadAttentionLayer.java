package xyz.gatoware.synapse.layer;

import static xyz.gatoware.synapse.transformer.TransformerMath.*;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.*;

/**
 * Causal MHA/GQA/MQA. Attention probabilities are recomputed during backward (linear workspace).
 */
public final class MultiHeadAttentionLayer extends TrainableLayer {
    private final Matrix wq, wk, wv, wo;
    final int hidden, heads, kvHeads, dim, context;
    final boolean rope;
    final boolean halfCache;
    private xyz.gatoware.synapse.backend.CudaTransformerOps.KVCache deviceCache;
    private Matrix input, q, k, v, attended, output, cacheK, cacheV;
    private int offset, cacheLength;
    private boolean cachedForward;

    public MultiHeadAttentionLayer(TransformerConfig c) {
        c.validate();
        hidden = c.hiddenSize;
        heads = c.heads;
        kvHeads = c.kvHeads;
        dim = hidden / heads;
        context = c.contextLength;
        halfCache = c.kvCachePrecision == KVCachePrecision.FP16;
        rope = c.positionEncoding == PositionEncoding.ROPE;
        wq = parameter(random(hidden, hidden));
        wk = parameter(random(kvHeads * dim, hidden));
        wv = parameter(random(kvHeads * dim, hidden));
        wo = parameter(random(hidden, hidden));
    }

    public void resetCache() {
        if (deviceCache != null) {
            deviceCache.close();
            deviceCache = null;
        }
        cacheLength = 0;
        cacheK = null;
        cacheV = null;
    }

    public int cacheLength() {
        return cacheLength;
    }

    @Override
    public Matrix forward(Matrix x) {
        return run(x, false);
    }

    /** Appends a prefill chunk or decode token to this layer's independent KV cache. */
    public Matrix forwardCached(Matrix x) {
        return run(x, true);
    }

    private Matrix run(Matrix x, boolean cached) {
        if (x.rows() != hidden
                || x.columns() < 1
                || x.columns() + (cached ? cacheLength : 0) > context)
            throw new IllegalArgumentException("Attention shape/context exceeded");
        input = x;
        cachedForward = cached;
        offset = cached ? cacheLength : 0;
        q = mm(wq, x);
        Matrix nk = mm(wk, x), nv = mm(wv, x);
        if (rope) {
            rotate(q, heads, offset, false);
            rotate(nk, kvHeads, offset, false);
        }
        if (cached && gpu() != null) {
            if (deviceCache == null)
                deviceCache = gpu().newKVCache(kvHeads * dim, context, halfCache);
            deviceCache.append(nk, nv, offset);
            cacheLength = offset + x.columns();
            attended = gpu().attentionCached(q, deviceCache, heads, kvHeads, dim, offset);
            return output = mm(wo, attended);
        } else if (cached) {
            int n = offset + x.columns();
            if (cacheK == null) {
                cacheK = new Matrix(kvHeads * dim, context);
                cacheV = new Matrix(kvHeads * dim, context);
            }
            if (halfCache)
                for (int i = 0; i < nk.rows(); i++)
                    for (int t = 0; t < nk.columns(); t++) {
                        nk.values[i][t] =
                                Float.float16ToFloat(Float.floatToFloat16(nk.values[i][t]));
                        nv.values[i][t] =
                                Float.float16ToFloat(Float.floatToFloat16(nv.values[i][t]));
                    }
            for (int i = 0; i < nk.rows(); i++) {
                System.arraycopy(nk.values[i], 0, cacheK.values[i], offset, x.columns());
                System.arraycopy(nv.values[i], 0, cacheV.values[i], offset, x.columns());
            }
            k = new Matrix(kvHeads * dim, n);
            v = new Matrix(kvHeads * dim, n);
            for (int i = 0; i < k.rows(); i++) {
                System.arraycopy(cacheK.values[i], 0, k.values[i], 0, n);
                System.arraycopy(cacheV.values[i], 0, v.values[i], 0, n);
            }
            cacheLength = n;
        } else {
            k = nk;
            v = nv;
        }
        if (gpu() != null) {
            attended = gpu().attention(q, k, v, heads, kvHeads, dim, offset);
            return output = mm(wo, attended);
        }
        attended = new Matrix(hidden, x.columns());
        double[] p = new double[k.columns()];
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < x.columns(); t++) {
                int kh = h / (heads / kvHeads);
                probabilities(h, kh, t, p);
                for (int d = 0; d < dim; d++) {
                    double sum = 0;
                    for (int s = 0; s <= offset + t; s++) sum += p[s] * v.values[kh * dim + d][s];
                    attended.values[h * dim + d][t] = (float) sum;
                }
            }
        return output = mm(wo, attended);
    }

    private void probabilities(int h, int kh, int t, double[] p) {
        double max = -Double.MAX_VALUE, scale = 1 / Math.sqrt(dim);
        for (int s = 0; s <= offset + t; s++) {
            double z = 0;
            for (int d = 0; d < dim; d++) z += q.values[h * dim + d][t] * k.values[kh * dim + d][s];
            p[s] = z * scale;
            max = Math.max(max, p[s]);
        }
        double sum = 0;
        for (int s = 0; s <= offset + t; s++) {
            p[s] = Math.exp(p[s] - max);
            sum += p[s];
        }
        for (int s = 0; s <= offset + t; s++) p[s] /= sum;
    }

    private void rotate(Matrix x, int count, int start, boolean inverse) {
        if (gpu() != null) {
            gpu().rope(x, count, dim, start, inverse);
            return;
        }
        for (int h = 0; h < count; h++)
            for (int d = 0; d < dim; d += 2)
                for (int t = 0; t < x.columns(); t++) {
                    double angle =
                            (start + t) / Math.pow(10000, (double) d / dim) * (inverse ? -1 : 1);
                    float a = x.values[h * dim + d][t], b = x.values[h * dim + d + 1][t];
                    x.values[h * dim + d][t] = (float) (a * Math.cos(angle) - b * Math.sin(angle));
                    x.values[h * dim + d + 1][t] =
                            (float) (a * Math.sin(angle) + b * Math.cos(angle));
                }
        x.markDirty();
    }

    @Override
    public Matrix backwardGradient(Matrix g) {
        shape(g, output);
        if (cachedForward)
            throw new IllegalStateException("Training requires an uncached forward pass");
        Matrix da = mm(transpose(wo), g);
        accumulate(gradients.get(3), mm(g, transpose(attended)));
        if (gpu() != null) {
            Matrix[] d = gpu().attentionBackward(q, k, v, attended, da, heads, kvHeads, dim);
            if (rope) {
                rotate(d[0], heads, 0, true);
                rotate(d[1], kvHeads, 0, true);
            }
            for (int i = 0; i < 3; i++) accumulate(gradients.get(i), mm(d[i], transpose(input)));
            return add(
                    add(mm(transpose(wq), d[0]), mm(transpose(wk), d[1])), mm(transpose(wv), d[2]));
        }
        Matrix dq = new Matrix(hidden, q.columns()),
                dk = new Matrix(k.rows(), k.columns()),
                dv = new Matrix(v.rows(), v.columns());
        double[] p = new double[k.columns()], dp = new double[k.columns()];
        double scale = 1 / Math.sqrt(dim);
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < q.columns(); t++) {
                int kh = h / (heads / kvHeads);
                probabilities(h, kh, t, p);
                double dot = 0;
                for (int s = 0; s <= t; s++) {
                    double z = 0;
                    for (int d = 0; d < dim; d++) {
                        z += da.values[h * dim + d][t] * v.values[kh * dim + d][s];
                        dv.values[kh * dim + d][s] += p[s] * da.values[h * dim + d][t];
                    }
                    dp[s] = z;
                    dot += p[s] * z;
                }
                for (int s = 0; s <= t; s++) {
                    double ds = p[s] * (dp[s] - dot) * scale;
                    for (int d = 0; d < dim; d++) {
                        dq.values[h * dim + d][t] += ds * k.values[kh * dim + d][s];
                        dk.values[kh * dim + d][s] += ds * q.values[h * dim + d][t];
                    }
                }
            }
        if (rope) {
            rotate(dq, heads, 0, true);
            rotate(dk, kvHeads, 0, true);
        }
        accumulate(gradients.get(0), mm(dq, transpose(input)));
        accumulate(gradients.get(1), mm(dk, transpose(input)));
        accumulate(gradients.get(2), mm(dv, transpose(input)));
        return add(add(mm(transpose(wq), dq), mm(transpose(wk), dk)), mm(transpose(wv), dv));
    }

    @Override
    public void clearTrainingState() {
        input = null;
        q = null;
        k = null;
        v = null;
        attended = null;
        output = null;
    }
}
