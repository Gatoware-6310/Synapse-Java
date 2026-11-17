import xyz.gatoware.synapse.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.AdamW;
import xyz.gatoware.synapse.transformer.*;

/**
 * Explicit hardware acceptance check. This command fails if CUDA is unavailable; no skipped cases.
 */
public final class CudaTransformerCheck {
    static void compare(Matrix a, Matrix b, float tolerance) {
        TransformerMath.host(a);
        TransformerMath.host(b);
        if (a.rows() != b.rows() || a.columns() != b.columns()) throw new AssertionError("shape");
        for (int r = 0; r < a.rows(); r++)
            for (int c = 0; c < a.columns(); c++) {
                float av = a.values[r][c], bv = b.values[r][c];
                if (!Float.isFinite(bv) || Math.abs(av - bv) > tolerance * (1 + Math.abs(av)))
                    throw new AssertionError(
                            "CPU/CUDA mismatch: " + av + " vs " + bv + " [" + r + "," + c + "]");
            }
    }

    public static void main(String[] args) throws Exception {
        if (!Synapse.isDeviceAvailable(Devices.CUDA))
            throw new IllegalStateException(
                    "CUDA hardware/libraries unavailable: acceptance checks were NOT run");
        int cases = 0;
        for (var precision : Precision.values())
            for (int kv : new int[] {1, 2, 4})
                for (boolean moe : new boolean[] {false, true})
                    for (var position : PositionEncoding.values())
                        for (boolean checkpoint : new boolean[] {false, true}) {
                            Synapse.useDevice(Devices.CPU);
                            var cfg =
                                    new TransformerConfig(7, 8, 2, 4)
                                            .setKVHeads(kv)
                                            .setFFNSize(12)
                                            .setContextLength(16)
                                            .setActivation(Activations.SWIGLU)
                                            .setPositionEncoding(position)
                                            .setNormalization(
                                                    cases % 2 == 0
                                                            ? Normalization.RMS_NORM
                                                            : Normalization.LAYER_NORM)
                                            .setPrecision(precision)
                                            .setKVCachePrecision(
                                                    cases % 2 == 0
                                                            ? KVCachePrecision.FP16
                                                            : KVCachePrecision.FP32)
                                            .setActivationCheckpointing(checkpoint);
                            if (moe)
                                cfg.setMoE(
                                        new MoEConfig(3, 2)
                                                .setRouter(
                                                        cases % 2 == 0
                                                                ? RouterType.TOP_K_SOFTMAX
                                                                : RouterType.TOP_K_SIGMOID)
                                                .setSharedExperts(1)
                                                .setLoadBalancing(.01f)
                                                .setCapacityFactor(cases % 3 == 0 ? .5f : 2f));
                            Transformer cpu = new Transformer(cfg), gpu = new Transformer(cfg);
                            var a = cpu.parameters();
                            var b = gpu.parameters();
                            for (int p = 0; p < a.size(); p++)
                                for (int r = 0; r < a.get(p).rows(); r++)
                                    System.arraycopy(
                                            a.get(p).values[r],
                                            0,
                                            b.get(p).values[r],
                                            0,
                                            a.get(p).columns());
                            int[] tokens = {0, 1, 2, 3};
                            Matrix expected = cpu.forward(tokens).copy(),
                                    delta = TransformerLayerTest.input(7, 4);
                            cpu.zeroGrad();
                            cpu.backwardGradient(delta);
                            float loss = cpu.loss(new int[] {0, 1, 2, 3, 4});
                            Synapse.useDevice(Devices.CUDA);
                            float tolerance = precision == Precision.FP32 ? .002f : .06f;
                            compare(expected, gpu.forward(tokens), tolerance);
                            gpu.zeroGrad();
                            gpu.backwardGradient(delta);
                            for (int p = 0; p < cpu.gradients().size(); p++)
                                compare(cpu.gradients().get(p), gpu.gradients().get(p), tolerance);
                            float actualLoss = gpu.loss(new int[] {0, 1, 2, 3, 4});
                            if (Math.abs(loss - actualLoss) > tolerance)
                                throw new AssertionError("Cross entropy mismatch");
                            Matrix full = gpu.forward(tokens);
                            gpu.resetCache();
                            for (int t = 0; t < tokens.length; t++) {
                                Matrix cached = gpu.forwardCached(new int[] {tokens[t]});
                                TransformerMath.host(full);
                                TransformerMath.host(cached);
                                for (int i = 0; i < 7; i++)
                                    if (Math.abs(full.values[i][t] - cached.values[i][0])
                                            > tolerance)
                                        throw new AssertionError("KV cache mismatch");
                            }
                            gpu.fit(
                                    new int[][] {{0, 1, 2, 3, 4, 0, 1, 2}},
                                    2,
                                    .001f,
                                    new AdamW(),
                                    1,
                                    1);
                            if (!Float.isFinite(gpu.getLastLoss()))
                                throw new AssertionError("nonfinite training loss");
                            cases++;
                        }
        Synapse.useDevice(Devices.CPU);
        System.out.println("CUDA transformer acceptance passed: " + cases + " configurations");
    }
}
