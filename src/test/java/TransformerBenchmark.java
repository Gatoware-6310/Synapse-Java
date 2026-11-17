import xyz.gatoware.synapse.*;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.AdamW;
import xyz.gatoware.synapse.transformer.*;

import java.util.*;

/**
 * Matched dense/MoE benchmark. Timings include Java orchestration and synchronize CUDA explicitly.
 */
public final class TransformerBenchmark {
    interface Work {
        void run();
    }

    static void sync() {
        if (Synapse.getDevice() == Devices.CUDA) jcuda.runtime.JCuda.cudaDeviceSynchronize();
    }

    static double measure(Work work, int warmup, int repeats) {
        for (int i = 0; i < warmup; i++) work.run();
        sync();
        double[] samples = new double[repeats];
        for (int i = 0; i < repeats; i++) {
            sync();
            long start = System.nanoTime();
            work.run();
            sync();
            samples[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2];
    }

    public static void main(String[] a) throws Exception {
        // device moe sequence vocabulary hidden layers heads kv ffn warmup repeats decode precision
        if (a.length != 13)
            throw new IllegalArgumentException("Expected 13 arguments; see benchmarks/README.md");
        Synapse.useDevice(Devices.valueOf(a[0].toUpperCase()));
        boolean moe = Boolean.parseBoolean(a[1]);
        int seq = Integer.parseInt(a[2]),
                vocab = Integer.parseInt(a[3]),
                hidden = Integer.parseInt(a[4]),
                layers = Integer.parseInt(a[5]),
                heads = Integer.parseInt(a[6]),
                kv = Integer.parseInt(a[7]),
                ffn = Integer.parseInt(a[8]),
                warmup = Integer.parseInt(a[9]),
                repeats = Integer.parseInt(a[10]),
                decode = Integer.parseInt(a[11]);
        if (seq < 1 || decode < 1 || warmup < 1 || repeats < 3)
            throw new IllegalArgumentException("Require sequence/decode/warmup>0 and repeats>=3");
        Precision precision = Precision.valueOf(a[12]);
        var c =
                new TransformerConfig(vocab, hidden, layers, heads)
                        .setKVHeads(kv)
                        .setFFNSize(ffn)
                        .setContextLength(seq + decode + 1)
                        .setActivation(Activations.SWIGLU)
                        .setPrecision(precision);
        if (moe) c.setMoE(new MoEConfig(4, 2).setSharedExperts(1).setLoadBalancing(.01f));
        Transformer m = new Transformer(c);
        Random initialization = new Random(7);
        for (Matrix parameter : m.parameters()) {
            for (float[] row : parameter.values)
                for (int j = 0; j < row.length; j++)
                    row[j] =
                            parameter.columns() == 1
                                    ? 1f
                                    : (float)
                                            ((initialization.nextDouble() * 2 - 1)
                                                    / Math.sqrt(parameter.columns()));
            parameter.markDirty();
        }
        String weightsPath = System.getenv("SYNAPSE_BENCH_WEIGHTS_PATH");
        if (weightsPath != null) m.save(weightsPath);
        int[] tokens = new int[seq], training = new int[seq + 1];
        for (int i = 0; i < training.length; i++) training[i] = i % vocab;
        System.arraycopy(training, 0, tokens, 0, seq);
        Matrix delta = new Matrix(vocab, seq);
        for (float[] row : delta.values) Arrays.fill(row, 1f / (vocab * seq));
        AdamW optimizer = new AdamW();
        float initialLoss = m.loss(training);
        double prefill = measure(() -> m.forward(tokens), warmup, repeats);
        double[] back = new double[repeats];
        for (int i = -warmup; i < repeats; i++) {
            m.forward(tokens);
            m.zeroGrad();
            sync();
            long start = System.nanoTime();
            m.backwardGradient(delta);
            sync();
            if (i >= 0) back[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(back);
        double backward = back[repeats / 2];
        double trainingMs =
                measure(
                        () -> m.fit(new int[][] {training}, 1, .0003f, optimizer, 1, 1),
                        warmup,
                        repeats);
        double[] decoding = new double[repeats];
        for (int i = -warmup; i < repeats; i++) {
            m.resetCache();
            m.forwardCached(tokens);
            sync();
            long start = System.nanoTime();
            for (int t = 0; t < decode; t++) m.forwardCached(new int[] {t % vocab});
            sync();
            if (i >= 0) decoding[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(decoding);
        double decodeMs = decoding[repeats / 2];
        m.resetCache();
        String deviceInfo =
                Synapse.backend() instanceof CudaBackend cuda
                        ? cuda.transformerDeviceInfo()
                        : System.getProperty("os.arch") + " " + System.getProperty("java.version");
        long peak =
                Synapse.backend() instanceof CudaBackend cuda ? cuda.getPeakAllocatedBytes() : 0;
        System.out.printf(
                Locale.ROOT,
                "{\"framework\":\"synapse\",\"device\":\"%s\",\"precision\":\"%s\",\"moe\":%s,\"sequence\":%d,\"vocabulary\":%d,\"hidden\":%d,\"layers\":%d,\"heads\":%d,\"kv_heads\":%d,\"ffn\":%d,\"batch\":1,\"decode_tokens\":%d,\"warmup\":%d,\"repeats\":%d,\"parameters\":%d,\"initial_loss\":%.9f,\"prefill_ms\":%.6f,\"backward_ms\":%.6f,\"train_ms\":%.6f,\"decode_ms\":%.6f,\"prefill_tokens_per_second\":%.3f,\"training_tokens_per_second\":%.3f,\"decode_tokens_per_second\":%.3f,\"owned_peak_cuda_bytes\":%d,\"environment\":\"%s\"}%n",
                a[0],
                precision,
                moe,
                seq,
                vocab,
                hidden,
                layers,
                heads,
                kv,
                ffn,
                decode,
                warmup,
                repeats,
                m.parameterCount(),
                initialLoss,
                prefill,
                backward,
                trainingMs,
                decodeMs,
                seq * 1000 / prefill,
                seq * 1000 / trainingMs,
                decode * 1000 / decodeMs,
                peak,
                deviceInfo.replace("\"", "'"));
    }
}
