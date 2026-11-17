package xyz.gatoware.synapse;

import static xyz.gatoware.synapse.transformer.TransformerMath.*;

import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.*;
import xyz.gatoware.synapse.transformer.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Decoder-only language model using Synapse layers, matrices, backpropagation and optimizers. Model
 * instances have mutable forward/cache state and must not be used concurrently.
 */
public final class Transformer extends TrainableLayer {
    private final TransformerConfig config;
    private final EmbeddingLayer embedding;
    private final TransformerDecoderLayer[] blocks;
    private final RMSNormLayer finalNorm;
    private final Matrix positions, head;
    private Matrix hidden, logits;
    private Matrix[] checkpointInputs;
    private int positionOffset;
    private boolean cachedForward;
    private boolean trainingMode;
    private float lastLoss = Float.NaN;

    public Transformer(TransformerConfig requested) {
        config = requested.copy();
        config.validate();
        long estimate = estimatedParameterCount(config) * 12;
        if (estimate > Runtime.getRuntime().maxMemory() * 0.8)
            throw new IllegalArgumentException(
                    "Model requires approximately "
                            + estimate
                            + " bytes for FP32 parameters/gradients with host row storage,"
                            + " exceeding JVM heap budget. Reduce dimensions or increase -Xmx.");
        embedding = new EmbeddingLayer(config.vocabularySize, config.hiddenSize);
        children.add(embedding);
        positions =
                config.positionEncoding == PositionEncoding.LEARNED
                        ? parameter(random(config.hiddenSize, config.contextLength))
                        : null;
        head =
                config.tieEmbeddings
                        ? embedding.getWeights()
                        : parameter(random(config.hiddenSize, config.vocabularySize));
        blocks = new TransformerDecoderLayer[config.layerCount];
        for (int i = 0; i < blocks.length; i++) {
            blocks[i] = new TransformerDecoderLayer(config, i);
            children.add(blocks[i]);
        }
        finalNorm =
                config.normalization == Normalization.RMS_NORM
                        ? new RMSNormLayer(config.hiddenSize)
                        : new LayerNormLayer(config.hiddenSize);
        children.add(finalNorm);
    }

    @Override
    public void setTraining(boolean value) {
        trainingMode = value;
        super.setTraining(value);
    }

    public TransformerConfig getConfig() {
        return config.copy();
    }

    public float auxiliaryLoss() {
        float total = 0;
        for (var b : blocks) total += b.auxiliaryLoss();
        return total;
    }

    public float getLastLoss() {
        return lastLoss;
    }

    public static long estimatedParameterCount(TransformerConfig c) {
        long h = c.hiddenSize,
                f = c.ffnSize,
                k = (long) c.kvHeads * h / c.heads,
                ff = (c.activation == Activations.SWIGLU ? 3 : 2) * h;
        long n =
                h * c.vocabularySize * (c.tieEmbeddings ? 1 : 2)
                        + (c.positionEncoding == PositionEncoding.LEARNED ? h * c.contextLength : 0)
                        + (c.normalization == Normalization.RMS_NORM ? h : 2 * h);
        for (int i = 0; i < c.layerCount; i++) {
            n += 2 * h * h + 2 * h * k + (c.normalization == Normalization.RMS_NORM ? 2 : 4) * h;
            if (c.moe != null && c.moe.includes(i)) {
                n +=
                        h * c.moe.experts
                                + ff
                                        * (c.moe.expertHiddenSize > 0 ? c.moe.expertHiddenSize : f)
                                        * (c.moe.experts + c.moe.sharedExperts);
            } else n += ff * f;
        }
        return n;
    }

    public long parameterCount() {
        long n = 0;
        for (Matrix p : parameters()) n += (long) p.rows() * p.columns();
        return n;
    }

    /** FP32 payload estimate, excluding Java object overhead/workspaces. */
    public long estimatedMemoryBytes(int sequenceLength, boolean training) {
        if (sequenceLength < 1 || sequenceLength > config.contextLength)
            throw new IllegalArgumentException();
        long n = parameterCount() * (training ? 16 : 4);
        long activations =
                (long) sequenceLength
                        * (config.hiddenSize * 8L + config.ffnSize * 3L + config.vocabularySize)
                        * 4;
        n += activations * (training && !config.activationCheckpointing ? config.layerCount : 1);
        n +=
                (long) config.layerCount
                        * config.contextLength
                        * config.kvHeads
                        * (config.hiddenSize / config.heads)
                        * (Synapse.getDevice()==Devices.CUDA && config.kvCachePrecision == KVCachePrecision.FP16 ? 4 : 8);
        return n;
    }

    public void resetCache() {
        for (var b : blocks) b.resetCache();
        positionOffset = 0;
    }

    public Matrix forward(int[] tokens) {
        return forward(tokenMatrix(tokens));
    }

    private Matrix tokenMatrix(int[] tokens) {
        if (tokens == null || tokens.length < 1)
            throw new IllegalArgumentException("Nonempty token sequence required");
        Matrix x = new Matrix(1, tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i] < 0 || tokens[i] >= config.vocabularySize)
                throw new IllegalArgumentException("Token outside vocabulary");
            x.values[0][i] = tokens[i];
        }
        return x;
    }

    @Override
    public Matrix forward(Matrix tokens) {
        return run(tokens, false);
    }

    public Matrix forwardCached(int[] tokens) {
        return run(tokenMatrix(tokens), true);
    }

    private Matrix run(Matrix tokens, boolean cached) {
        Precision old = setPrecision(config.precision);
        try {
            return runInternal(tokens, cached);
        } finally {
            setPrecision(old);
        }
    }

    private Matrix runInternal(Matrix tokens, boolean cached) {
        for (var block : blocks) block.setTraining(trainingMode && !cached);
        int start = cached ? positionOffset : 0;
        if (tokens.columns() < 1 || (long) start + tokens.columns() > config.contextLength)
            throw new IllegalArgumentException("Sequence exceeds configured context length");
        cachedForward = cached;
        Matrix x = embedding.forward(tokens);
        if (positions != null && gpu() != null) {
            x = add(x, gpu().slice(positions, start, x.columns()));
        } else if (positions != null) {
            host(positions);
            for (int i = 0; i < x.rows(); i++)
                for (int t = 0; t < x.columns(); t++)
                    x.values[i][t] += positions.values[i][start + t];
            x.markDirty();
        }
        checkpointInputs =
                config.activationCheckpointing && !cached ? new Matrix[blocks.length] : null;
        for (int i = 0; i < blocks.length; i++) {
            if (checkpointInputs != null) checkpointInputs[i] = x;
            x = cached ? blocks[i].forwardCached(x) : blocks[i].forward(x);
            if (checkpointInputs != null) blocks[i].clearTrainingState();
        }
        hidden = finalNorm.forward(x);
        logits = mm(transpose(head), hidden);
        if (cached) positionOffset += tokens.columns();
        return logits;
    }

    @Override
    public Matrix backwardGradient(Matrix g) {
        Precision old = setPrecision(config.precision);
        try {
            return backwardInternal(g);
        } finally {
            setPrecision(old);
        }
    }

    private Matrix backwardInternal(Matrix g) {
        shape(g, logits);
        if (cachedForward)
            throw new IllegalStateException("Training requires an uncached forward pass");
        Matrix dh = mm(head, g), dw = mm(hidden, transpose(g));
        int headIndex = positions == null ? 0 : 1;
        accumulate(
                config.tieEmbeddings ? embedding.gradients().get(0) : gradients.get(headIndex), dw);
        Matrix dx = finalNorm.backwardGradient(dh);
        for (int i = blocks.length - 1; i >= 0; i--) {
            if (checkpointInputs != null) blocks[i].forward(checkpointInputs[i]);
            dx = blocks[i].backwardGradient(dx);
            if (checkpointInputs != null) blocks[i].clearTrainingState();
        }
        if (positions != null && gpu() != null) {
            gpu().append(dx, gradients.get(0), 0, true);
        } else if (positions != null) {
            Matrix dp = gradients.get(0);
            for (int i = 0; i < dx.rows(); i++)
                for (int t = 0; t < dx.columns(); t++) dp.values[i][t] += dx.values[i][t];
            dp.markDirty();
        }
        return embedding.backwardGradient(dx);
    }

    /** Mean shifted next-token cross entropy plus MoE auxiliary losses. */
    public float loss(int[] sequence) {
        boolean old = trainingMode;
        trainingMode = true;
        try {
            return trainingLoss(sequence);
        } finally {
            trainingMode = old;
        }
    }

    private float trainingLoss(int[] sequence) {
        validateSequence(sequence);
        int[] x = Arrays.copyOf(sequence, sequence.length - 1);
        forward(x);
        return crossEntropy(sequence, null);
    }

    private void validateSequence(int[] s) {
        if (s == null || s.length < 2 || s.length - 1 > config.contextLength)
            throw new IllegalArgumentException("Training sequence needs 2..context+1 tokens");
        for (int t : s)
            if (t < 0 || t >= config.vocabularySize)
                throw new IllegalArgumentException("Token outside vocabulary");
    }

    private float crossEntropy(int[] s, Matrix delta) {
        if (gpu() != null) {
            var result = gpu().crossEntropy(logits, s);
            if (delta != null) gpu().accumulate(delta, result.gradient());
            float total = result.loss();
            for (var b : blocks) total += b.auxiliaryLoss();
            return total;
        }
        double total = 0;
        int n = s.length - 1;
        for (int t = 0; t < n; t++) {
            double max = -Double.MAX_VALUE;
            for (int j = 0; j < logits.rows(); j++) max = Math.max(max, logits.values[j][t]);
            double sum = 0;
            for (int j = 0; j < logits.rows(); j++) sum += Math.exp(logits.values[j][t] - max);
            total += Math.log(sum) + max - logits.values[s[t + 1]][t];
            if (delta != null)
                for (int j = 0; j < logits.rows(); j++)
                    delta.values[j][t] =
                            (float)
                                    ((Math.exp(logits.values[j][t] - max) / sum
                                                    - (j == s[t + 1] ? 1 : 0))
                                            / n);
        }
        total /= n;
        for (var b : blocks) total += b.auxiliaryLoss();
        return (float) total;
    }

    public void fit(int[][] sequences, int epochs, float rate) {
        fit(sequences, epochs, rate, new AdamW(), 1, 1f);
    }

    /**
     * Independent sequences never attend across batch boundaries. Accumulate before each optimizer
     * step.
     */
    public void fit(
            int[][] sequences,
            int epochs,
            float rate,
            Optimizer optimizer,
            int accumulationSteps,
            float clipNorm) {
        boolean old = trainingMode;
        trainingMode = true;
        try {
            fitInternal(sequences, epochs, rate, optimizer, accumulationSteps, clipNorm);
        } finally {
            trainingMode = old;
        }
    }

    private void fitInternal(
            int[][] sequences,
            int epochs,
            float rate,
            Optimizer optimizer,
            int accumulationSteps,
            float clipNorm) {
        if (sequences == null
                || sequences.length == 0
                || epochs < 1
                || !Float.isFinite(rate)
                || rate <= 0
                || optimizer == null
                || accumulationSteps < 1
                || !Float.isFinite(clipNorm)
                || clipNorm < 0) throw new IllegalArgumentException("Invalid training arguments");
        for (int[] s : sequences) validateSequence(s);
        for (int epoch = 0; epoch < epochs; epoch++) {
            double sum = 0;
            for (int start = 0; start < sequences.length; start += accumulationSteps) {
                zeroGrad();
                float lossScaling =
                        config.precision == Precision.FP16_COMPUTE && gpu() != null
                                ? config.lossScale
                                : 1f;
                int count = Math.min(accumulationSteps, sequences.length - start);
                for (int b = 0; b < count; b++) {
                    int[] s = sequences[start + b];
                    forward(Arrays.copyOf(s, s.length - 1));
                    Matrix delta = new Matrix(config.vocabularySize, s.length - 1);
                    sum += crossEntropy(s, delta);
                    delta = scale(delta, lossScaling);
                    setAuxiliaryGradientScale(lossScaling);
                    try {
                        backwardGradient(delta);
                    } finally {
                        setAuxiliaryGradientScale(1f);
                    }
                }
                List<Matrix> gs = gradients();
                double squared =
                        gpu() != null
                                ? gpu().squaredSums(gs)
                                        / ((double) count * count * lossScaling * lossScaling)
                                : 0;
                for (Matrix g : gs) {
                    if (gpu() == null)
                        for (float[] row : g.values)
                            for (float v : row) squared += (double) (v / count) * (v / count);
                }
                if (!Double.isFinite(squared))
                    throw new IllegalStateException(
                            "Nonfinite gradients; no update applied. Reduce lossScale/learning rate"
                                    + " or select FP32.");
                float scale = 1f / (count * lossScaling);
                if (clipNorm > 0 && Math.sqrt(squared) > clipNorm)
                    scale *= clipNorm / Math.sqrt(squared);
                for (Matrix g : gs) {
                    if (gpu() != null) {
                        Matrix scaled = gpu().scale(g, scale);
                        gpu().zero(g);
                        gpu().accumulate(g, scaled);
                    } else g.multiply(scale);
                }
                step(optimizer, rate);
            }
            lastLoss = (float) (sum / sequences.length);
        }
    }

    public int[] generate(int[] prompt, GenerationConfig c) {
        Objects.requireNonNull(c);
        tokenMatrix(prompt);
        if (c.maxNewTokens < 0
                || (long) prompt.length + c.maxNewTokens > config.contextLength
                || !Float.isFinite(c.temperature)
                || c.temperature < 0
                || c.topK < 0
                || !Float.isFinite(c.topP)
                || c.topP <= 0
                || c.topP > 1
                || c.eosToken >= config.vocabularySize)
            throw new IllegalArgumentException("Invalid generation configuration/context");
        boolean wasTraining=trainingMode;trainingMode=false;
        int[] out = Arrays.copyOf(prompt, prompt.length + c.maxNewTokens);
        Random random = new Random(c.seed);
        resetCache();
        try {
            for (int i = prompt.length; i < out.length; i++) {
                Matrix y =
                        c.useKVCache
                                ? forwardCached(
                                        i == prompt.length ? prompt : new int[] {out[i - 1]})
                                : forward(Arrays.copyOf(out, i));
                int token = sample(y, y.columns() - 1, c, random);
                out[i] = token;
                if (token == c.eosToken) return Arrays.copyOf(out, i + 1);
            }
            return out;
        } finally {
            trainingMode=wasTraining;resetCache();
        }
    }

    private int sample(Matrix y, int t, GenerationConfig c, Random rng) {
        if (gpu() != null) {
            if (c.temperature == 0) return gpu().greedy(y, t);
            y = gpu().slice(y, t, 1);
            t = 0;
        }
        host(y);
        final Matrix sampledLogits = y;
        final int sampledColumn = t;
        Integer[] ids = new Integer[y.rows()];
        for (int i = 0; i < ids.length; i++) ids[i] = i;
        Arrays.sort(
                ids,
                (a, b) ->
                        Float.compare(
                                sampledLogits.values[b][sampledColumn],
                                sampledLogits.values[a][sampledColumn]));
        if (c.temperature == 0) return ids[0];
        int k = c.topK == 0 ? ids.length : Math.min(c.topK, ids.length);
        double[] p = new double[k];
        double sum = 0, max = y.values[ids[0]][t];
        for (int i = 0; i < k; i++) {
            p[i] = Math.exp((y.values[ids[i]][t] - max) / c.temperature);
            sum += p[i];
        }
        double cumulative = 0;
        int end = k;
        for (int i = 0; i < k; i++) {
            cumulative += p[i] / sum;
            if (cumulative >= c.topP) {
                end = i + 1;
                break;
            }
        }
        sum = 0;
        for (int i = 0; i < end; i++) sum += p[i];
        double draw = rng.nextDouble() * sum;
        for (int i = 0; i < end; i++) {
            draw -= p[i];
            if (draw <= 0) return ids[i];
        }
        return ids[end - 1];
    }

    public void save(String path) throws IOException {
        save(Path.of(path));
    }

    /** Versioned transformer .snn extension; weights/config only, optimizer state is not saved. */
    public void save(Path path) throws IOException {
        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(0x534e5431);
            out.writeInt(2);
            TransformerIO.writeConfig(out, config);
            TransformerIO.writeParameters(out, this);
        }
    }

    public static Transformer load(String path) throws IOException {
        return load(Path.of(path));
    }

    public static Transformer load(Path path) throws IOException {
        try (DataInputStream in =
                new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != 0x534e5431)
                throw new IOException("Unsupported transformer .snn format");
            int version = in.readInt();
            if (version != 1 && version != 2)
                throw new IOException("Unsupported transformer .snn version");
            Transformer m = new Transformer(TransformerIO.readConfig(in, version));
            TransformerIO.readParameters(in, m);
            if (in.read() != -1) throw new IOException("Trailing checkpoint data");
            return m;
        } catch (IllegalArgumentException ex) {
            throw new IOException("Invalid transformer checkpoint", ex);
        }
    }
}
