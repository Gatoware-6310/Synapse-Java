package xyz.gatoware.synapse.transformer;

/** Decoder configuration. Defaults: pre-RMSNorm, RoPE, GELU, tied embeddings, FP32. */
public final class TransformerConfig {
    public final int vocabularySize, hiddenSize, layerCount, heads;
    public int contextLength = 512, kvHeads, ffnSize;
    public boolean preNormalization = true, tieEmbeddings = true, activationCheckpointing = false;
    public PositionEncoding positionEncoding = PositionEncoding.ROPE;
    public Normalization normalization = Normalization.RMS_NORM;
    public Activations activation = Activations.GELU;
    public MoEConfig moe;
    public Precision precision = Precision.FP32;
    public float lossScale = 128f;
    public KVCachePrecision kvCachePrecision = KVCachePrecision.FP32;

    public TransformerConfig(int vocabularySize, int hiddenSize, int layerCount, int heads) {
        if (vocabularySize < 2
                || hiddenSize < 1
                || layerCount < 1
                || heads < 1
                || hiddenSize % heads != 0)
            throw new IllegalArgumentException("Invalid transformer dimensions");
        this.vocabularySize = vocabularySize;
        this.hiddenSize = hiddenSize;
        this.layerCount = layerCount;
        this.heads = heads;
        kvHeads = heads;
        ffnSize = Math.multiplyExact(hiddenSize, 4);
    }

    public TransformerConfig setContextLength(int n) {
        if (n < 1) throw new IllegalArgumentException();
        contextLength = n;
        return this;
    }

    public TransformerConfig setKVHeads(int n) {
        if (n < 1 || heads % n != 0)
            throw new IllegalArgumentException("KV heads must divide query heads");
        kvHeads = n;
        return this;
    }

    public TransformerConfig setFFNSize(int n) {
        if (n < 1) throw new IllegalArgumentException();
        ffnSize = n;
        return this;
    }

    public TransformerConfig setPositionEncoding(PositionEncoding n) {
        positionEncoding = java.util.Objects.requireNonNull(n);
        return this;
    }

    public TransformerConfig setNormalization(Normalization n) {
        normalization = java.util.Objects.requireNonNull(n);
        return this;
    }

    public TransformerConfig setActivation(Activations n) {
        activation = java.util.Objects.requireNonNull(n);
        return this;
    }

    public TransformerConfig setPreNormalization(boolean n) {
        preNormalization = n;
        return this;
    }

    public TransformerConfig setTieEmbeddings(boolean n) {
        tieEmbeddings = n;
        return this;
    }

    public TransformerConfig setActivationCheckpointing(boolean n) {
        activationCheckpointing = n;
        return this;
    }

    public TransformerConfig setMoE(MoEConfig n) {
        moe = java.util.Objects.requireNonNull(n).copy();
        return this;
    }

    public TransformerConfig setPrecision(Precision n) {
        precision = java.util.Objects.requireNonNull(n);
        return this;
    }

    public TransformerConfig setKVCachePrecision(KVCachePrecision p) {
        kvCachePrecision = java.util.Objects.requireNonNull(p);
        return this;
    }

    public TransformerConfig setLossScale(float n) {
        if (!Float.isFinite(n) || n < 1) throw new IllegalArgumentException();
        lossScale = n;
        return this;
    }

    public TransformerConfig copy() {
        TransformerConfig c = new TransformerConfig(vocabularySize, hiddenSize, layerCount, heads);
        c.contextLength = contextLength;
        c.kvHeads = kvHeads;
        c.ffnSize = ffnSize;
        c.positionEncoding = positionEncoding;
        c.normalization = normalization;
        c.activation = activation;
        c.preNormalization = preNormalization;
        c.tieEmbeddings = tieEmbeddings;
        c.activationCheckpointing = activationCheckpointing;
        c.kvCachePrecision = kvCachePrecision;
        c.precision = precision;
        c.lossScale = lossScale;
        c.moe = moe == null ? null : moe.copy();
        return c;
    }

    public void validate() {
        if (contextLength < 1
                || kvHeads < 1
                || heads % kvHeads != 0
                || ffnSize < 1
                || positionEncoding == null
                || normalization == null
                || activation == null
                || precision == null
                || kvCachePrecision == null
                || !Float.isFinite(lossScale)
                || lossScale < 1)
            throw new IllegalArgumentException("Invalid transformer configuration");
        if ((long) hiddenSize * vocabularySize > Integer.MAX_VALUE
                || (long) hiddenSize * ffnSize > Integer.MAX_VALUE
                || (long) hiddenSize * contextLength > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Matrix exceeds Java/CUDA indexing limit");
        if (hiddenSize / heads > 4096)
            throw new IllegalArgumentException(
                    "Head dimension exceeds kernel shared-memory limit (4096)");
        if (moe != null) moe.validate();
        if (positionEncoding == PositionEncoding.ROPE && (hiddenSize / heads) % 2 != 0)
            throw new IllegalArgumentException("RoPE requires an even head dimension");
    }
}
