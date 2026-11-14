package xyz.gatoware.synapse.transformer;

/** Decoder configuration. Defaults: pre-RMSNorm, RoPE, GELU, tied embeddings, FP32. */
public final class TransformerConfig {
    public final int vocabularySize, hiddenSize, layerCount, heads;
    public int contextLength=512, kvHeads, ffnSize;
    public boolean preNormalization=true, tieEmbeddings=true, activationCheckpointing=false;
    public PositionEncoding positionEncoding=PositionEncoding.ROPE;
    public Normalization normalization=Normalization.RMS_NORM;
    public Activations activation=Activations.GELU;
    public MoEConfig moe;
    public TransformerConfig(int vocabularySize,int hiddenSize,int layerCount,int heads) {
        if(vocabularySize<2||hiddenSize<1||layerCount<1||heads<1||hiddenSize%heads!=0) throw new IllegalArgumentException("Invalid transformer dimensions");
        this.vocabularySize=vocabularySize; this.hiddenSize=hiddenSize; this.layerCount=layerCount; this.heads=heads; kvHeads=heads; ffnSize=Math.multiplyExact(hiddenSize,4);
    }
    public TransformerConfig setContextLength(int n) { if(n<1) throw new IllegalArgumentException(); contextLength=n; return this; }
    public TransformerConfig setKVHeads(int n) { if(n<1||heads%n!=0) throw new IllegalArgumentException("KV heads must divide query heads"); kvHeads=n; return this; }
    public TransformerConfig setFFNSize(int n) { if(n<1) throw new IllegalArgumentException(); ffnSize=n; return this; }
    public TransformerConfig setPositionEncoding(PositionEncoding n) { positionEncoding=java.util.Objects.requireNonNull(n); return this; }
    public TransformerConfig setNormalization(Normalization n) { normalization=java.util.Objects.requireNonNull(n); return this; }
    public TransformerConfig setActivation(Activations n) { activation=java.util.Objects.requireNonNull(n); return this; }
    public TransformerConfig setPreNormalization(boolean n) { preNormalization=n; return this; }
    public TransformerConfig setTieEmbeddings(boolean n) { tieEmbeddings=n; return this; }
    public TransformerConfig setActivationCheckpointing(boolean n) { activationCheckpointing=n; return this; }
    public TransformerConfig setMoE(MoEConfig n) { moe=java.util.Objects.requireNonNull(n).copy(); return this; }
    public TransformerConfig setPrecision(Precision n) { java.util.Objects.requireNonNull(n); return this; }
    public TransformerConfig copy() { TransformerConfig c=new TransformerConfig(vocabularySize,hiddenSize,layerCount,heads); c.contextLength=contextLength;c.kvHeads=kvHeads;c.ffnSize=ffnSize;c.positionEncoding=positionEncoding;c.normalization=normalization;c.activation=activation;c.preNormalization=preNormalization;c.tieEmbeddings=tieEmbeddings;c.activationCheckpointing=activationCheckpointing;c.moe=moe==null?null:moe.copy(); return c; }
    public void validate() { if(positionEncoding==PositionEncoding.ROPE&&(hiddenSize/heads)%2!=0) throw new IllegalArgumentException("RoPE requires an even head dimension"); }
}
