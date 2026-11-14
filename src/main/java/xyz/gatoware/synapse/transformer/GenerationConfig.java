package xyz.gatoware.synapse.transformer;

/** Sampling configuration; temperature zero selects greedy decoding. */
public final class GenerationConfig {
    public int maxNewTokens = 32, topK = 0, eosToken = -1;
    public float temperature = 0, topP = 1;
    public long seed = 0;
    public boolean useKVCache = true;

    public GenerationConfig setMaxNewTokens(int n) {
        if (n < 0) throw new IllegalArgumentException();
        maxNewTokens = n;
        return this;
    }

    public GenerationConfig setTemperature(float n) {
        if (!Float.isFinite(n) || n < 0) throw new IllegalArgumentException();
        temperature = n;
        return this;
    }

    public GenerationConfig setTopK(int n) {
        if (n < 0) throw new IllegalArgumentException();
        topK = n;
        return this;
    }

    public GenerationConfig setTopP(float n) {
        if (!Float.isFinite(n) || n <= 0 || n > 1) throw new IllegalArgumentException();
        topP = n;
        return this;
    }

    public GenerationConfig setEOS(int n) {
        if (n < 0) throw new IllegalArgumentException();
        eosToken = n;
        return this;
    }

    public GenerationConfig setSeed(long n) {
        seed = n;
        return this;
    }

    public GenerationConfig setUseKVCache(boolean n) {
        useKVCache = n;
        return this;
    }
}
