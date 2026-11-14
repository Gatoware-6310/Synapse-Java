package xyz.gatoware.synapse.transformer;

/** Token-choice routing configuration. Capacity drops retain their original routing weights. */
public final class MoEConfig {
    public final int experts, topK;
    public int expertHiddenSize = 0, sharedExperts = 0, interval = 1;
    public float loadBalancing = 0, capacityFactor = Float.POSITIVE_INFINITY;
    public OverflowPolicy overflowPolicy = OverflowPolicy.DROP;

    public MoEConfig setOverflowPolicy(OverflowPolicy policy) {
        overflowPolicy = java.util.Objects.requireNonNull(policy);
        return this;
    }

    public RouterType router = RouterType.TOP_K_SOFTMAX;
    public int[] layers = null;

    public MoEConfig(int experts, int topK) {
        if (experts < 1 || topK < 1 || topK > experts)
            throw new IllegalArgumentException("Require 1 <= topK <= experts");
        this.experts = experts;
        this.topK = topK;
    }

    public MoEConfig setExpertHiddenSize(int n) {
        if (n < 1) throw new IllegalArgumentException();
        expertHiddenSize = n;
        return this;
    }

    public MoEConfig setSharedExperts(int n) {
        if (n < 0) throw new IllegalArgumentException();
        sharedExperts = n;
        return this;
    }

    public MoEConfig setRouter(RouterType r) {
        router = java.util.Objects.requireNonNull(r);
        return this;
    }

    public MoEConfig setLoadBalancing(float n) {
        if (!Float.isFinite(n) || n < 0) throw new IllegalArgumentException();
        loadBalancing = n;
        return this;
    }

    public MoEConfig setCapacityFactor(float n) {
        if (!(n > 0)) throw new IllegalArgumentException();
        capacityFactor = n;
        return this;
    }

    public MoEConfig setLayers(MoELayers n) {
        interval = n == MoELayers.ALL ? 1 : 2;
        layers = null;
        return this;
    }

    public MoEConfig setLayers(int... n) {
        for (int i : n) if (i < 0) throw new IllegalArgumentException();
        layers = n.clone();
        return this;
    }

    public MoEConfig setInterval(int n) {
        if (n < 1) throw new IllegalArgumentException();
        interval = n;
        layers = null;
        return this;
    }

    public boolean includes(int n) {
        if (layers == null) return n % interval == 0;
        for (int i : layers) if (i == n) return true;
        return false;
    }

    public void validate() {
        if (expertHiddenSize < 0
                || sharedExperts < 0
                || interval < 1
                || !Float.isFinite(loadBalancing)
                || loadBalancing < 0
                || !(capacityFactor > 0)
                || router == null || overflowPolicy==null) throw new IllegalArgumentException("Invalid MoE configuration");
        if (layers != null)
            for (int i : layers)
                if (i < 0) throw new IllegalArgumentException("Negative MoE layer index");
    }

    public MoEConfig copy() {
        MoEConfig c = new MoEConfig(experts, topK);
        c.expertHiddenSize = expertHiddenSize;
        c.sharedExperts = sharedExperts;
        c.interval = interval;
        c.loadBalancing = loadBalancing;
        c.capacityFactor = capacityFactor;
        c.router = router;
        c.overflowPolicy = overflowPolicy;
        c.layers = layers == null ? null : layers.clone();
        return c;
    }
}
