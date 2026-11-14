package xyz.gatoware.synapse.layer;

/** Per-token LayerNorm with trainable scale and bias. */
public final class LayerNormLayer extends RMSNormLayer {
    public LayerNormLayer(int size) {
        super(size, 1e-5f, true);
    }
}
