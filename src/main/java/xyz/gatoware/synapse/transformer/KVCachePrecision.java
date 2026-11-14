package xyz.gatoware.synapse.transformer;

/** KV cache storage formats. CPU FP16 caches retain half-rounded values in float arrays. */
public enum KVCachePrecision {
    FP32,
    FP16
}
