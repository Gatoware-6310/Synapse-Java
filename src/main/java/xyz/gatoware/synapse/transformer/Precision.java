package xyz.gatoware.synapse.transformer;

/** FP16_COMPUTE uses Tensor Core GEMM with FP32 master storage and accumulation. */
public enum Precision {
    FP32,
    FP16_COMPUTE
}
