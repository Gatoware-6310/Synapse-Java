# GPT, decoder Transformer, and MoE

These are additions to Synapse's Java library. They reuse `Matrix`, `Layer`, the selected backend, and `Optimizer`; there is no separate model language or external neural-network runtime.

## Verification status

- CPU: numerical parameter/input gradient checks; causal attention; MHA/GQA/MQA; normalization; RoPE; GELU/SwiGLU; MoE routing/shared experts/overflow/balance gradients; end-to-end training; tied weights; checkpoint recomputation; generation; FP32 and half-rounded KV cache; model/layer serialization.
- CUDA: Java bindings compile, and every CUDA kernel compiles with NVRTC for `compute_52` and `compute_86`. **No CUDA GPU exists in the development environment. CUDA numerical agreement, runtime memory safety, FP16 accuracy, GPU training, and RTX 3070 throughput remain unverified.**
- `cudaTransformerCheck` explicitly fails when CUDA is absent. It is not counted as a passed CPU test. Existing repository CUDA JUnit tests retain their existing hardware assumptions.
- CPU benchmark smoke results are in `benchmarks/results`. They do not establish RTX 3070 performance.

## Small GPT

```java
import xyz.gatoware.synapse.*;
import xyz.gatoware.synapse.transformer.*;

Transformer model = new Transformer(
    new TransformerConfig(128, 64, 2, 4)
        .setContextLength(128)
        .setFFNSize(128)
        .setActivation(Activations.SWIGLU)
);

int[][] sequences = {
    {0, 1, 2, 3, 0, 1, 2, 3},
    {1, 2, 3, 0, 1, 2, 3, 0}
};
model.fit(sequences, 100, 0.003f);
int[] output = model.generate(new int[]{0, 1},
    new GenerationConfig().setMaxNewTokens(16));
model.save("small-gpt.snn");
Transformer restored = Transformer.load("small-gpt.snn");
```

Constructor arguments are `(vocabularySize, hiddenSize, layerCount, queryHeads)`. A training sequence supplies both inputs and shifted next-token labels: the last token is only a label. `fit` reports mean token cross-entropy plus the sum of per-block MoE balance losses. Independent sequences never share attention. Accumulation averages **sequence means**, so differently sized sequences receive equal weight.

```java
model.fit(sequences, 100, 0.003f,
    new xyz.gatoware.synapse.optimizer.AdamW(0.01f),
    4,       // independent sequences per optimizer update
    1.0f);   // global gradient norm cap; 0 disables clipping
```

AdamW applies decoupled decay to all parameter matrices, including normalization parameters. Optimizers and forward/KV state are mutable; model instances are not thread-safe. An optimizer must be reset/recreated when switching CPU/CUDA because its moment storage belongs to the backend that performed updates.

## GQA, MQA, positions, and normalization

```java
TransformerConfig config = new TransformerConfig(256, 128, 4, 8)
    .setKVHeads(2) // GQA; 1 = MQA; queryHeads = MHA
    .setContextLength(512)
    .setPositionEncoding(PositionEncoding.ROPE)
    .setNormalization(Normalization.RMS_NORM)
    .setPreNormalization(true)
    .setTieEmbeddings(true);
```

RoPE rotates adjacent channel pairs, with base 10000, and requires an even head dimension. `LEARNED` uses a trainable absolute-position table instead. RMSNorm and LayerNorm use epsilon `1e-5`. Each token is normalized across hidden channels. GELU uses the tanh approximation; SwiGLU uses `SiLU(up(x)) * gate(x)` followed by the down projection. Attention and FFN projections have no biases.

Defaults: context 512, KV heads equal query heads, FFN width `4 * hidden`, RoPE, RMSNorm, GELU, pre-normalization, tied embedding/output weights, FP32 computation/cache, checkpointing disabled. Query-head dimension is limited to 4096 for the CUDA shared-memory attention kernel.

## MoE

```java
Transformer model = new Transformer(
    new TransformerConfig(256, 128, 4, 8)
        .setActivation(Activations.SWIGLU)
        .setMoE(new MoEConfig(8, 2)
            .setExpertHiddenSize(256)
            .setSharedExperts(1)
            .setRouter(RouterType.TOP_K_SOFTMAX)
            .setLoadBalancing(0.01f)
            .setCapacityFactor(1.25f)
            .setOverflowPolicy(OverflowPolicy.DROP)
            .setLayers(MoELayers.EVERY_OTHER))
);
```

Token-choice routers: `TOP_K_SOFTMAX` and `TOP_K_SIGMOID`. Selected scores are normalized to sum to one **before capacity dropping**. Ties prefer lower expert indices. Shared experts always run and their outputs are added without routing weights.

Capacity per expert is `max(1, ceil(factor * tokens * topK / experts))`. Overflow defaults to `DROP`: accept assignments in token order and discard excess assignments without renormalizing surviving weights. `ROUTE_ALL` ignores capacity and computes every selected assignment. Shared experts continue to contribute when routed assignments are dropped.

Capacity applies in training mode only. Inference routes all selected assignments, which preserves KV-cache consistency. `Transformer.fit`/`loss` select training mode; `forward` defaults to inference. `setTraining(true)` enables training-mode capacity for manual model forward/backward work. Standalone layers default to training mode.

Balance loss is `alpha * E * sum(mean(routerSoftmax) * selectedFrequency)`, where frequency counts selected assignments before dropping and divides by `tokens * topK`. Frequency/top-k selection are treated as piecewise constant during backward. Sigmoid routers still use full softmax probabilities for this balance term. `auxiliaryLoss()` exposes the value. `backwardGradient` adds its gradients; `backwardInput` differentiates only the requested layer output and preserves parameter gradients.

Select layers using zero-based `.setLayers(0, 2, 5)` or `.setInterval(3)` (indices 0, 3, 6, ...). `EVERY_OTHER` selects 0, 2, 4, ... . Unselected blocks use dense FFNs. Expert-choice routing is not implemented.

## Standalone layers and shapes

```java
import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.matrix.Matrix;

TransformerConfig c = new TransformerConfig(128, 32, 2, 4);
NeuralNetwork network = new NeuralNetwork(new Layer[]{
    new EmbeddingLayer(128, 32),
    new TransformerDecoderLayer(c),
    new RMSNormLayer(32)
});
Matrix hidden = network.forward(new Matrix(new float[][]{{65, 66, 67}}));
network.save("layers.snn");
```

IDs have shape `[1, tokens]`; hidden states `[hidden, tokens]`; logits `[vocabulary, tokens]`. IDs are validated integers in the configured vocabulary. Columns form **one causal sequence**, not independent minibatch examples. Do not use `NeuralNetwork.fit`'s column minibatches to represent independent attention sequences; use `Transformer.fit`. Standalone FFN/normalization/MoE layers also implement ordinary `Layer.backward` and reuse existing optimizers. This entry point averages parameter gradients over columns, matching existing dense layers. `backwardGradient` instead accumulates sums without stepping, for manual training/accumulation.

`NeuralNetwork` writes `.snn` version 3 with explicit transformer-layer payloads and still reads its original versions 1/2. `Transformer.save/load` uses the separate `SNT1` magic and format version 2, and reads its initial version 1. Model/config/weights are saved, including tied-weight semantics; optimizer moments, KV cache, and random-generator state are not saved. Resuming therefore starts a new optimizer trajectory. Use `Transformer.load` for a model checkpoint, and `NeuralNetwork.load` for a layer network.

## Generation and KV cache

```java
int[] output = model.generate(prompt,
    new GenerationConfig()
        .setMaxNewTokens(64)
        .setTemperature(0.8f)
        .setTopK(50)
        .setTopP(0.95f)
        .setSeed(42)
        .setUseKVCache(true));
```

The returned array includes the prompt. Temperature zero is greedy (the default). Top-k 0 disables that filter; top-p 1 disables nucleus filtering. Top-p is applied after top-k. `setEOS(token)` stops after appending that token. Prompt plus requested new tokens must fit the configured context; generation never silently truncates or slides the window.

`forwardCached` appends a prefill chunk or one token; call `resetCache` before a new independent prompt, after modifying weights, and before switching devices. `generate` resets before and after its work. Cached forward state is for inference; backward requires an uncached forward.

`setKVCachePrecision(KVCachePrecision.FP16)` uses actual packed half storage on CUDA. CPU caches retain half-rounded values in float arrays as a correctness reference; CPU half mode does not reduce allocation size. FP32 is the default. Cache allocation is lazy and reused during a generation session; resetting releases CUDA cache buffers back to the backend pool.

## CUDA execution and precision

```java
Synapse.useDevice(Devices.CUDA);
Transformer model = new Transformer(config
    .setPrecision(Precision.FP16_COMPUTE)
    .setLossScale(128f)
    .setKVCachePrecision(KVCachePrecision.FP16)
    .setActivationCheckpointing(true));
```

CUDA requires compatible NVIDIA driver, CUDA runtime/cuBLAS, and NVRTC libraries. Synapse bundles the JCuda bindings, not the CUDA toolkit. Kernel compilation errors include NVRTC's log and are never hidden by a CPU fallback. CPU remains the default and can construct/train/load/run the same models.

CUDA GEMMs use the existing cuBLAS handle. FP32 uses SGEMM/SGEMV. `FP16_COMPUTE` uses `cublasGemmEx`'s FP32-storage/FP16-compute mode with FP32 accumulation and requires Ampere or newer. **It is not persistent FP16 weight storage**: weights, activations, gradients, and optimizer states remain FP32. Static loss scaling applies to both task and auxiliary gradients before unscaling/clipping. Nonfinite gradients stop the update with a diagnostic. CPU execution of this configuration still computes in FP32. BF16 is not offered.

Attention uses a fused cooperative online-softmax forward kernel with linear workspace; backward recomputes probabilities. This is **not a tuned FlashAttention implementation**. RMSNorm/LayerNorm, RoPE, FFN activations/derivatives, residual adds, embeddings, loss/gradient, MoE routing/gather/scatter, and built-in optimizer updates have resident CUDA paths. MoE transfers expert counts to Java for variable-sized GEMM dispatch; loss and global gradient norm transfer small reductions. Greedy sampling transfers one token ID; stochastic sampling transfers the last vocabulary column to Java. No dense activation is intentionally computed by Java when CUDA is selected.

Built-in SGD, Momentum, AdaGrad, RMSProp, Adam, and AdamW are supported on CUDA. Custom optimizer implementations require explicit CPU selection. CUDA Graphs, cuBLASLt epilogue/algorithm tuning, persistent half model weights, fused grouped expert GEMM, and a hardware-tuned FlashAttention backward are not implemented. There is no demonstrated PyTorch GPU parity.

## Memory and limits

- `parameterCount()` is exact. `Transformer.estimatedParameterCount(config)` does not allocate a model.
- `estimatedMemoryBytes(sequenceLength, training)` is a conservative payload estimate, not a peak-VRAM guarantee. It excludes Java row/object overhead, cuBLAS workspace, backend cached temporaries/pools, and allocator overhead. The GPU cache can hold both training and inference buffers.
- Standard AdamW training needs roughly 16 bytes per parameter for FP32 weight/gradient/two moments, before activations and caches. The constructor also checks a conservative host parameter/gradient budget against the JVM heap.
- Activation checkpointing retains block inputs and recomputes attention/FFN/norm intermediates during backward. This trades compute for live activation storage; the backend may retain released temporaries in its bounded cache/pool.
- The existing CUDA matrix cache defaults to 128 entries. It reports its first eviction and exposes `getCacheEvictions()`. Dirty evictions materialize host data. Tune `((CudaBackend) Synapse.backend()).setMatrixCacheLimit(n)` to avoid eviction-driven transfers, while respecting VRAM. Pinned KV buffers are independent of this matrix cache.
- Allocation failures include requested/free/total CUDA bytes. Reduce model/context/FFN/batch accumulation or enable checkpointing; the model never silently moves execution to CPU.
- There is no true fused multi-sequence batch kernel yet: `fit` processes independent sequences serially and supports configurable accumulation. `estimatedMemoryBytes` is not an admission-control promise.

## Runnable examples and checks

```sh
./gradlew test build
./gradlew transformerExamples --args='basic'
./gradlew transformerExamples --args='train 30'
./gradlew transformerExamples --args='gqa'
./gradlew transformerExamples --args='moe'
./gradlew transformerExamples --args='layers'
./gradlew transformerExamples --args='cuda'
./gradlew cudaTransformerCheck
```

The training example uses ASCII character IDs and saves/reloads `build/character-gpt.snn`. Its vocabulary mapping is fixed ASCII, so no separate tokenizer artifact is needed. `cudaTransformerCheck` covers 48 CPU/CUDA configurations, both compute modes, both KV precisions, dense/MoE, GQA variants, positions, checkpointing, gradients, loss, cache, and a training step. It must actually run on CUDA before treating the backend as verified.

See [benchmarks/README.md](benchmarks/README.md) for paired PyTorch benchmarks and exact measurement limitations.
