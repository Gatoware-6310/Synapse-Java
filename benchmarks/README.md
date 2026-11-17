# Synapse / PyTorch transformer benchmark

`run_comparison.py` runs frameworks sequentially on the same host and checks architecture, exact parameter count, precision label, and workload settings before producing ratios. PyTorch is only a benchmark dependency; Synapse does not depend on it.

```sh
python -m pip install torch
python benchmarks/run_comparison.py --device CPU --output benchmarks/results/my-cpu-dense.json
python benchmarks/run_comparison.py --device CPU --moe --output benchmarks/results/my-cpu-moe.json

# On the RTX 3070, after cudaTransformerCheck passes:
python benchmarks/run_comparison.py --device CUDA --sequence 256 --vocabulary 4096 \
  --hidden 256 --layers 4 --heads 8 --kv-heads 2 --ffn 768 \
  --warmup 20 --repeats 50 --decode-tokens 64 \
  --output benchmarks/results/rtx3070-dense-fp32.json
# Repeat with --moe, and repeat both with --precision FP16_COMPUTE.
```

Use `--gradle /path/to/gradle` if needed. The wrapper is the default. A CUDA request fails if NVIDIA hardware/libraries are unavailable. There is no synthesized GPU result.

The equivalent models have pre-RMSNorm (epsilon 1e-5), adjacent-pair RoPE, no projection biases, SwiGLU, tied embeddings, no dropout, GQA, and all-layer MoE when enabled. MoE uses 4 routed experts, top-2 softmax, 1 shared expert, and balance coefficient 0.01 with no capacity drops. The Java benchmark initializes weights deterministically with seed 7 and exports its initial checkpoint; the runner loads those exact parameters into PyTorch and checks agreement of the initial cross-entropy plus balance loss before reporting ratios. The checkpoint checksum is recorded; the temporary checkpoint is removed afterward. CPU PyTorch uses one thread. CUDA FP32 disables PyTorch TF32. Mixed precision uses FP32 master weights/AdamW states and FP16 GEMMs, static loss scale 128; PyTorch autocast also stores some intermediate tensors in half whereas Synapse currently retains FP32 tensors. Half-mode memory/attention arithmetic is therefore not bitwise equivalent; report that distinction with the results.

Default CPU smoke workload: batch 1, vocabulary 32, hidden 32, 2 layers, 4 query/2 KV heads, FFN 64, sequence 16, 8 decode tokens, 5 warmups, 9 timed repetitions. The checked-in measurements use 100 warmups and 31 repetitions; the CLI defaults are the lighter smoke settings above. These sizes expose overhead and are not representative of a useful large GPT or of RTX 3070 scaling.

Timings are median synchronized **wall-clock** measurements, including language/runtime orchestration:

- Prefill: full uncached forward and logits, with no sampling/host-logit transfer on CUDA.
- Backward: forward and zeroing occur outside the timer; backward accumulates the gradient of mean logits plus MoE balance loss. No optimizer step.
- Training: shifted cross-entropy plus balance loss, zeroing, backward, unscale, global clipping at 1, and AdamW (learning rate 0.0003, decay 0.01, epsilon 1e-7). Weights change during warmup/measurements.
- Decode: prefill and cache allocation occur outside the timer; measure a series of single-token cached forwards. Inputs are fixed tokens on both implementations; sampling is excluded.

JSON includes forward/backward/train/decode milliseconds, tokens/second, actual PyTorch/CUDA/runtime/cuBLAS versions as available, GPU identity, dimensions, warmup/repetition counts, source SHA-256, source commit, and dirty-tree status. When results were collected before the final documentation commit, source SHA-256 identifies the exact measured Java/CUDA/benchmark sources despite a dirty tree. Ratios are **Synapse tokens/s divided by PyTorch tokens/s**; above 1 means Synapse is faster for that measured workload only.

For CUDA, Synapse reports its peak owned allocations (including pooled buffers); PyTorch reports peak tensor allocation. These exclude some runtime/library allocations and are **not directly comparable total process VRAM**. The runner also polls `nvidia-smi` for board memory and utilization every 100 ms. Board samples include process startup and any other GPU processes, and can miss short-lived peaks. Run on an otherwise idle single-GPU host; use Nsight Systems for reliable attribution and kernel timelines. CPU memory/GPU fields are null or zero when unavailable, never invented.

An RTX 3070 was unavailable during development. Only the checked-in CPU smoke results were measured. Run hardware acceptance and collect GPU results before making CUDA correctness, speed, memory, or near-parity claims.

Direct Java invocation, for custom profiling:

```sh
./gradlew transformerBenchmark --args='CUDA false 256 4096 256 4 8 2 768 20 50 64 FP32'
```

Arguments: device, MoE boolean, sequence, vocabulary, hidden, layers, query heads, KV heads, FFN width, warmups, repetitions, decode tokens, precision. Use `nsys profile` around that command with child-process tracing enabled if profiling through Gradle.

Kernel syntax can be checked without a GPU using a locally installed NVRTC library:

```sh
python tools/check_cuda_kernels.py /path/to/libnvrtc.so.12
```

References: [NVIDIA cuBLAS compute modes](https://docs.nvidia.com/cuda/cublas/), [PyTorch SDPA/GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [PyTorch CUDA memory/timing](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Recorded CPU smoke ratios

These are the tiny batch-1 CPU workloads above, using PyTorch 2.15.0+cpu; they are not GPU predictions.

| Model | Prefill | Training | Decode |
|---|---:|---:|---:|
| dense | 0.99x | 2.89x | 9.49x |
| moe | 1.95x | 2.20x | 9.94x |
