# Experimental CUDA backend

This branch is a proof-of-concept CUDA backend for Synapse while keeping `Matrix` as the public numeric type.

## User API

CPU remains the default:

```java
NeuralNetwork network = new NeuralNetwork(...);
```

Enable CUDA with:

```java
import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;

Synapse.useDevice(Devices.CUDA);
```

Switch back with:

```java
Synapse.useDevice(Devices.CPU);
```

Availability can be checked with:

```java
Synapse.isDeviceAvailable(Devices.CUDA);
```

## CUDA execution

`Matrix.multiply(Matrix)` and dense-layer multiplication use cuBLAS. Batch-size-1 multiplication uses SGEMV; multi-column/batched multiplication uses SGEMM.

`DenseLayer.forward` accepts `inputSize x batchSize` matrices, with one sample per column.

For `NeuralNetwork.forward`, eligible hidden `DenseLayer + ReLU` stages use a GPU-resident fast path. Synapse performs the matrix multiply on cuBLAS, then launches a fused bias+ReLU CUDA kernel compiled once with NVRTC. The hidden result stays in VRAM and feeds directly into the next layer without a host readback or re-upload. The final layer materializes its output so the returned `Matrix.values` is valid Java data.

## CUDA training

When CUDA is selected, `NeuralNetwork.fit()` now uses mini-batches of 32 samples for compatible dense networks.

The training path keeps the expensive work on the GPU:

- hidden Dense + ReLU forward passes stay resident in VRAM
- ReLU derivatives run in a CUDA kernel
- input gradients use cuBLAS GEMM
- weight gradients use cuBLAS GEMM and are averaged across the mini-batch
- bias gradients are reduced in a CUDA kernel
- SGD, Momentum, AdaGrad, RMSProp, and Adam parameter updates run in CUDA kernels
- Adam/Momentum/AdaGrad/RMSProp state remains resident in VRAM
- updated weights and biases remain device-authoritative between batches

The final activation/loss boundary is intentionally materialized. Softmax and the configured `LossFunction` continue to use the existing Java API; their resulting gradient is uploaded once and backpropagation immediately returns to the GPU.

GPU-updated parameters are synchronized back into public `Matrix.values` before `fit()` returns, before model saving, and when the CUDA backend is closed.

CUDA mini-batching is only selected when the network is made of dense layers and every hidden layer can use the resident ReLU path. Unsupported/custom network structures keep the existing CPU training behavior.

## GPU memory/cache optimizations

The CUDA backend keeps an identity cache for recently used matrices and a size-segregated VRAM allocation pool. This avoids repeated uploads of unchanged weights/inputs and substantially reduces `cudaMalloc`/`cudaFree` traffic.

The previous implementation fingerprinted every cached `float[][]` on every CUDA operation. That scan was removed because it became a major part of the cost once cuBLAS itself was fast.

Synapse-owned mutating `Matrix` operations automatically invalidate their cached device representation. If application code directly modifies the public `values` array after that matrix has already been used on CUDA, explicitly mark it dirty:

```java
matrix.values[0][0] = 123.0f;
matrix.markDirty();
```

before the next CUDA operation involving that matrix.

Device-authoritative matrices are materialized automatically before eviction if necessary. The hot matrix cache is capped at 128 entries, with a reusable pool of up to 64 device buffers.

## Batched inference

```java
// 784 features, 64 samples
Matrix batch = new Matrix(784, 64);
Matrix output = layer.forward(batch);
```

A full `NeuralNetwork.forward(batch)` can also process batched matrices. Hidden ReLU layers remain GPU-resident when the fused NVRTC path is available.

## Requirements

- Java 21
- NVIDIA CUDA-capable GPU
- NVIDIA driver
- CUDA/cuBLAS compatible with JCuda 12.6
- CUDA NVRTC library for resident fused kernels and CUDA training

On the current test setup, CUDA 12.9 is installed separately and Synapse is launched with its `lib64` directory in `LD_LIBRARY_PATH`.

## Build and test

```bash
git switch cuda
git pull
LD_LIBRARY_PATH="$HOME/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH" ./gradlew clean test
```

CUDA tests cover CPU/CUDA matrix parity, cache invalidation, chained cached results, batched dense parity, GPU-resident network parity, and a CUDA training update/materialization check when NVRTC is available.

Benchmark with:

```bash
LD_LIBRARY_PATH="$HOME/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH" ./gradlew cudaBenchmark
```

The benchmark includes raw/cached 512x512 multiplication, dense forward at batch sizes 1 through 256, a multi-layer `784 -> 1024 x3 -> 10` network, and a full training-epoch CPU/CUDA comparison.

Build normal artifacts with:

```bash
./gradlew build
```

or the experimental bundled runtime jar with:

```bash
./gradlew fatJar
```

Artifacts are written under `build/libs/`.
