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

`DenseLayer.forward` accepts `inputSize x batchSize` matrices for inference, with one sample per column.

For `NeuralNetwork.forward`, eligible hidden `DenseLayer + ReLU` stages use a GPU-resident fast path. Synapse performs the matrix multiply on cuBLAS, then launches a fused bias+ReLU CUDA kernel compiled once with NVRTC. The hidden result stays in VRAM and is fed directly into the next layer without a host readback or re-upload. The final layer materializes its output normally so the returned `Matrix.values` is immediately valid Java data.

The resident ReLU kernel is optional. If NVRTC/driver-module initialization is unavailable, ordinary cuBLAS CUDA execution still works and Synapse falls back to materialized layer execution.

Training intentionally uses the materialized path so existing backward state and optimizer behavior stay correct. Batched backward/training is not implemented yet.

## GPU memory/cache optimizations

The CUDA backend keeps a small identity cache for recently used matrices and a size-segregated VRAM allocation pool. This avoids repeated uploads of unchanged weights/inputs and substantially reduces `cudaMalloc`/`cudaFree` traffic.

The previous implementation fingerprinted every cached `float[][]` on every CUDA operation. That scan was removed because it became a major part of the cost once cuBLAS itself was fast.

Synapse-owned mutating `Matrix` operations automatically invalidate their cached device representation. If application code directly modifies the public `values` array after that matrix has already been used on CUDA, explicitly mark it dirty:

```java
matrix.values[0][0] = 123.0f;
matrix.markDirty();
```

before the next CUDA operation involving that matrix.

CUDA multiplication results are cached too, so chained multiplication can reuse a device-side result. The hot matrix cache is capped at 16 entries. Evicted allocations may enter a reusable pool of up to 32 device buffers instead of immediately being freed.

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
- CUDA NVRTC library for the optional GPU-resident fused-ReLU path

On the current test setup, CUDA 12.9 is installed separately and Synapse is launched with its `lib64` directory in `LD_LIBRARY_PATH`.

## Build and test

```bash
git switch cuda
git pull
LD_LIBRARY_PATH="$HOME/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH" ./gradlew clean test
```

CUDA tests cover CPU/CUDA matrix parity, cache invalidation, chained cached results, batched dense parity, and GPU-resident network parity when NVRTC is available.

Benchmark with:

```bash
LD_LIBRARY_PATH="$HOME/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH" ./gradlew cudaBenchmark
```

The benchmark includes raw/cached 512x512 multiplication, dense forward at batch sizes 1 through 256, and a multi-layer `784 -> 1024 x3 -> 10` network at several batch sizes.

Build normal artifacts with:

```bash
./gradlew build
```

or the experimental bundled runtime jar with:

```bash
./gradlew fatJar
```

Artifacts are written under `build/libs/`.
