# Experimental CUDA backend

This branch is a proof-of-concept CUDA backend for Synapse while keeping `Matrix` as the public numeric type.

## User API

CPU is the default and requires no configuration:

```java
NeuralNetwork network = new NeuralNetwork(...);
```

To enable CUDA:

```java
import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;

Synapse.useDevice(Devices.CUDA);
```

To switch back:

```java
Synapse.useDevice(Devices.CPU);
```

Availability can be checked before switching:

```java
if (Synapse.isDeviceAvailable(Devices.CUDA)) {
    Synapse.useDevice(Devices.CUDA);
}
```

## What is accelerated

`Matrix.multiply(Matrix)` and dense-layer matrix multiplication are routed through cuBLAS SGEMM when CUDA is selected.

`DenseLayer.forward` also supports batched inference. Inputs are shaped as `inputSize x batchSize`, with one sample per column. This turns dense inference into matrix-matrix work, which is substantially more GPU-friendly than batch-size-1 inference.

Backpropagation, optimizers, activations, and element-wise matrix operations are still CPU code in this branch. Batched backward/training is not implemented yet.

## High-performance GPU memory cache

The CUDA backend keeps a bounded cache of device allocations for recently used matrices. Reusing a matrix therefore reuses its VRAM allocation instead of allocating and uploading it for every multiplication.

The previous implementation fingerprinted every cached `float[][]` on every operation so direct host writes could be detected automatically. That scan was expensive enough to dominate small and medium CUDA operations, so it has been removed from the hot path.

Synapse-owned matrix operations invalidate their cached device copy automatically. If application code modifies the public `Matrix.values` array directly after that matrix has already been used on CUDA, call:

```java
matrix.values[0][0] = 123.0f;
matrix.markDirty();
```

before the matrix participates in another CUDA operation.

CUDA results are cached too, allowing a multiplication result to feed another multiplication without re-uploading the result. The hot cache is capped at 16 matrices.

Evicted allocations are not immediately passed back through `cudaFree`. A small size-segregated device-buffer pool keeps up to 32 allocations for reuse, reducing the cost of repeated `cudaMalloc`/`cudaFree` calls. This is similar in spirit to the caching allocators used by larger ML runtimes.

A host copy of each public multiplication result is still produced because `Matrix.values` is immediately readable Java state. Avoiding that final device-to-host synchronization requires an internal GPU-resident execution path for full network chains; that is the next major performance ceiling.

## Batched inference

A dense layer can now receive multiple samples at once:

```java
// 784 features, 64 samples
Matrix batch = new Matrix(784, 64);
Matrix output = layer.forward(batch); // outputSize x 64
```

Batch-size-1 forward/backward behavior remains unchanged. Calling `backward` immediately after a batched forward currently throws because batched gradient accumulation has not been implemented yet.

## Requirements

- Java 21
- NVIDIA GPU with CUDA support
- NVIDIA driver
- CUDA/cuBLAS compatible with the JCuda runtime in this branch

JCuda's Java and native binding jars are resolved by Gradle for the current operating system and architecture.

## Build and test

```bash
git switch cuda
git pull
./gradlew test
```

CUDA-specific tests automatically skip when the complete CUDA/cuBLAS backend cannot initialize. When CUDA is available, the suite checks CPU/CUDA numerical parity, cached-result reuse, explicit host-cache invalidation, and batched dense parity.

To benchmark the CUDA path:

```bash
./gradlew cudaBenchmark
```

The benchmark includes raw 512x512 multiplication, fully cached 512x512 multiplication, and dense forward passes at batch sizes 1, 8, 32, 64, 128, and 256.

To build the normal jars:

```bash
./gradlew build
```

To additionally create an experimental bundled runtime jar containing Synapse plus its JCuda dependencies:

```bash
./gradlew fatJar
```

Artifacts are written under `build/libs/`.
