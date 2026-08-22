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

This implementation routes `Matrix.multiply(Matrix)` and the matrix/vector multiplication in `DenseLayer.forward` through the selected backend. CUDA multiplication uses cuBLAS SGEMM through JCuda.

Backpropagation, optimizers, activations, and element-wise matrix operations are still CPU code in this branch.

## GPU memory caching

The CUDA backend now keeps a bounded cache of device allocations for matrices used by multiplication. Reusing the same matrix can therefore reuse its VRAM allocation instead of allocating and uploading it again for every multiply.

Because `Matrix.values` remains public API, callers can still mutate the underlying `float[][]` directly. The CUDA backend validates cached host contents before reuse and re-uploads a matrix when its values have changed, preserving compatibility with existing Synapse code.

CUDA multiplication results are also retained in the cache, so a result that immediately feeds another multiplication can reuse the already-populated device allocation.

The cache is currently capped at 64 matrices and evicts the least-recently-used allocation when full. All cached allocations are released when switching away from CUDA.

A host copy of each multiplication result is still produced because the current public API exposes `Matrix.values` directly. This means the branch no longer pays every upload/allocation repeatedly, but still pays a device-to-host copy after each multiply. Removing that final synchronization cost would require a deeper `Matrix` storage redesign or changing the semantics of direct `values` access.

Mini-batch training is still strongly recommended before treating CUDA benchmarks as representative; the current training loop processes one sample at a time.

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

CUDA-specific tests automatically skip when the complete CUDA/cuBLAS backend cannot initialize. When CUDA is available, the suite checks CPU/CUDA numerical parity, cached-result reuse, and correctness after direct `Matrix.values` mutation.

To build the normal jars:

```bash
./gradlew build
```

To additionally create an experimental bundled runtime jar containing Synapse plus its JCuda dependencies:

```bash
./gradlew fatJar
```

Artifacts are written under `build/libs/`.
