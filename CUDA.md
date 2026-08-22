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

`isDeviceAvailable(Devices.CUDA)` verifies the CUDA device/runtime and that cuBLAS can actually be initialized. A machine with a working NVIDIA driver but a missing or incompatible cuBLAS installation therefore correctly reports CUDA as unavailable.

## What is accelerated

This first implementation routes `Matrix.multiply(Matrix)` and the matrix/vector multiplication in `DenseLayer.forward` through the selected backend. CUDA multiplication uses cuBLAS SGEMM through JCuda.

Backpropagation, optimizers, activations, and element-wise matrix operations are still CPU code in this branch.

The CUDA backend currently allocates device memory and transfers data for each multiplication. It is intentionally a correctness/proof-of-concept implementation, not the final performance design. A later version should cache matrix data in VRAM and add mini-batch training before performance comparisons are considered meaningful.

## Requirements

- Java 21
- NVIDIA GPU with CUDA support
- NVIDIA driver
- CUDA 12.x runtime/cuBLAS compatible with JCuda 12.6

JCuda's Java and native binding jars are resolved by Gradle for the current operating system and architecture. The NVIDIA CUDA/cuBLAS libraries themselves must still be present on the system.

The current Arch Linux `cuda` package is CUDA 13.x and provides `libcublas.so.13`, while this experimental backend uses JCuda 12.6 and therefore needs a CUDA 12.x cuBLAS installation. CUDA 13 also removed library support for Maxwell, Pascal, and Volta GPUs, so CUDA 12.x is required for those architectures. A compatible CUDA 12.x installation such as CUDA 12.9 is therefore the intended test environment for this branch.

Useful Linux diagnostics:

```bash
nvidia-smi
nvcc --version
ldconfig -p | grep -E 'libcublas\.so|libcublasLt\.so'
```

For this branch, the important result is that a CUDA 12 installation exposes `libcublas.so.12` and `libcublasLt.so.12` to the JVM.

## Build and test

```bash
git switch cuda
./gradlew test
```

The CUDA parity test automatically skips when the complete CUDA/cuBLAS backend is unavailable.

To build the normal jars:

```bash
./gradlew build
```

To additionally create an experimental bundled runtime jar containing Synapse plus its JCuda dependencies:

```bash
./gradlew fatJar
```

Artifacts are written under `build/libs/`.