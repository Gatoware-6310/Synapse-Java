# CUDA Setup

Synapse 0.1.2 supports CUDA **12.0 through 12.9**. CUDA is optional; CPU remains the default.

For the simplest setup, use CUDA **12.9**.

## Requirements

- Java 21
- An NVIDIA CUDA-capable GPU
- A compatible NVIDIA driver
- CUDA Toolkit 12.x

Synapse includes its JCuda Java dependencies, but the NVIDIA CUDA libraries themselves must be installed on the system.

## Windows

1. Install an NVIDIA driver for your GPU.
2. Install the NVIDIA CUDA Toolkit 12.x. CUDA 12.9 is recommended.
3. Open a new terminal after installation.
4. Verify CUDA:

```powershell
nvidia-smi
nvcc --version
```

The CUDA installer normally sets `CUDA_PATH` and adds the toolkit's `bin` directory to `PATH`. A typical CUDA 12.9 install is located at:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
```

You can verify the required CUDA libraries are visible with:

```powershell
where cublas64_12.dll
where nvrtc64*.dll
```

Then run Synapse normally:

```powershell
.\gradlew.bat test
.\gradlew.bat cudaBenchmark
```

If the CUDA DLLs are not found, make sure the CUDA `bin` directory is on `PATH`, for example:

```powershell
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;$env:Path"
```

## Linux

1. Install an NVIDIA driver for your GPU.
2. Install CUDA Toolkit 12.x. CUDA 12.9 is recommended.
3. Verify CUDA:

```bash
nvidia-smi
nvcc --version
```

A standard CUDA 12.9 installation is commonly located at:

```text
/usr/local/cuda-12.9
```

For a runfile or custom installation, make sure the CUDA binaries and libraries are visible:

```bash
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
```

Then run:

```bash
./gradlew test
./gradlew cudaBenchmark
```

### Side-by-side CUDA versions

If your system CUDA is newer than Synapse supports, you can keep a CUDA 12.x installation separately and point Synapse at it.

For example, if CUDA 12.9 is installed in `~/.local/cuda-12.9`:

```bash
LD_LIBRARY_PATH="$HOME/.local/cuda-12.9/lib64:$LD_LIBRARY_PATH" \
./gradlew cudaBenchmark
```

The important library for the current JCuda backend is `libcublas.so.12`. You can verify it is present with:

```bash
ls /path/to/cuda-12.x/lib64/libcublas.so.12
```

## Using CUDA in Synapse

CPU is used by default. To switch to CUDA:

```java
Synapse.useDevice(Devices.CUDA);
```

You can check availability first:

```java
if (Synapse.isDeviceAvailable(Devices.CUDA)) {
    Synapse.useDevice(Devices.CUDA);
}
```

If CUDA initialization fails with an error mentioning `libcublas.so.12` on Linux or `cublas64_12.dll` on Windows, the CUDA 12.x runtime libraries are either not installed or not visible through `LD_LIBRARY_PATH` / `PATH`.
