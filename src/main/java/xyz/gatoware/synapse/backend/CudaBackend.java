package xyz.gatoware.synapse.backend;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import xyz.gatoware.synapse.matrix.Matrix;

/** Experimental CUDA backend backed by JCuda and cuBLAS. */
public final class CudaBackend implements Backend {
	private static final int MAX_CACHED_MATRICES = 16;
	private static final int MAX_POOL_BUFFERS = 32;
	private static final int MAX_POOL_PER_SIZE = 8;
	private static final int KERNEL_BLOCK_SIZE = 256;
	private static final String BIAS_RELU_SOURCE = """
		extern \"C\" __global__ void bias_relu(float* data, const float* bias, int rows, int cols) {
		    int index = blockIdx.x * blockDim.x + threadIdx.x;
		    int count = rows * cols;
		    if (index < count) {
		        int row = index / cols;
		        float value = data[index] + bias[row];
		        data[index] = value > 0.0f ? value : 0.0f;
		    }
		}
		""";

	private final cublasHandle handle = new cublasHandle();
	private final Map<float[][], DeviceBuffer> cache = new IdentityHashMap<>();
	private final ArrayDeque<float[][]> cacheOrder = new ArrayDeque<>();
	private final Map<Integer, ArrayDeque<Pointer>> bufferPool = new HashMap<>();
	private final float[] alphaValue = {1.0f};
	private final float[] betaValue = {0.0f};
	private final Pointer alpha = Pointer.to(alphaValue);
	private final Pointer beta = Pointer.to(betaValue);
	private CUmodule reluModule;
	private CUfunction biasReluFunction;
	private boolean reluKernelAttempted;
	private boolean reluKernelAvailable;
	private int pooledBuffers;
	private boolean closed;

	private static final class DeviceBuffer {
		private final Pointer pointer;
		private final int elements;

		private DeviceBuffer(Pointer pointer, int elements) {
			this.pointer = pointer;
			this.elements = elements;
		}
	}

	public CudaBackend() {
		try {
			JCuda.setExceptionsEnabled(true);
			JCublas2.setExceptionsEnabled(true);
			int[] deviceCount = {0};
			JCuda.cudaGetDeviceCount(deviceCount);
			if (deviceCount[0] <= 0)
				throw new IllegalStateException("No CUDA-capable NVIDIA device was found");
			JCublas2.cublasCreate(handle);
		} catch (Throwable error) {
			throw new IllegalStateException("CUDA backend could not initialize: " + rootMessage(error), error);
		}
	}

	public static boolean isAvailable() {
		cublasHandle probeHandle = new cublasHandle();
		boolean handleCreated = false;
		try {
			JCuda.setExceptionsEnabled(true);
			JCublas2.setExceptionsEnabled(true);
			int[] deviceCount = {0};
			JCuda.cudaGetDeviceCount(deviceCount);
			if (deviceCount[0] <= 0)
				return false;
			JCublas2.cublasCreate(probeHandle);
			handleCreated = true;
			return true;
		} catch (Throwable ignored) {
			return false;
		} finally {
			if (handleCreated) {
				try {
					JCublas2.cublasDestroy(probeHandle);
				} catch (Throwable ignored) {
				}
			}
		}
	}

	@Override
	public synchronized float[][] multiply(float[][] left, float[][] right, int rows, int shared, int columns) {
		ensureOpen();
		DeviceBuffer deviceLeft = getOrUpload(left, rows, shared);
		DeviceBuffer deviceRight = getOrUpload(right, shared, columns);
		DeviceBuffer deviceResult = multiplyDevice(deviceLeft, deviceRight, rows, shared, columns);
		boolean resultOwned = true;
		try {
			float[][] result = download(deviceResult, rows, columns);
			cacheResult(result, deviceResult);
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned)
				release(deviceResult);
		}
	}

	public synchronized boolean supportsResidentRelu() {
		ensureOpen();
		return ensureReluKernel();
	}

	public synchronized Matrix denseReluResident(Matrix weights, Matrix biases, Matrix input) {
		ensureOpen();
		if (!ensureReluKernel())
			throw new IllegalStateException("GPU-resident ReLU kernel is unavailable");
		if (weights.columns() != input.rows() || biases.rows() != weights.rows() || biases.columns() != 1)
			throw new IllegalArgumentException("Incompatible dense layer dimensions");

		int rows = weights.rows();
		int shared = weights.columns();
		int columns = input.columns();
		DeviceBuffer deviceWeights = getOrUpload(weights.values, rows, shared);
		DeviceBuffer deviceInput = getOrUpload(input.values, shared, columns);
		DeviceBuffer deviceBias = getOrUpload(biases.values, rows, 1);
		DeviceBuffer deviceResult = multiplyDevice(deviceWeights, deviceInput, rows, shared, columns);
		boolean resultOwned = true;
		try {
			launchBiasRelu(deviceResult, deviceBias, rows, columns);
			Matrix result = new Matrix(rows, columns);
			cacheResult(result.values, deviceResult);
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned)
				release(deviceResult);
		}
	}

	private DeviceBuffer multiplyDevice(DeviceBuffer left, DeviceBuffer right, int rows, int shared, int columns) {
		DeviceBuffer result = acquire(rows * columns);
		boolean owned = true;
		try {
			if (columns == 1) {
				/* Row-major left is seen by cuBLAS as the column-major transpose.
				 * SGEMV with OP_T therefore computes the original left * vector. */
				JCublas2.cublasSgemv(handle,
					cublasOperation.CUBLAS_OP_T,
					shared, rows,
					alpha,
					left.pointer, shared,
					right.pointer, 1,
					beta,
					result.pointer, 1);
			} else {
				JCublas2.cublasSgemm(handle,
					cublasOperation.CUBLAS_OP_N,
					cublasOperation.CUBLAS_OP_N,
					columns, rows, shared,
					alpha,
					right.pointer, columns,
					left.pointer, shared,
					beta,
					result.pointer, columns);
			}
			owned = false;
			return result;
		} finally {
			if (owned)
				release(result);
		}
	}

	private void launchBiasRelu(DeviceBuffer data, DeviceBuffer bias, int rows, int columns) {
		int[] rowsArg = {rows};
		int[] columnsArg = {columns};
		Pointer kernelParameters = Pointer.to(
			Pointer.to(data.pointer),
			Pointer.to(bias.pointer),
			Pointer.to(rowsArg),
			Pointer.to(columnsArg));
		int elements = rows * columns;
		int blocks = (elements + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
		JCudaDriver.cuLaunchKernel(
			biasReluFunction,
			blocks, 1, 1,
			KERNEL_BLOCK_SIZE, 1, 1,
			0, null,
			kernelParameters, null);
	}

	private boolean ensureReluKernel() {
		if (reluKernelAttempted)
			return reluKernelAvailable;
		reluKernelAttempted = true;
		nvrtcProgram program = new nvrtcProgram();
		try {
			JNvrtc.setExceptionsEnabled(true);
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			JNvrtc.nvrtcCreateProgram(program, BIAS_RELU_SOURCE, "synapse_bias_relu.cu", 0, null, null);
			String[] options = {"--gpu-architecture=compute_52", "--use_fast_math"};
			JNvrtc.nvrtcCompileProgram(program, options.length, options);
			String[] ptx = new String[1];
			JNvrtc.nvrtcGetPTX(program, ptx);
			reluModule = new CUmodule();
			biasReluFunction = new CUfunction();
			JCudaDriver.cuModuleLoadData(reluModule, ptx[0]);
			JCudaDriver.cuModuleGetFunction(biasReluFunction, reluModule, "bias_relu");
			reluKernelAvailable = true;
			return true;
		} catch (Throwable ignored) {
			if (reluModule != null) {
				try {
					JCudaDriver.cuModuleUnload(reluModule);
				} catch (Throwable ignoredCleanup) {
				}
			}
			reluModule = null;
			biasReluFunction = null;
			reluKernelAvailable = false;
			return false;
		} finally {
			try {
				JNvrtc.nvrtcDestroyProgram(program);
			} catch (Throwable ignored) {
			}
		}
	}

	private float[][] download(DeviceBuffer buffer, int rows, int columns) {
		float[] hostResult = new float[rows * columns];
		JCuda.cudaMemcpy(Pointer.to(hostResult), buffer.pointer,
			(long) hostResult.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		return reshape(hostResult, rows, columns);
	}

	private DeviceBuffer getOrUpload(float[][] matrix, int rows, int columns) {
		int elements = rows * columns;
		DeviceBuffer cached = cache.get(matrix);
		if (cached != null && cached.elements == elements) {
			touch(matrix);
			return cached;
		}
		if (cached != null)
			removeCached(matrix, cached);

		DeviceBuffer buffer = acquire(elements);
		boolean owned = true;
		try {
			float[] flattened = flatten(matrix, rows, columns);
			JCuda.cudaMemcpy(buffer.pointer, Pointer.to(flattened),
				(long) elements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
			putCached(matrix, buffer);
			owned = false;
			return buffer;
		} finally {
			if (owned)
				release(buffer);
		}
	}

	private DeviceBuffer acquire(int elements) {
		ArrayDeque<Pointer> available = bufferPool.get(elements);
		if (available != null) {
			Pointer pointer = available.pollFirst();
			if (pointer != null) {
				pooledBuffers--;
				if (available.isEmpty())
					bufferPool.remove(elements);
				return new DeviceBuffer(pointer, elements);
			}
		}
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, (long) elements * Sizeof.FLOAT);
		return new DeviceBuffer(pointer, elements);
	}

	private void release(DeviceBuffer buffer) {
		ArrayDeque<Pointer> available = bufferPool.computeIfAbsent(buffer.elements, ignored -> new ArrayDeque<>());
		if (pooledBuffers < MAX_POOL_BUFFERS && available.size() < MAX_POOL_PER_SIZE) {
			available.addLast(buffer.pointer);
			pooledBuffers++;
			return;
		}
		JCuda.cudaFree(buffer.pointer);
	}

	private void cacheResult(float[][] matrix, DeviceBuffer buffer) {
		DeviceBuffer previous = cache.get(matrix);
		if (previous != null)
			removeCached(matrix, previous);
		putCached(matrix, buffer);
	}

	private void putCached(float[][] matrix, DeviceBuffer buffer) {
		while (cache.size() >= MAX_CACHED_MATRICES)
			evictOldest();
		cache.put(matrix, buffer);
		cacheOrder.addLast(matrix);
	}

	private void touch(float[][] matrix) {
		cacheOrder.remove(matrix);
		cacheOrder.addLast(matrix);
	}

	private void evictOldest() {
		float[][] oldest = cacheOrder.pollFirst();
		if (oldest == null)
			return;
		DeviceBuffer buffer = cache.remove(oldest);
		if (buffer != null)
			release(buffer);
	}

	private void removeCached(float[][] matrix, DeviceBuffer buffer) {
		cache.remove(matrix);
		cacheOrder.remove(matrix);
		release(buffer);
	}

	@Override
	public synchronized void invalidate(float[][] matrix) {
		if (closed || matrix == null)
			return;
		DeviceBuffer buffer = cache.get(matrix);
		if (buffer != null)
			removeCached(matrix, buffer);
	}

	private void ensureOpen() {
		if (closed)
			throw new IllegalStateException("CUDA backend has already been closed");
	}

	private static float[] flatten(float[][] matrix, int rows, int columns) {
		float[] flattened = new float[rows * columns];
		for (int row = 0; row < rows; row++)
			System.arraycopy(matrix[row], 0, flattened, row * columns, columns);
		return flattened;
	}

	private static float[][] reshape(float[] flattened, int rows, int columns) {
		float[][] matrix = new float[rows][columns];
		for (int row = 0; row < rows; row++)
			System.arraycopy(flattened, row * columns, matrix[row], 0, columns);
		return matrix;
	}

	private static String rootMessage(Throwable error) {
		Throwable root = error;
		while (root.getCause() != null)
			root = root.getCause();
		String message = root.getMessage();
		return message == null || message.isBlank() ? root.getClass().getSimpleName() : message;
	}

	@Override
	public synchronized void close() {
		if (closed)
			return;
		if (reluModule != null) {
			try {
				JCudaDriver.cuModuleUnload(reluModule);
			} catch (Throwable ignored) {
			}
			reluModule = null;
			biasReluFunction = null;
		}
		for (DeviceBuffer buffer : cache.values())
			JCuda.cudaFree(buffer.pointer);
		cache.clear();
		cacheOrder.clear();
		for (ArrayDeque<Pointer> pointers : bufferPool.values())
			for (Pointer pointer : pointers)
				JCuda.cudaFree(pointer);
		bufferPool.clear();
		pooledBuffers = 0;
		JCublas2.cublasDestroy(handle);
		closed = true;
	}
}
