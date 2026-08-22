package xyz.gatoware.synapse.backend;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

/** Experimental CUDA backend backed by JCuda and cuBLAS. */
public final class CudaBackend implements Backend {
	/* Keep the hot cache deliberately small. Old entries are still safe because
	 * their host values remain authoritative and can simply be uploaded again. */
	private static final int MAX_CACHED_MATRICES = 16;
	private static final int MAX_POOL_BUFFERS = 32;
	private static final int MAX_POOL_PER_SIZE = 8;

	private final cublasHandle handle = new cublasHandle();
	private final Map<float[][], DeviceBuffer> cache = new IdentityHashMap<>();
	private final ArrayDeque<float[][]> cacheOrder = new ArrayDeque<>();
	private final Map<Integer, ArrayDeque<Pointer>> bufferPool = new HashMap<>();
	private final float[] alphaValue = {1.0f};
	private final float[] betaValue = {0.0f};
	private final Pointer alpha = Pointer.to(alphaValue);
	private final Pointer beta = Pointer.to(betaValue);
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

	/** Creates and initializes a CUDA backend.
	 * @throws IllegalStateException if CUDA or cuBLAS cannot be initialized
	 */
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
			throw new IllegalStateException(
				"CUDA backend could not initialize: " + rootMessage(error), error);
		}
	}

	/** Checks whether the complete CUDA backend can be initialized.
	 * This verifies both the CUDA runtime/device and the cuBLAS native library.
	 * @return whether CUDA is available
	 */
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
					// Availability probing must never leak native cleanup errors.
				}
			}
		}
	}

	@Override
	public synchronized float[][] multiply(float[][] left, float[][] right, int rows, int shared, int columns) {
		if (closed)
			throw new IllegalStateException("CUDA backend has already been closed");

		DeviceBuffer deviceLeft = getOrUpload(left, rows, shared);
		DeviceBuffer deviceRight = getOrUpload(right, shared, columns);
		int resultElements = rows * columns;
		DeviceBuffer deviceResult = acquire(resultElements);
		boolean resultOwned = true;

		try {
			/* cuBLAS is column-major. Swapping the row-major operands computes
			 * C^T = B^T A^T without a separate transpose. */
			JCublas2.cublasSgemm(handle,
				cublasOperation.CUBLAS_OP_N,
				cublasOperation.CUBLAS_OP_N,
				columns,
				rows,
				shared,
				alpha,
				deviceRight.pointer,
				columns,
				deviceLeft.pointer,
				shared,
				beta,
				deviceResult.pointer,
				columns);

			float[] hostResult = new float[resultElements];
			long resultBytes = (long) resultElements * Sizeof.FLOAT;
			JCuda.cudaMemcpy(Pointer.to(hostResult), deviceResult.pointer, resultBytes,
				cudaMemcpyKind.cudaMemcpyDeviceToHost);
			float[][] result = reshape(hostResult, rows, columns);

			cacheResult(result, deviceResult);
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned)
				release(deviceResult);
		}
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
			long bytes = (long) elements * Sizeof.FLOAT;
			JCuda.cudaMemcpy(buffer.pointer, Pointer.to(flattened), bytes,
				cudaMemcpyKind.cudaMemcpyHostToDevice);
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
