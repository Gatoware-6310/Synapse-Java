package xyz.gatoware.synapse.backend;

import java.util.ArrayDeque;
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
	private static final int MAX_CACHED_MATRICES = 64;

	private final cublasHandle handle = new cublasHandle();
	private final Map<float[][], DeviceBuffer> cache = new IdentityHashMap<>();
	private final ArrayDeque<float[][]> cacheOrder = new ArrayDeque<>();
	private boolean closed;

	private static final class DeviceBuffer {
		private final Pointer pointer;
		private final int elements;
		private long fingerprint;

		private DeviceBuffer(Pointer pointer, int elements, long fingerprint) {
			this.pointer = pointer;
			this.elements = elements;
			this.fingerprint = fingerprint;
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
		long resultBytes = (long) resultElements * Sizeof.FLOAT;
		Pointer deviceResult = new Pointer();
		boolean resultOwned = false;

		try {
			JCuda.cudaMalloc(deviceResult, resultBytes);
			resultOwned = true;

			Pointer alpha = Pointer.to(new float[] {1.0f});
			Pointer beta = Pointer.to(new float[] {0.0f});

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
				deviceResult,
				columns);

			float[] hostResult = new float[resultElements];
			JCuda.cudaMemcpy(Pointer.to(hostResult), deviceResult, resultBytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			float[][] result = reshape(hostResult, rows, columns);

			cacheResult(result, deviceResult, resultElements, fingerprint(result, rows, columns));
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned)
				JCuda.cudaFree(deviceResult);
		}
	}

	private DeviceBuffer getOrUpload(float[][] matrix, int rows, int columns) {
		int elements = rows * columns;
		long fingerprint = fingerprint(matrix, rows, columns);
		DeviceBuffer cached = cache.get(matrix);

		if (cached != null && cached.elements == elements) {
			touch(matrix);
			if (cached.fingerprint != fingerprint) {
				float[] flattened = flatten(matrix, rows, columns);
				long bytes = (long) elements * Sizeof.FLOAT;
				JCuda.cudaMemcpy(cached.pointer, Pointer.to(flattened), bytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
				cached.fingerprint = fingerprint;
			}
			return cached;
		}

		if (cached != null)
			removeCached(matrix, cached);

		float[] flattened = flatten(matrix, rows, columns);
		Pointer pointer = new Pointer();
		long bytes = (long) elements * Sizeof.FLOAT;
		JCuda.cudaMalloc(pointer, bytes);
		boolean allocated = true;
		try {
			JCuda.cudaMemcpy(pointer, Pointer.to(flattened), bytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
			DeviceBuffer buffer = new DeviceBuffer(pointer, elements, fingerprint);
			putCached(matrix, buffer);
			allocated = false;
			return buffer;
		} finally {
			if (allocated)
				JCuda.cudaFree(pointer);
		}
	}

	private void cacheResult(float[][] matrix, Pointer pointer, int elements, long fingerprint) {
		DeviceBuffer previous = cache.get(matrix);
		if (previous != null)
			removeCached(matrix, previous);
		putCached(matrix, new DeviceBuffer(pointer, elements, fingerprint));
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
			JCuda.cudaFree(buffer.pointer);
	}

	private void removeCached(float[][] matrix, DeviceBuffer buffer) {
		cache.remove(matrix);
		cacheOrder.remove(matrix);
		JCuda.cudaFree(buffer.pointer);
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

	/* Matrix.values is intentionally public in Synapse. Since callers may mutate
	 * it directly, use a cheap content fingerprint to detect host-side changes
	 * before reusing a cached device allocation. */
	private static long fingerprint(float[][] matrix, int rows, int columns) {
		long hash = 0xcbf29ce484222325L;
		hash ^= rows;
		hash *= 0x100000001b3L;
		hash ^= columns;
		hash *= 0x100000001b3L;
		for (int row = 0; row < rows; row++) {
			float[] values = matrix[row];
			for (int column = 0; column < columns; column++) {
				hash ^= Float.floatToRawIntBits(values[column]);
				hash *= 0x100000001b3L;
			}
		}
		return hash;
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
		if (!closed) {
			for (DeviceBuffer buffer : cache.values())
				JCuda.cudaFree(buffer.pointer);
			cache.clear();
			cacheOrder.clear();
			JCublas2.cublasDestroy(handle);
			closed = true;
		}
	}
}
