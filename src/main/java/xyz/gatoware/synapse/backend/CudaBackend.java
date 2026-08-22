package xyz.gatoware.synapse.backend;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

/** Experimental CUDA backend backed by JCuda and cuBLAS. */
public final class CudaBackend implements Backend {
	private final cublasHandle handle = new cublasHandle();
	private boolean closed;

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
	public float[][] multiply(float[][] left, float[][] right, int rows, int shared, int columns) {
		if (closed)
			throw new IllegalStateException("CUDA backend has already been closed");

		float[] hostLeft = flatten(left, rows, shared);
		float[] hostRight = flatten(right, shared, columns);
		float[] hostResult = new float[rows * columns];

		Pointer deviceLeft = new Pointer();
		Pointer deviceRight = new Pointer();
		Pointer deviceResult = new Pointer();
		boolean leftAllocated = false;
		boolean rightAllocated = false;
		boolean resultAllocated = false;

		long leftBytes = (long) hostLeft.length * Sizeof.FLOAT;
		long rightBytes = (long) hostRight.length * Sizeof.FLOAT;
		long resultBytes = (long) hostResult.length * Sizeof.FLOAT;

		try {
			JCuda.cudaMalloc(deviceLeft, leftBytes);
			leftAllocated = true;
			JCuda.cudaMalloc(deviceRight, rightBytes);
			rightAllocated = true;
			JCuda.cudaMalloc(deviceResult, resultBytes);
			resultAllocated = true;

			JCuda.cudaMemcpy(deviceLeft, Pointer.to(hostLeft), leftBytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaMemcpy(deviceRight, Pointer.to(hostRight), rightBytes, cudaMemcpyKind.cudaMemcpyHostToDevice);

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
				deviceRight,
				columns,
				deviceLeft,
				shared,
				beta,
				deviceResult,
				columns);

			JCuda.cudaMemcpy(Pointer.to(hostResult), deviceResult, resultBytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
			return reshape(hostResult, rows, columns);
		} finally {
			if (leftAllocated)
				JCuda.cudaFree(deviceLeft);
			if (rightAllocated)
				JCuda.cudaFree(deviceRight);
			if (resultAllocated)
				JCuda.cudaFree(deviceResult);
		}
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
	public void close() {
		if (!closed) {
			JCublas2.cublasDestroy(handle);
			closed = true;
		}
	}
}
