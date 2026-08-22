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
import xyz.gatoware.synapse.optimizer.AdaGrad;
import xyz.gatoware.synapse.optimizer.Adam;
import xyz.gatoware.synapse.optimizer.Momentum;
import xyz.gatoware.synapse.optimizer.Optimizer;
import xyz.gatoware.synapse.optimizer.RMSProp;
import xyz.gatoware.synapse.optimizer.SGD;

/** Experimental CUDA backend backed by JCuda and cuBLAS. */
public final class CudaBackend implements Backend {
	private static final int MAX_CACHED_MATRICES = 128;
	private static final int MAX_POOL_BUFFERS = 64;
	private static final int MAX_POOL_PER_SIZE = 16;
	private static final int KERNEL_BLOCK_SIZE = 256;

	private static final String KERNEL_SOURCE = """
		extern \"C\" __global__ void bias_relu(float* data, const float* bias, int rows, int cols) {
		    int index = blockIdx.x * blockDim.x + threadIdx.x;
		    int count = rows * cols;
		    if (index < count) {
		        int row = index / cols;
		        float value = data[index] + bias[row];
		        data[index] = value > 0.0f ? value : 0.0f;
		    }
		}

		extern \"C\" __global__ void relu_backward(
		        float* delta, const float* output, const float* gradient, int count) {
		    int index = blockIdx.x * blockDim.x + threadIdx.x;
		    if (index < count)
		        delta[index] = output[index] > 0.0f ? gradient[index] : 0.0f;
		}

		extern \"C\" __global__ void bias_gradient(
		        float* result, const float* delta, int rows, int cols, float scale) {
		    int row = blockIdx.x * blockDim.x + threadIdx.x;
		    if (row < rows) {
		        float sum = 0.0f;
		        for (int col = 0; col < cols; col++)
		            sum += delta[row * cols + col];
		        result[row] = sum * scale;
		    }
		}

		extern \"C\" __global__ void optimizer_update(
		        float* parameters, const float* gradients, float* state1, float* state2,
		        int count, int kind, float learningRate, float p1, float p2,
		        float epsilon, float correction1, float correction2) {
		    int index = blockIdx.x * blockDim.x + threadIdx.x;
		    if (index >= count) return;
		    float g = gradients[index];
		    if (kind == 0) {
		        parameters[index] -= learningRate * g;
		    } else if (kind == 1) {
		        float v = p1 * state1[index] + g;
		        state1[index] = v;
		        parameters[index] -= learningRate * v;
		    } else if (kind == 2) {
		        float a = state1[index] + g * g;
		        state1[index] = a;
		        parameters[index] -= learningRate * g / (sqrtf(a) + epsilon);
		    } else if (kind == 3) {
		        float a = p1 * state1[index] + (1.0f - p1) * g * g;
		        state1[index] = a;
		        parameters[index] -= learningRate * g / (sqrtf(a) + epsilon);
		    } else if (kind == 4) {
		        float m = p1 * state1[index] + (1.0f - p1) * g;
		        float v = p2 * state2[index] + (1.0f - p2) * g * g;
		        state1[index] = m;
		        state2[index] = v;
		        float mh = m / correction1;
		        float vh = v / correction2;
		        parameters[index] -= learningRate * mh / (sqrtf(vh) + epsilon);
		    }
		}
		""";

	private final cublasHandle handle = new cublasHandle();
	private final Map<float[][], DeviceBuffer> cache = new IdentityHashMap<>();
	private final ArrayDeque<float[][]> cacheOrder = new ArrayDeque<>();
	private final Map<Integer, ArrayDeque<Pointer>> bufferPool = new HashMap<>();
	private final Map<Optimizer, IdentityHashMap<float[][], OptimizerState>> optimizerStates = new IdentityHashMap<>();
	private final float[] alphaValue = {1.0f};
	private final float[] betaValue = {0.0f};
	private final Pointer alpha = Pointer.to(alphaValue);
	private final Pointer beta = Pointer.to(betaValue);
	private CUmodule kernelModule;
	private CUfunction biasReluFunction;
	private CUfunction reluBackwardFunction;
	private CUfunction biasGradientFunction;
	private CUfunction optimizerUpdateFunction;
	private boolean kernelsAttempted;
	private boolean kernelsAvailable;
	private int pooledBuffers;
	private boolean closed;

	private static final class DeviceBuffer {
		private final Pointer pointer;
		private final int elements;
		private boolean deviceDirty;

		private DeviceBuffer(Pointer pointer, int elements) {
			this.pointer = pointer;
			this.elements = elements;
		}
	}

	private static final class OptimizerState {
		private DeviceBuffer first;
		private DeviceBuffer second;
		private int step;
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
				try { JCublas2.cublasDestroy(probeHandle); } catch (Throwable ignored) { }
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
			deviceResult.deviceDirty = false;
			cacheResult(result, deviceResult);
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned) release(deviceResult);
		}
	}

	public synchronized boolean supportsResidentRelu() {
		ensureOpen();
		return ensureKernels();
	}

	/** Computes weights*input + bias followed by ReLU without a host readback. */
	public synchronized Matrix denseReluResident(Matrix weights, Matrix biases, Matrix input) {
		ensureOpen();
		if (!ensureKernels())
			throw new IllegalStateException("CUDA training kernels are unavailable");
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
			deviceResult.deviceDirty = true;
			cacheResult(result.values, deviceResult);
			resultOwned = false;
			return result;
		} finally {
			if (resultOwned) release(deviceResult);
		}
	}

	/** Performs ReLU backward, dense gradients, input-gradient GEMM and optimizer updates on the GPU. */
	public synchronized Matrix denseReluBackwardUpdate(Matrix weights, Matrix biases, Matrix input,
			Matrix output, Matrix outputGradient, Optimizer optimizer, float learningRate) {
		ensureOpen();
		if (!ensureKernels())
			throw new IllegalStateException("CUDA training kernels are unavailable");
		int rows = weights.rows();
		int batch = input.columns();
		DeviceBuffer deviceOutput = getOrUpload(output.values, rows, batch);
		DeviceBuffer deviceGradient = getOrUpload(outputGradient.values, rows, batch);
		DeviceBuffer delta = acquire(rows * batch);
		boolean deltaOwned = true;
		try {
			launchReluBackward(delta, deviceOutput, deviceGradient, rows * batch);
			return denseBackwardUpdateDevice(weights, biases, input, delta, optimizer, learningRate);
		} finally {
			if (deltaOwned) release(delta);
		}
	}

	/** Performs dense backward/update when the activation derivative was calculated on the host (e.g. Softmax). */
	public synchronized Matrix denseBackwardUpdate(Matrix weights, Matrix biases, Matrix input,
			Matrix weightedGradient, Optimizer optimizer, float learningRate) {
		ensureOpen();
		DeviceBuffer delta = getOrUpload(weightedGradient.values, weightedGradient.rows(), weightedGradient.columns());
		return denseBackwardUpdateDevice(weights, biases, input, delta, optimizer, learningRate);
	}

	private Matrix denseBackwardUpdateDevice(Matrix weights, Matrix biases, Matrix input, DeviceBuffer delta,
			Optimizer optimizer, float learningRate) {
		int outputSize = weights.rows();
		int inputSize = weights.columns();
		int batch = input.columns();
		if (delta.elements != outputSize * batch)
			throw new IllegalArgumentException("Gradient dimensions do not match dense layer output");

		DeviceBuffer deviceWeights = getOrUpload(weights.values, outputSize, inputSize);
		DeviceBuffer deviceBiases = getOrUpload(biases.values, outputSize, 1);
		DeviceBuffer deviceInput = getOrUpload(input.values, inputSize, batch);
		DeviceBuffer inputGradient = acquire(inputSize * batch);
		DeviceBuffer weightGradient = acquire(outputSize * inputSize);
		DeviceBuffer biasGradient = acquire(outputSize);
		boolean inputOwned = true;
		try {
			// inputGradient = weights^T * delta. Compute before updating weights.
			JCublas2.cublasSgemm(handle,
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
				batch, inputSize, outputSize,
				alpha,
				delta.pointer, batch,
				deviceWeights.pointer, inputSize,
				beta,
				inputGradient.pointer, batch);

			float scaleValue = 1.0f / batch;
			Pointer scale = Pointer.to(new float[] {scaleValue});
			// weightGradient = delta * input^T, averaged across the batch.
			JCublas2.cublasSgemm(handle,
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
				inputSize, outputSize, batch,
				scale,
				deviceInput.pointer, batch,
				delta.pointer, batch,
				beta,
				weightGradient.pointer, inputSize);
			launchBiasGradient(biasGradient, delta, outputSize, batch, scaleValue);

			updateOptimizer(weights.values, deviceWeights, weightGradient, optimizer, learningRate);
			updateOptimizer(biases.values, deviceBiases, biasGradient, optimizer, learningRate);

			Matrix result = new Matrix(inputSize, batch);
			inputGradient.deviceDirty = true;
			cacheResult(result.values, inputGradient);
			inputOwned = false;
			return result;
		} finally {
			if (inputOwned) release(inputGradient);
			release(weightGradient);
			release(biasGradient);
		}
	}

	private void updateOptimizer(float[][] parameterKey, DeviceBuffer parameters, DeviceBuffer gradients,
			Optimizer optimizer, float learningRate) {
		int kind;
		float p1 = 0.0f;
		float p2 = 0.0f;
		float epsilon = 1e-7f;
		int stateCount = 0;
		if (optimizer instanceof SGD) {
			kind = 0;
		} else if (optimizer instanceof Momentum momentum) {
			kind = 1;
			p1 = momentum.getMomentum();
			stateCount = 1;
		} else if (optimizer instanceof AdaGrad adaGrad) {
			kind = 2;
			epsilon = adaGrad.getEpsilon();
			stateCount = 1;
		} else if (optimizer instanceof RMSProp rmsProp) {
			kind = 3;
			p1 = rmsProp.getDecay();
			epsilon = rmsProp.getEpsilon();
			stateCount = 1;
		} else if (optimizer instanceof Adam adam) {
			kind = 4;
			p1 = adam.getBeta1();
			p2 = adam.getBeta2();
			epsilon = adam.getEpsilon();
			stateCount = 2;
		} else {
			throw new IllegalArgumentException("Optimizer is not supported by CUDA training: " + optimizer.getClass().getName());
		}

		OptimizerState state = null;
		if (stateCount > 0) {
			IdentityHashMap<float[][], OptimizerState> states = optimizerStates.computeIfAbsent(optimizer,
				ignored -> new IdentityHashMap<>());
			state = states.get(parameterKey);
			if (state == null) {
				state = new OptimizerState();
				state.first = acquire(parameters.elements);
				JCuda.cudaMemset(state.first.pointer, 0, (long) parameters.elements * Sizeof.FLOAT);
				if (stateCount == 2) {
					state.second = acquire(parameters.elements);
					JCuda.cudaMemset(state.second.pointer, 0, (long) parameters.elements * Sizeof.FLOAT);
				}
				states.put(parameterKey, state);
			}
			state.step++;
		}

		float correction1 = 1.0f;
		float correction2 = 1.0f;
		if (kind == 4) {
			correction1 = 1.0f - (float) Math.pow(p1, state.step);
			correction2 = 1.0f - (float) Math.pow(p2, state.step);
		}
		launchOptimizerUpdate(parameters, gradients, state, parameters.elements, kind,
			learningRate, p1, p2, epsilon, correction1, correction2);
		parameters.deviceDirty = true;
	}

	public synchronized void resetOptimizer(Optimizer optimizer) {
		IdentityHashMap<float[][], OptimizerState> states = optimizerStates.remove(optimizer);
		if (states == null) return;
		for (OptimizerState state : states.values()) {
			if (state.first != null) release(state.first);
			if (state.second != null) release(state.second);
		}
	}

	/** Synchronizes a matrix whose newest copy is on the GPU back into Matrix.values. */
	public synchronized void materialize(Matrix matrix) {
		if (matrix == null) return;
		DeviceBuffer buffer = cache.get(matrix.values);
		if (buffer == null || !buffer.deviceDirty) return;
		downloadInto(buffer, matrix.values, matrix.rows(), matrix.columns());
		buffer.deviceDirty = false;
	}

	private DeviceBuffer multiplyDevice(DeviceBuffer left, DeviceBuffer right, int rows, int shared, int columns) {
		DeviceBuffer result = acquire(rows * columns);
		boolean owned = true;
		try {
			if (columns == 1) {
				JCublas2.cublasSgemv(handle,
					cublasOperation.CUBLAS_OP_T,
					shared, rows, alpha,
					left.pointer, shared,
					right.pointer, 1,
					beta, result.pointer, 1);
			} else {
				JCublas2.cublasSgemm(handle,
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
					columns, rows, shared, alpha,
					right.pointer, columns,
					left.pointer, shared,
					beta, result.pointer, columns);
			}
			owned = false;
			return result;
		} finally {
			if (owned) release(result);
		}
	}

	private void launchBiasRelu(DeviceBuffer data, DeviceBuffer bias, int rows, int columns) {
		int[] rowsArg = {rows};
		int[] columnsArg = {columns};
		Pointer params = Pointer.to(Pointer.to(data.pointer), Pointer.to(bias.pointer),
			Pointer.to(rowsArg), Pointer.to(columnsArg));
		launch1d(biasReluFunction, rows * columns, params);
	}

	private void launchReluBackward(DeviceBuffer delta, DeviceBuffer output, DeviceBuffer gradient, int count) {
		int[] countArg = {count};
		Pointer params = Pointer.to(Pointer.to(delta.pointer), Pointer.to(output.pointer),
			Pointer.to(gradient.pointer), Pointer.to(countArg));
		launch1d(reluBackwardFunction, count, params);
	}

	private void launchBiasGradient(DeviceBuffer result, DeviceBuffer delta, int rows, int columns, float scale) {
		int[] rowsArg = {rows};
		int[] columnsArg = {columns};
		float[] scaleArg = {scale};
		Pointer params = Pointer.to(Pointer.to(result.pointer), Pointer.to(delta.pointer),
			Pointer.to(rowsArg), Pointer.to(columnsArg), Pointer.to(scaleArg));
		launch1d(biasGradientFunction, rows, params);
	}

	private void launchOptimizerUpdate(DeviceBuffer parameters, DeviceBuffer gradients, OptimizerState state,
			int count, int kind, float learningRate, float p1, float p2, float epsilon,
			float correction1, float correction2) {
		Pointer nullPointer = new Pointer();
		Pointer state1 = state != null && state.first != null ? state.first.pointer : nullPointer;
		Pointer state2 = state != null && state.second != null ? state.second.pointer : nullPointer;
		int[] countArg = {count};
		int[] kindArg = {kind};
		float[] lrArg = {learningRate};
		float[] p1Arg = {p1};
		float[] p2Arg = {p2};
		float[] epsilonArg = {epsilon};
		float[] c1Arg = {correction1};
		float[] c2Arg = {correction2};
		Pointer params = Pointer.to(
			Pointer.to(parameters.pointer), Pointer.to(gradients.pointer),
			Pointer.to(state1), Pointer.to(state2),
			Pointer.to(countArg), Pointer.to(kindArg), Pointer.to(lrArg),
			Pointer.to(p1Arg), Pointer.to(p2Arg), Pointer.to(epsilonArg),
			Pointer.to(c1Arg), Pointer.to(c2Arg));
		launch1d(optimizerUpdateFunction, count, params);
	}

	private void launch1d(CUfunction function, int count, Pointer params) {
		int blocks = (count + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
		JCudaDriver.cuLaunchKernel(function, blocks, 1, 1, KERNEL_BLOCK_SIZE, 1, 1,
			0, null, params, null);
	}

	private boolean ensureKernels() {
		if (kernelsAttempted) return kernelsAvailable;
		kernelsAttempted = true;
		nvrtcProgram program = new nvrtcProgram();
		try {
			JNvrtc.setExceptionsEnabled(true);
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			JNvrtc.nvrtcCreateProgram(program, KERNEL_SOURCE, "synapse_kernels.cu", 0, null, null);
			String[] options = {"--gpu-architecture=compute_52", "--use_fast_math"};
			JNvrtc.nvrtcCompileProgram(program, options.length, options);
			String[] ptx = new String[1];
			JNvrtc.nvrtcGetPTX(program, ptx);
			kernelModule = new CUmodule();
			JCudaDriver.cuModuleLoadData(kernelModule, ptx[0]);
			biasReluFunction = function("bias_relu");
			reluBackwardFunction = function("relu_backward");
			biasGradientFunction = function("bias_gradient");
			optimizerUpdateFunction = function("optimizer_update");
			kernelsAvailable = true;
			return true;
		} catch (Throwable ignored) {
			if (kernelModule != null) {
				try { JCudaDriver.cuModuleUnload(kernelModule); } catch (Throwable ignoredCleanup) { }
			}
			kernelModule = null;
			kernelsAvailable = false;
			return false;
		} finally {
			try { JNvrtc.nvrtcDestroyProgram(program); } catch (Throwable ignored) { }
		}
	}

	private CUfunction function(String name) {
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, kernelModule, name);
		return function;
	}

	private float[][] download(DeviceBuffer buffer, int rows, int columns) {
		float[][] result = new float[rows][columns];
		downloadInto(buffer, result, rows, columns);
		return result;
	}

	private void downloadInto(DeviceBuffer buffer, float[][] destination, int rows, int columns) {
		float[] flat = new float[rows * columns];
		JCuda.cudaMemcpy(Pointer.to(flat), buffer.pointer, (long) flat.length * Sizeof.FLOAT,
			cudaMemcpyKind.cudaMemcpyDeviceToHost);
		for (int row = 0; row < rows; row++)
			System.arraycopy(flat, row * columns, destination[row], 0, columns);
	}

	private DeviceBuffer getOrUpload(float[][] matrix, int rows, int columns) {
		int elements = rows * columns;
		DeviceBuffer cached = cache.get(matrix);
		if (cached != null && cached.elements == elements) {
			touch(matrix);
			return cached;
		}
		if (cached != null) removeCached(matrix, cached);

		DeviceBuffer buffer = acquire(elements);
		boolean owned = true;
		try {
			float[] flattened = flatten(matrix, rows, columns);
			JCuda.cudaMemcpy(buffer.pointer, Pointer.to(flattened), (long) elements * Sizeof.FLOAT,
				cudaMemcpyKind.cudaMemcpyHostToDevice);
			buffer.deviceDirty = false;
			putCached(matrix, buffer);
			owned = false;
			return buffer;
		} finally {
			if (owned) release(buffer);
		}
	}

	private DeviceBuffer acquire(int elements) {
		ArrayDeque<Pointer> available = bufferPool.get(elements);
		if (available != null) {
			Pointer pointer = available.pollFirst();
			if (pointer != null) {
				pooledBuffers--;
				if (available.isEmpty()) bufferPool.remove(elements);
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
		if (previous != null) removeCached(matrix, previous);
		putCached(matrix, buffer);
	}

	private void putCached(float[][] matrix, DeviceBuffer buffer) {
		while (cache.size() >= MAX_CACHED_MATRICES) evictOldest();
		cache.put(matrix, buffer);
		cacheOrder.addLast(matrix);
	}

	private void touch(float[][] matrix) {
		cacheOrder.remove(matrix);
		cacheOrder.addLast(matrix);
	}

	private void evictOldest() {
		float[][] oldest = cacheOrder.pollFirst();
		if (oldest == null) return;
		DeviceBuffer buffer = cache.remove(oldest);
		if (buffer != null) {
			if (buffer.deviceDirty)
				downloadInto(buffer, oldest, oldest.length, oldest.length == 0 ? 0 : oldest[0].length);
			release(buffer);
		}
	}

	private void removeCached(float[][] matrix, DeviceBuffer buffer) {
		cache.remove(matrix);
		cacheOrder.remove(matrix);
		release(buffer);
	}

	@Override
	public synchronized void invalidate(float[][] matrix) {
		if (closed || matrix == null) return;
		DeviceBuffer buffer = cache.get(matrix);
		if (buffer != null) removeCached(matrix, buffer);
	}

	private void ensureOpen() {
		if (closed) throw new IllegalStateException("CUDA backend has already been closed");
	}

	private static float[] flatten(float[][] matrix, int rows, int columns) {
		float[] flattened = new float[rows * columns];
		for (int row = 0; row < rows; row++)
			System.arraycopy(matrix[row], 0, flattened, row * columns, columns);
		return flattened;
	}

	private static String rootMessage(Throwable error) {
		Throwable root = error;
		while (root.getCause() != null) root = root.getCause();
		String message = root.getMessage();
		return message == null || message.isBlank() ? root.getClass().getSimpleName() : message;
	}

	@Override
	public synchronized void close() {
		if (closed) return;
		for (Map.Entry<float[][], DeviceBuffer> entry : cache.entrySet()) {
			DeviceBuffer buffer = entry.getValue();
			if (buffer.deviceDirty) {
				float[][] host = entry.getKey();
				downloadInto(buffer, host, host.length, host.length == 0 ? 0 : host[0].length);
			}
			JCuda.cudaFree(buffer.pointer);
		}
		cache.clear();
		cacheOrder.clear();
		for (IdentityHashMap<float[][], OptimizerState> states : optimizerStates.values())
			for (OptimizerState state : states.values()) {
				if (state.first != null) JCuda.cudaFree(state.first.pointer);
				if (state.second != null) JCuda.cudaFree(state.second.pointer);
			}
		optimizerStates.clear();
		for (ArrayDeque<Pointer> pointers : bufferPool.values())
			for (Pointer pointer : pointers) JCuda.cudaFree(pointer);
		bufferPool.clear();
		pooledBuffers = 0;
		if (kernelModule != null) {
			try { JCudaDriver.cuModuleUnload(kernelModule); } catch (Throwable ignored) { }
			kernelModule = null;
		}
		JCublas2.cublasDestroy(handle);
		closed = true;
	}
}
