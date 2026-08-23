package xyz.gatoware.synapse.backend;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;

/** CUDA kernels for convolution and pooling layers. */
public final class CudaCnnOps {
	private static final int BLOCK_SIZE = 256;

	private static final String SOURCE = """
		extern "C" __global__ void conv_relu_forward(
			const float* input, const float* kernels, const float* biases, float* output,
			int inW, int inH, int inC, int filters, int k, int stride,
			int outW, int outH, int padLeft, int padTop, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = filters * outW * outH;
			if (index >= outputRows * batch) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int filter = outputIndex / spatial;
			int pos = outputIndex - filter * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			int originY = oy * stride - padTop;
			int originX = ox * stride - padLeft;
			int kernelValues = inC * k * k;
			float sum = biases[filter];
			for (int c = 0; c < inC; c++) {
				for (int ky = 0; ky < k; ky++) {
					int iy = originY + ky;
					if (iy < 0 || iy >= inH) continue;
					for (int kx = 0; kx < k; kx++) {
						int ix = originX + kx;
						if (ix < 0 || ix >= inW) continue;
						int inputRow = (c * inH + iy) * inW + ix;
						int kernelIndex = (c * k + ky) * k + kx;
						sum += input[inputRow * batch + sample] * kernels[filter * kernelValues + kernelIndex];
					}
				}
			}
			output[index] = sum > 0.0f ? sum : 0.0f;
		}

		extern "C" __global__ void conv_relu_input_gradient(
			const float* output, const float* outputGradient, const float* kernels, float* inputGradient,
			int inW, int inH, int inC, int filters, int k, int stride,
			int outW, int outH, int padLeft, int padTop, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = filters * outW * outH;
			if (index >= outputRows * batch) return;
			if (output[index] <= 0.0f) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int filter = outputIndex / spatial;
			int pos = outputIndex - filter * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			int originY = oy * stride - padTop;
			int originX = ox * stride - padLeft;
			int kernelValues = inC * k * k;
			float gradient = outputGradient[index];
			for (int c = 0; c < inC; c++) {
				for (int ky = 0; ky < k; ky++) {
					int iy = originY + ky;
					if (iy < 0 || iy >= inH) continue;
					for (int kx = 0; kx < k; kx++) {
						int ix = originX + kx;
						if (ix < 0 || ix >= inW) continue;
						int inputRow = (c * inH + iy) * inW + ix;
						int kernelIndex = (c * k + ky) * k + kx;
						atomicAdd(&inputGradient[inputRow * batch + sample],
							kernels[filter * kernelValues + kernelIndex] * gradient);
					}
				}
			}
		}

		extern "C" __global__ void conv_relu_kernel_gradient(
			const float* input, const float* output, const float* outputGradient, float* kernelGradient,
			int inW, int inH, int inC, int filters, int k, int stride,
			int outW, int outH, int padLeft, int padTop, int batch) {
			int parameter = blockIdx.x * blockDim.x + threadIdx.x;
			int kernelValues = inC * k * k;
			if (parameter >= filters * kernelValues) return;
			int filter = parameter / kernelValues;
			int kernelIndex = parameter - filter * kernelValues;
			int c = kernelIndex / (k * k);
			int rem = kernelIndex - c * k * k;
			int ky = rem / k;
			int kx = rem - ky * k;
			float sum = 0.0f;
			for (int oy = 0; oy < outH; oy++) {
				int iy = oy * stride - padTop + ky;
				if (iy < 0 || iy >= inH) continue;
				for (int ox = 0; ox < outW; ox++) {
					int ix = ox * stride - padLeft + kx;
					if (ix < 0 || ix >= inW) continue;
					int inputRow = (c * inH + iy) * inW + ix;
					int outputRow = (filter * outH + oy) * outW + ox;
					for (int sample = 0; sample < batch; sample++) {
						int outIndex = outputRow * batch + sample;
						if (output[outIndex] > 0.0f)
							sum += input[inputRow * batch + sample] * outputGradient[outIndex];
					}
				}
			}
			kernelGradient[parameter] = sum / (float)batch;
		}

		extern "C" __global__ void conv_relu_bias_gradient(
			const float* output, const float* outputGradient, float* biasGradient,
			int filters, int outW, int outH, int batch) {
			int filter = blockIdx.x * blockDim.x + threadIdx.x;
			if (filter >= filters) return;
			float sum = 0.0f;
			for (int oy = 0; oy < outH; oy++)
				for (int ox = 0; ox < outW; ox++) {
					int outputRow = (filter * outH + oy) * outW + ox;
					for (int sample = 0; sample < batch; sample++) {
						int index = outputRow * batch + sample;
						if (output[index] > 0.0f) sum += outputGradient[index];
					}
				}
			biasGradient[filter] = sum / (float)batch;
		}

		extern "C" __global__ void maxpool_forward(
			const float* input, float* output, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = channels * outW * outH;
			if (index >= outputRows * batch) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int channel = outputIndex / spatial;
			int pos = outputIndex - channel * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			int firstRow = (channel * inH + oy * stride) * inW + ox * stride;
			float maximum = input[firstRow * batch + sample];
			for (int py = 0; py < pool; py++)
				for (int px = 0; px < pool; px++) {
					int row = (channel * inH + oy * stride + py) * inW + ox * stride + px;
					float value = input[row * batch + sample];
					if (value > maximum) maximum = value;
				}
			output[index] = maximum;
		}

		extern "C" __global__ void maxpool_backward(
			const float* input, const float* outputGradient, float* inputGradient,
			int inW, int inH, int channels, int pool, int stride, int outW, int outH, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = channels * outW * outH;
			if (index >= outputRows * batch) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int channel = outputIndex / spatial;
			int pos = outputIndex - channel * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			int maxRow = (channel * inH + oy * stride) * inW + ox * stride;
			float maximum = input[maxRow * batch + sample];
			for (int py = 0; py < pool; py++)
				for (int px = 0; px < pool; px++) {
					int row = (channel * inH + oy * stride + py) * inW + ox * stride + px;
					float value = input[row * batch + sample];
					if (value > maximum) { maximum = value; maxRow = row; }
				}
			atomicAdd(&inputGradient[maxRow * batch + sample], outputGradient[index]);
		}

		extern "C" __global__ void avgpool_forward(
			const float* input, float* output, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = channels * outW * outH;
			if (index >= outputRows * batch) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int channel = outputIndex / spatial;
			int pos = outputIndex - channel * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			float sum = 0.0f;
			for (int py = 0; py < pool; py++)
				for (int px = 0; px < pool; px++) {
					int row = (channel * inH + oy * stride + py) * inW + ox * stride + px;
					sum += input[row * batch + sample];
				}
			output[index] = sum / (float)(pool * pool);
		}

		extern "C" __global__ void avgpool_backward(
			const float* outputGradient, float* inputGradient,
			int inW, int inH, int channels, int pool, int stride, int outW, int outH, int batch) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int outputRows = channels * outW * outH;
			if (index >= outputRows * batch) return;
			int sample = index % batch;
			int outputIndex = index / batch;
			int spatial = outW * outH;
			int channel = outputIndex / spatial;
			int pos = outputIndex - channel * spatial;
			int oy = pos / outW;
			int ox = pos - oy * outW;
			float gradient = outputGradient[index] / (float)(pool * pool);
			for (int py = 0; py < pool; py++)
				for (int px = 0; px < pool; px++) {
					int row = (channel * inH + oy * stride + py) * inW + ox * stride + px;
					atomicAdd(&inputGradient[row * batch + sample], gradient);
				}
		}
		""";

	private static CUmodule module;
	private static CUfunction convForward;
	private static CUfunction convInputGradient;
	private static CUfunction convKernelGradient;
	private static CUfunction convBiasGradient;
	private static CUfunction maxPoolForward;
	private static CUfunction maxPoolBackward;
	private static CUfunction avgPoolForward;
	private static CUfunction avgPoolBackward;
	private static boolean attempted;
	private static boolean available;

	private CudaCnnOps() { }

	/** Returns whether the CUDA CNN kernels are available. */
	public static synchronized boolean isAvailable() {
		if (!attempted) initialize();
		return available;
	}

	private static void initialize() {
		attempted = true;
		nvrtcProgram program = new nvrtcProgram();
		try {
			JCuda.setExceptionsEnabled(true);
			JNvrtc.setExceptionsEnabled(true);
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			JNvrtc.nvrtcCreateProgram(program, SOURCE, "synapse_cnn.cu", 0, null, null);
			String[] options = {"--gpu-architecture=compute_52", "--use_fast_math"};
			JNvrtc.nvrtcCompileProgram(program, options.length, options);
			String[] ptx = new String[1];
			JNvrtc.nvrtcGetPTX(program, ptx);
			module = new CUmodule();
			JCudaDriver.cuModuleLoadData(module, ptx[0]);
			convForward = function("conv_relu_forward");
			convInputGradient = function("conv_relu_input_gradient");
			convKernelGradient = function("conv_relu_kernel_gradient");
			convBiasGradient = function("conv_relu_bias_gradient");
			maxPoolForward = function("maxpool_forward");
			maxPoolBackward = function("maxpool_backward");
			avgPoolForward = function("avgpool_forward");
			avgPoolBackward = function("avgpool_backward");
			available = true;
		} catch (Throwable ignored) {
			available = false;
			if (module != null) {
				try { JCudaDriver.cuModuleUnload(module); } catch (Throwable cleanupIgnored) { }
				module = null;
			}
		} finally {
			try { JNvrtc.nvrtcDestroyProgram(program); } catch (Throwable ignored) { }
		}
	}

	private static CUfunction function(String name) {
		CUfunction result = new CUfunction();
		JCudaDriver.cuModuleGetFunction(result, module, name);
		return result;
	}

	/** Runs a batched ReLU convolution on CUDA. */
	public static Matrix convReluForward(Matrix input, Matrix kernels, Matrix biases,
			int inW, int inH, int inC, int filters, int kernelSize, int stride,
			int outW, int outH, int padLeft, int padTop) {
		requireAvailable();
		int batch = input.columns();
		int outputRows = filters * outW * outH;
		Pointer dInput = upload(input);
		Pointer dKernels = upload(kernels);
		Pointer dBiases = upload(biases);
		Pointer dOutput = allocate((long) outputRows * batch);
		try {
			Pointer params = Pointer.to(Pointer.to(dInput), Pointer.to(dKernels), Pointer.to(dBiases), Pointer.to(dOutput),
				intArg(inW), intArg(inH), intArg(inC), intArg(filters), intArg(kernelSize), intArg(stride),
				intArg(outW), intArg(outH), intArg(padLeft), intArg(padTop), intArg(batch));
			launch(convForward, outputRows * batch, params);
			return download(dOutput, outputRows, batch);
		} finally {
			free(dInput, dKernels, dBiases, dOutput);
		}
	}

	/** Runs convolution backpropagation and parameter updates on CUDA. */
	public static Matrix convReluBackwardUpdate(Matrix input, Matrix output, Matrix outputGradient,
			Matrix kernels, Matrix biases, int inW, int inH, int inC, int filters,
			int kernelSize, int stride, int outW, int outH, int padLeft, int padTop,
			Optimizer optimizer, float learningRate) {
		requireAvailable();
		int batch = input.columns();
		int inputRows = inW * inH * inC;
		int outputRows = filters * outW * outH;
		int kernelValues = inC * kernelSize * kernelSize;
		Pointer dInput = upload(input);
		Pointer dOutput = upload(output);
		Pointer dOutputGradient = upload(outputGradient);
		Pointer dKernels = upload(kernels);
		Pointer dInputGradient = allocate((long) inputRows * batch);
		Pointer dKernelGradient = allocate((long) filters * kernelValues);
		Pointer dBiasGradient = allocate(filters);
		try {
			JCuda.cudaMemset(dInputGradient, 0, (long) inputRows * batch * Sizeof.FLOAT);
			Pointer inputParams = Pointer.to(Pointer.to(dOutput), Pointer.to(dOutputGradient), Pointer.to(dKernels),
				Pointer.to(dInputGradient), intArg(inW), intArg(inH), intArg(inC), intArg(filters),
				intArg(kernelSize), intArg(stride), intArg(outW), intArg(outH), intArg(padLeft), intArg(padTop), intArg(batch));
			launch(convInputGradient, outputRows * batch, inputParams);

			Pointer kernelParams = Pointer.to(Pointer.to(dInput), Pointer.to(dOutput), Pointer.to(dOutputGradient),
				Pointer.to(dKernelGradient), intArg(inW), intArg(inH), intArg(inC), intArg(filters),
				intArg(kernelSize), intArg(stride), intArg(outW), intArg(outH), intArg(padLeft), intArg(padTop), intArg(batch));
			launch(convKernelGradient, filters * kernelValues, kernelParams);

			Pointer biasParams = Pointer.to(Pointer.to(dOutput), Pointer.to(dOutputGradient), Pointer.to(dBiasGradient),
				intArg(filters), intArg(outW), intArg(outH), intArg(batch));
			launch(convBiasGradient, filters, biasParams);

			Matrix kernelGradient = download(dKernelGradient, filters, kernelValues);
			Matrix biasGradient = download(dBiasGradient, filters, 1);
			Matrix inputGradient = download(dInputGradient, inputRows, batch);
			optimizer.update(kernels, kernelGradient, learningRate);
			optimizer.update(biases, biasGradient, learningRate);
			kernels.markDirty();
			biases.markDirty();
			return inputGradient;
		} finally {
			free(dInput, dOutput, dOutputGradient, dKernels, dInputGradient, dKernelGradient, dBiasGradient);
		}
	}

	/** Runs batched max pooling on CUDA. */
	public static Matrix maxPoolForward(Matrix input, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH) {
		return poolForward(input, inW, inH, channels, pool, stride, outW, outH, true);
	}

	/** Runs max-pooling backpropagation on CUDA. */
	public static Matrix maxPoolBackward(Matrix input, Matrix outputGradient, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH) {
		requireAvailable();
		int batch = input.columns();
		int inputRows = inW * inH * channels;
		int outputRows = outW * outH * channels;
		Pointer dInput = upload(input);
		Pointer dGradient = upload(outputGradient);
		Pointer dInputGradient = allocate((long) inputRows * batch);
		try {
			JCuda.cudaMemset(dInputGradient, 0, (long) inputRows * batch * Sizeof.FLOAT);
			Pointer params = Pointer.to(Pointer.to(dInput), Pointer.to(dGradient), Pointer.to(dInputGradient),
				intArg(inW), intArg(inH), intArg(channels), intArg(pool), intArg(stride), intArg(outW), intArg(outH), intArg(batch));
			launch(maxPoolBackward, outputRows * batch, params);
			return download(dInputGradient, inputRows, batch);
		} finally {
			free(dInput, dGradient, dInputGradient);
		}
	}

	/** Runs batched average pooling on CUDA. */
	public static Matrix avgPoolForward(Matrix input, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH) {
		return poolForward(input, inW, inH, channels, pool, stride, outW, outH, false);
	}

	/** Runs average-pooling backpropagation on CUDA. */
	public static Matrix avgPoolBackward(Matrix outputGradient, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH) {
		requireAvailable();
		int batch = outputGradient.columns();
		int inputRows = inW * inH * channels;
		int outputRows = outW * outH * channels;
		Pointer dGradient = upload(outputGradient);
		Pointer dInputGradient = allocate((long) inputRows * batch);
		try {
			JCuda.cudaMemset(dInputGradient, 0, (long) inputRows * batch * Sizeof.FLOAT);
			Pointer params = Pointer.to(Pointer.to(dGradient), Pointer.to(dInputGradient),
				intArg(inW), intArg(inH), intArg(channels), intArg(pool), intArg(stride), intArg(outW), intArg(outH), intArg(batch));
			launch(avgPoolBackward, outputRows * batch, params);
			return download(dInputGradient, inputRows, batch);
		} finally {
			free(dGradient, dInputGradient);
		}
	}

	private static Matrix poolForward(Matrix input, int inW, int inH, int channels,
			int pool, int stride, int outW, int outH, boolean max) {
		requireAvailable();
		int batch = input.columns();
		int outputRows = outW * outH * channels;
		Pointer dInput = upload(input);
		Pointer dOutput = allocate((long) outputRows * batch);
		try {
			Pointer params = Pointer.to(Pointer.to(dInput), Pointer.to(dOutput), intArg(inW), intArg(inH),
				intArg(channels), intArg(pool), intArg(stride), intArg(outW), intArg(outH), intArg(batch));
			launch(max ? maxPoolForward : avgPoolForward, outputRows * batch, params);
			return download(dOutput, outputRows, batch);
		} finally {
			free(dInput, dOutput);
		}
	}

	private static void requireAvailable() {
		if (!isAvailable()) throw new IllegalStateException("CUDA CNN kernels are unavailable");
	}

	private static Pointer upload(Matrix matrix) {
		float[] values = new float[matrix.rows() * matrix.columns()];
		for (int row = 0; row < matrix.rows(); row++)
			System.arraycopy(matrix.values[row], 0, values, row * matrix.columns(), matrix.columns());
		Pointer pointer = allocate(values.length);
		JCuda.cudaMemcpy(pointer, Pointer.to(values), (long) values.length * Sizeof.FLOAT,
			cudaMemcpyKind.cudaMemcpyHostToDevice);
		return pointer;
	}

	private static Matrix download(Pointer pointer, int rows, int columns) {
		float[] values = new float[rows * columns];
		JCuda.cudaMemcpy(Pointer.to(values), pointer, (long) values.length * Sizeof.FLOAT,
			cudaMemcpyKind.cudaMemcpyDeviceToHost);
		Matrix result = new Matrix(rows, columns);
		for (int row = 0; row < rows; row++)
			System.arraycopy(values, row * columns, result.values[row], 0, columns);
		return result;
	}

	private static Pointer allocate(long elements) {
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, elements * Sizeof.FLOAT);
		return pointer;
	}

	private static Pointer intArg(int value) {
		return Pointer.to(new int[] {value});
	}

	private static void launch(CUfunction function, int count, Pointer params) {
		int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
		JCudaDriver.cuLaunchKernel(function, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, null, params, null);
	}

	private static void free(Pointer... pointers) {
		for (Pointer pointer : pointers)
			if (pointer != null) JCuda.cudaFree(pointer);
	}
}
