package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.backend.CudaCnnOps;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;
import xyz.gatoware.synapse.optimizer.SGD;

/** A 2D convolution layer using flattened channel-first matrices. */
public class Conv2DLayer implements Layer {
	private final int inputWidth;
	private final int inputHeight;
	private final int inputChannels;
	private final int filters;
	private final int kernelSize;
	private final int stride;
	private final Padding padding;
	private final ActivationFunction activationFunction;
	private final int outputWidth;
	private final int outputHeight;
	private final int kernelValues;
	private final Matrix kernels;
	private final Matrix biases;
	private Matrix lastInput;
	private Matrix lastWeighted;
	private Matrix lastOutput;
	private boolean lastForwardCuda;

	/** Creates a convolution layer with a stride of 1 and no padding.
	 * @param inputWidth width of each input image or feature map
	 * @param inputHeight height of each input image or feature map
	 * @param inputChannels number of input channels
	 * @param filters number of filters learned by the layer
	 * @param kernelSize width and height of each square filter
	 * @param activationFunction activation function applied to the convolution output
	 */
	public Conv2DLayer(int inputWidth, int inputHeight, int inputChannels, int filters, int kernelSize,
			ActivationFunction activationFunction) {
		this(inputWidth, inputHeight, inputChannels, filters, kernelSize, 1, Padding.NONE, activationFunction);
	}

	/** Creates a convolution layer with a stride of 1.
	 * @param inputWidth width of each input image or feature map
	 * @param inputHeight height of each input image or feature map
	 * @param inputChannels number of input channels
	 * @param filters number of filters learned by the layer
	 * @param kernelSize width and height of each square filter
	 * @param padding padding mode used around the input
	 * @param activationFunction activation function applied to the convolution output
	 */
	public Conv2DLayer(int inputWidth, int inputHeight, int inputChannels, int filters, int kernelSize,
			Padding padding, ActivationFunction activationFunction) {
		this(inputWidth, inputHeight, inputChannels, filters, kernelSize, 1, padding, activationFunction);
	}

	/** Creates a convolution layer.
	 * @param inputWidth width of each input image or feature map
	 * @param inputHeight height of each input image or feature map
	 * @param inputChannels number of input channels
	 * @param filters number of filters learned by the layer
	 * @param kernelSize width and height of each square filter
	 * @param stride number of pixels each filter moves per step
	 * @param padding padding mode used around the input
	 * @param activationFunction activation function applied to the convolution output
	 */
	public Conv2DLayer(int inputWidth, int inputHeight, int inputChannels, int filters, int kernelSize,
			int stride, Padding padding, ActivationFunction activationFunction) {
		if (inputWidth <= 0 || inputHeight <= 0 || inputChannels <= 0)
			throw new IllegalArgumentException("Input dimensions must be positive");
		if (filters <= 0 || kernelSize <= 0 || stride <= 0)
			throw new IllegalArgumentException("Filters, kernel size, and stride must be positive");
		if (padding == null)
			throw new IllegalArgumentException("Padding cannot be null");
		if (activationFunction == null)
			throw new IllegalArgumentException("Activation function cannot be null");
		if (padding == Padding.NONE && (kernelSize > inputWidth || kernelSize > inputHeight))
			throw new IllegalArgumentException("Kernel cannot be larger than the input when using no padding");

		long kernelValuesLong = (long) inputChannels * kernelSize * kernelSize;
		if (kernelValuesLong > Integer.MAX_VALUE)
			throw new IllegalArgumentException("Convolution kernel is too large");

		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.inputChannels = inputChannels;
		this.filters = filters;
		this.kernelSize = kernelSize;
		this.stride = stride;
		this.padding = padding;
		this.activationFunction = activationFunction;
		this.outputWidth = outputSize(inputWidth, kernelSize, stride, padding);
		this.outputHeight = outputSize(inputHeight, kernelSize, stride, padding);
		this.kernelValues = (int) kernelValuesLong;
		validateFlattenedSize((long) inputWidth * inputHeight * inputChannels, "Input");
		validateFlattenedSize((long) outputWidth * outputHeight * filters, "Output");

		this.kernels = new Matrix(filters, kernelValues);
		this.biases = new Matrix(filters, 1);
		float scale = (float) Math.sqrt(2.0 / kernelValues);
		for (int filter = 0; filter < filters; filter++)
			for (int value = 0; value < kernelValues; value++)
				kernels.values[filter][value] = (float) ((Math.random() * 2.0 - 1.0) * scale);
	}

	@Override
	public Matrix forward(Matrix input) {
		int expectedRows = inputWidth * inputHeight * inputChannels;
		if (input.rows() != expectedRows || input.columns() <= 0)
			throw new IllegalArgumentException("Conv2D input must have " + expectedRows + " rows");

		lastInput = input;
		if (canUseCuda()) {
			lastOutput = CudaCnnOps.convReluForward(input, kernels, biases, inputWidth, inputHeight, inputChannels,
				filters, kernelSize, stride, outputWidth, outputHeight,
				padBefore(inputWidth, outputWidth), padBefore(inputHeight, outputHeight));
			lastWeighted = lastOutput;
			lastForwardCuda = true;
			return lastOutput;
		}

		lastForwardCuda = false;
		int batchSize = input.columns();
		int outputSize = outputWidth * outputHeight * filters;
		lastWeighted = new Matrix(outputSize, batchSize);
		lastOutput = new Matrix(outputSize, batchSize);
		int padLeft = padBefore(inputWidth, outputWidth);
		int padTop = padBefore(inputHeight, outputHeight);
		float[] weighted = new float[outputSize];
		float[] activated = new float[outputSize];

		for (int sample = 0; sample < batchSize; sample++) {
			for (int filter = 0; filter < filters; filter++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						float sum = biases.values[filter][0];
						int inputOriginY = outputY * stride - padTop;
						int inputOriginX = outputX * stride - padLeft;
						for (int channel = 0; channel < inputChannels; channel++) {
							for (int kernelY = 0; kernelY < kernelSize; kernelY++) {
								int inputY = inputOriginY + kernelY;
								if (inputY < 0 || inputY >= inputHeight) continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth) continue;
									sum += input.values[inputIndex(channel, inputY, inputX)][sample]
										* kernels.values[filter][kernelIndex(channel, kernelY, kernelX)];
								}
							}
						}
						int outputIndex = outputIndex(filter, outputY, outputX);
						weighted[outputIndex] = sum;
						lastWeighted.values[outputIndex][sample] = sum;
					}
				}
			}
			activationFunction.apply(weighted, activated);
			for (int value = 0; value < outputSize; value++)
				lastOutput.values[value][sample] = activated[value];
		}
		return lastOutput;
	}

	@Override
	public Matrix backwardInput(Matrix outputGradient) {
		return inputGradient(activationGradient(outputGradient));
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate) {
		return backward(outputGradient, learningRate, new SGD());
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
		if (optimizer == null)
			throw new IllegalArgumentException("Optimizer cannot be null");
		validateBackwardGradient(outputGradient);

		if (lastForwardCuda && canUseCuda()) {
			return CudaCnnOps.convReluBackwardUpdate(lastInput, lastOutput, outputGradient, kernels, biases,
				inputWidth, inputHeight, inputChannels, filters, kernelSize, stride, outputWidth, outputHeight,
				padBefore(inputWidth, outputWidth), padBefore(inputHeight, outputHeight), optimizer, learningRate);
		}

		Matrix weightedGradient = activationGradient(outputGradient);
		Matrix inputGradient = inputGradient(weightedGradient);
		Matrix kernelGradients = new Matrix(filters, kernelValues);
		Matrix biasGradients = new Matrix(filters, 1);
		int batchSize = lastInput.columns();
		float scale = 1.0f / batchSize;
		int padLeft = padBefore(inputWidth, outputWidth);
		int padTop = padBefore(inputHeight, outputHeight);

		for (int sample = 0; sample < batchSize; sample++) {
			for (int filter = 0; filter < filters; filter++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						int outputIndex = outputIndex(filter, outputY, outputX);
						float gradient = weightedGradient.values[outputIndex][sample];
						biasGradients.values[filter][0] += gradient;
						int inputOriginY = outputY * stride - padTop;
						int inputOriginX = outputX * stride - padLeft;
						for (int channel = 0; channel < inputChannels; channel++) {
							for (int kernelY = 0; kernelY < kernelSize; kernelY++) {
								int inputY = inputOriginY + kernelY;
								if (inputY < 0 || inputY >= inputHeight) continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth) continue;
									kernelGradients.values[filter][kernelIndex(channel, kernelY, kernelX)] +=
										gradient * lastInput.values[inputIndex(channel, inputY, inputX)][sample];
								}
							}
						}
					}
				}
			}
		}

		for (int filter = 0; filter < filters; filter++) {
			biasGradients.values[filter][0] *= scale;
			for (int value = 0; value < kernelValues; value++)
				kernelGradients.values[filter][value] *= scale;
		}
		optimizer.update(kernels, kernelGradients, learningRate);
		optimizer.update(biases, biasGradients, learningRate);
		kernels.markDirty();
		biases.markDirty();
		return inputGradient;
	}

	private boolean canUseCuda() {
		return activationFunction instanceof ReLU
			&& Synapse.backend() instanceof CudaBackend
			&& CudaCnnOps.isAvailable();
	}

	private void validateBackwardGradient(Matrix outputGradient) {
		if (lastInput == null || lastOutput == null)
			throw new IllegalStateException("Conv2D layer must run forward before backward");
		if (outputGradient.rows() != lastOutput.rows() || outputGradient.columns() != lastOutput.columns())
			throw new IllegalArgumentException("Conv2D output gradient dimensions do not match the last forward pass");
	}

	private Matrix activationGradient(Matrix outputGradient) {
		validateBackwardGradient(outputGradient);
		if (Synapse.backend() instanceof CudaBackend cuda) {
			cuda.materialize(lastWeighted);
			cuda.materialize(lastOutput);
			cuda.materialize(outputGradient);
		}
		int outputSize = lastOutput.rows();
		int batchSize = lastOutput.columns();
		Matrix result = new Matrix(outputSize, batchSize);
		float[] weighted = new float[outputSize];
		float[] output = new float[outputSize];
		float[] upstream = new float[outputSize];
		float[] activatedGradient = new float[outputSize];
		for (int sample = 0; sample < batchSize; sample++) {
			for (int value = 0; value < outputSize; value++) {
				weighted[value] = lastWeighted.values[value][sample];
				output[value] = lastOutput.values[value][sample];
				upstream[value] = outputGradient.values[value][sample];
			}
			activationFunction.backward(weighted, output, upstream, activatedGradient);
			for (int value = 0; value < outputSize; value++)
				result.values[value][sample] = activatedGradient[value];
		}
		return result;
	}

	private Matrix inputGradient(Matrix weightedGradient) {
		if (Synapse.backend() instanceof CudaBackend cuda) {
			cuda.materialize(lastInput);
			cuda.materialize(kernels);
			cuda.materialize(weightedGradient);
		}
		int batchSize = lastInput.columns();
		Matrix inputGradient = new Matrix(inputWidth * inputHeight * inputChannels, batchSize);
		int padLeft = padBefore(inputWidth, outputWidth);
		int padTop = padBefore(inputHeight, outputHeight);
		for (int sample = 0; sample < batchSize; sample++) {
			for (int filter = 0; filter < filters; filter++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						float gradient = weightedGradient.values[outputIndex(filter, outputY, outputX)][sample];
						int inputOriginY = outputY * stride - padTop;
						int inputOriginX = outputX * stride - padLeft;
						for (int channel = 0; channel < inputChannels; channel++) {
							for (int kernelY = 0; kernelY < kernelSize; kernelY++) {
								int inputY = inputOriginY + kernelY;
								if (inputY < 0 || inputY >= inputHeight) continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth) continue;
									inputGradient.values[inputIndex(channel, inputY, inputX)][sample] +=
										kernels.values[filter][kernelIndex(channel, kernelY, kernelX)] * gradient;
								}
							}
						}
					}
				}
			}
		}
		return inputGradient;
	}

	private int padBefore(int inputSize, int outputSize) {
		if (padding == Padding.NONE) return 0;
		int totalPadding = Math.max((outputSize - 1) * stride + kernelSize - inputSize, 0);
		return totalPadding / 2;
	}

	private int inputIndex(int channel, int y, int x) {
		return (channel * inputHeight + y) * inputWidth + x;
	}

	private int outputIndex(int filter, int y, int x) {
		return (filter * outputHeight + y) * outputWidth + x;
	}

	private int kernelIndex(int channel, int y, int x) {
		return (channel * kernelSize + y) * kernelSize + x;
	}

	private static int outputSize(int inputSize, int kernelSize, int stride, Padding padding) {
		if (padding == Padding.SAME)
			return (inputSize + stride - 1) / stride;
		return (inputSize - kernelSize) / stride + 1;
	}

	private static void validateFlattenedSize(long size, String name) {
		if (size <= 0 || size > Integer.MAX_VALUE)
			throw new IllegalArgumentException(name + " is too large");
	}

	private void materializeParameters() {
		if (Synapse.backend() instanceof CudaBackend cuda) {
			cuda.materialize(kernels);
			cuda.materialize(biases);
		}
	}

	/** Returns the input width.
	 * @return the input width
	 */
	public int getInputWidth() { return inputWidth; }

	/** Returns the input height.
	 * @return the input height
	 */
	public int getInputHeight() { return inputHeight; }

	/** Returns the number of input channels.
	 * @return the number of input channels
	 */
	public int getInputChannels() { return inputChannels; }

	/** Returns the number of learned filters.
	 * @return the number of filters
	 */
	public int getFilters() { return filters; }

	/** Returns the width and height of each square filter.
	 * @return the filter size
	 */
	public int getKernelSize() { return kernelSize; }

	/** Returns the convolution stride.
	 * @return the stride
	 */
	public int getStride() { return stride; }

	/** Returns the padding mode.
	 * @return the padding mode
	 */
	public Padding getPadding() { return padding; }

	/** Returns the output width.
	 * @return the output width
	 */
	public int getOutputWidth() { return outputWidth; }

	/** Returns the output height.
	 * @return the output height
	 */
	public int getOutputHeight() { return outputHeight; }

	/** Returns the learned convolution kernels.
	 * @return the kernel matrix
	 */
	public Matrix getKernels() {
		materializeParameters();
		return kernels;
	}

	/** Returns the learned filter biases.
	 * @return the bias matrix
	 */
	public Matrix getBiases() {
		materializeParameters();
		return biases;
	}

	/** Returns the activation function used by this layer.
	 * @return the activation function
	 */
	public ActivationFunction getActivationFunction() { return activationFunction; }
}
