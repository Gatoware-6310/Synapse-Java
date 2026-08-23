package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.activation.ActivationFunction;
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

	public Conv2DLayer(int inputWidth, int inputHeight, int inputChannels, int filters, int kernelSize,
			ActivationFunction activationFunction) {
		this(inputWidth, inputHeight, inputChannels, filters, kernelSize, 1, Padding.VALID, activationFunction);
	}

	public Conv2DLayer(int inputWidth, int inputHeight, int inputChannels, int filters, int kernelSize,
			Padding padding, ActivationFunction activationFunction) {
		this(inputWidth, inputHeight, inputChannels, filters, kernelSize, 1, padding, activationFunction);
	}

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
		if (padding == Padding.VALID && (kernelSize > inputWidth || kernelSize > inputHeight))
			throw new IllegalArgumentException("Kernel cannot be larger than the input when using VALID padding");

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

		int batchSize = input.columns();
		int outputSize = outputWidth * outputHeight * filters;
		lastInput = input;
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
								if (inputY < 0 || inputY >= inputHeight)
									continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth)
										continue;
									int inputIndex = inputIndex(channel, inputY, inputX);
									int kernelIndex = kernelIndex(channel, kernelY, kernelX);
									sum += input.values[inputIndex][sample] * kernels.values[filter][kernelIndex];
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
		Matrix weightedGradient = activationGradient(outputGradient);
		return inputGradient(weightedGradient);
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
								if (inputY < 0 || inputY >= inputHeight)
									continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth)
										continue;
									int inputIndex = inputIndex(channel, inputY, inputX);
									int kernelIndex = kernelIndex(channel, kernelY, kernelX);
									kernelGradients.values[filter][kernelIndex] +=
										gradient * lastInput.values[inputIndex][sample];
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

	private Matrix activationGradient(Matrix outputGradient) {
		if (lastInput == null || lastWeighted == null || lastOutput == null)
			throw new IllegalStateException("Conv2D layer must run forward before backward");
		if (outputGradient.rows() != lastOutput.rows() || outputGradient.columns() != lastOutput.columns())
			throw new IllegalArgumentException("Conv2D output gradient dimensions do not match the last forward pass");

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
								if (inputY < 0 || inputY >= inputHeight)
									continue;
								for (int kernelX = 0; kernelX < kernelSize; kernelX++) {
									int inputX = inputOriginX + kernelX;
									if (inputX < 0 || inputX >= inputWidth)
										continue;
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
		if (padding == Padding.VALID)
			return 0;
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

	public int getInputWidth() { return inputWidth; }
	public int getInputHeight() { return inputHeight; }
	public int getInputChannels() { return inputChannels; }
	public int getFilters() { return filters; }
	public int getKernelSize() { return kernelSize; }
	public int getStride() { return stride; }
	public Padding getPadding() { return padding; }
	public int getOutputWidth() { return outputWidth; }
	public int getOutputHeight() { return outputHeight; }
	public Matrix getKernels() { return kernels; }
	public Matrix getBiases() { return biases; }
	public ActivationFunction getActivationFunction() { return activationFunction; }
}
