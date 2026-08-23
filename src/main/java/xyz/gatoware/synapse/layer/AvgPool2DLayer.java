package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;

/** A 2D average-pooling layer using flattened channel-first matrices. */
public class AvgPool2DLayer implements Layer {
	private final int inputWidth;
	private final int inputHeight;
	private final int channels;
	private final int poolSize;
	private final int stride;
	private final int outputWidth;
	private final int outputHeight;
	private int lastBatchSize;

	public AvgPool2DLayer(int inputWidth, int inputHeight, int channels, int poolSize) {
		this(inputWidth, inputHeight, channels, poolSize, poolSize);
	}

	public AvgPool2DLayer(int inputWidth, int inputHeight, int channels, int poolSize, int stride) {
		validate(inputWidth, inputHeight, channels, poolSize, stride);
		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.channels = channels;
		this.poolSize = poolSize;
		this.stride = stride;
		this.outputWidth = (inputWidth - poolSize) / stride + 1;
		this.outputHeight = (inputHeight - poolSize) / stride + 1;
	}

	@Override
	public Matrix forward(Matrix input) {
		int expectedRows = inputWidth * inputHeight * channels;
		if (input.rows() != expectedRows || input.columns() <= 0)
			throw new IllegalArgumentException("AvgPool2D input must have " + expectedRows + " rows");
		lastBatchSize = input.columns();
		Matrix output = new Matrix(outputWidth * outputHeight * channels, lastBatchSize);
		float divisor = poolSize * poolSize;

		for (int sample = 0; sample < lastBatchSize; sample++) {
			for (int channel = 0; channel < channels; channel++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						float sum = 0.0f;
						for (int poolY = 0; poolY < poolSize; poolY++)
							for (int poolX = 0; poolX < poolSize; poolX++)
								sum += input.values[inputIndex(channel, outputY * stride + poolY, outputX * stride + poolX)][sample];
						output.values[outputIndex(channel, outputY, outputX)][sample] = sum / divisor;
					}
				}
			}
		}
		return output;
	}

	@Override
	public Matrix backwardInput(Matrix outputGradient) {
		if (lastBatchSize <= 0)
			throw new IllegalStateException("AvgPool2D layer must run forward before backward");
		int outputSize = outputWidth * outputHeight * channels;
		if (outputGradient.rows() != outputSize || outputGradient.columns() != lastBatchSize)
			throw new IllegalArgumentException("AvgPool2D output gradient dimensions do not match the last forward pass");
		Matrix inputGradient = new Matrix(inputWidth * inputHeight * channels, lastBatchSize);
		float divisor = poolSize * poolSize;
		for (int sample = 0; sample < lastBatchSize; sample++) {
			for (int channel = 0; channel < channels; channel++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						float gradient = outputGradient.values[outputIndex(channel, outputY, outputX)][sample] / divisor;
						for (int poolY = 0; poolY < poolSize; poolY++)
							for (int poolX = 0; poolX < poolSize; poolX++)
								inputGradient.values[inputIndex(channel, outputY * stride + poolY, outputX * stride + poolX)][sample] += gradient;
					}
				}
			}
		}
		return inputGradient;
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate) {
		return backwardInput(outputGradient);
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		return backwardInput(outputGradient);
	}

	private int inputIndex(int channel, int y, int x) {
		return (channel * inputHeight + y) * inputWidth + x;
	}

	private int outputIndex(int channel, int y, int x) {
		return (channel * outputHeight + y) * outputWidth + x;
	}

	private static void validate(int width, int height, int channels, int poolSize, int stride) {
		if (width <= 0 || height <= 0 || channels <= 0 || poolSize <= 0 || stride <= 0)
			throw new IllegalArgumentException("Pooling dimensions and stride must be positive");
		if (poolSize > width || poolSize > height)
			throw new IllegalArgumentException("Pool size cannot be larger than the input");
		long inputSize = (long) width * height * channels;
		long outputSize = (long) ((width - poolSize) / stride + 1) * ((height - poolSize) / stride + 1) * channels;
		if (inputSize > Integer.MAX_VALUE || outputSize > Integer.MAX_VALUE)
			throw new IllegalArgumentException("Pooling input or output is too large");
	}

	public int getInputWidth() { return inputWidth; }
	public int getInputHeight() { return inputHeight; }
	public int getChannels() { return channels; }
	public int getPoolSize() { return poolSize; }
	public int getStride() { return stride; }
	public int getOutputWidth() { return outputWidth; }
	public int getOutputHeight() { return outputHeight; }
}
