package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.backend.CudaCnnOps;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;

/** A 2D max-pooling layer using flattened channel-first matrices. */
public class MaxPool2DLayer implements Layer {
	private final int inputWidth;
	private final int inputHeight;
	private final int channels;
	private final int poolSize;
	private final int stride;
	private final int outputWidth;
	private final int outputHeight;
	private int[][] lastMaxIndices;
	private int lastBatchSize;
	private Matrix lastInput;
	private boolean lastForwardCuda;

	/** Creates a max-pooling layer whose stride matches the pooling size.
	 * @param inputWidth width of each input image or feature map
	 * @param inputHeight height of each input image or feature map
	 * @param channels number of input channels
	 * @param poolSize width and height of each square pooling region
	 */
	public MaxPool2DLayer(int inputWidth, int inputHeight, int channels, int poolSize) {
		this(inputWidth, inputHeight, channels, poolSize, poolSize);
	}

	/** Creates a max-pooling layer.
	 * @param inputWidth width of each input image or feature map
	 * @param inputHeight height of each input image or feature map
	 * @param channels number of input channels
	 * @param poolSize width and height of each square pooling region
	 * @param stride number of pixels the pooling region moves each step
	 */
	public MaxPool2DLayer(int inputWidth, int inputHeight, int channels, int poolSize, int stride) {
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
			throw new IllegalArgumentException("MaxPool2D input must have " + expectedRows + " rows");
		lastInput = input;
		lastBatchSize = input.columns();
		if (canUseCuda()) {
			lastForwardCuda = true;
			lastMaxIndices = null;
			return CudaCnnOps.maxPoolForward(input, inputWidth, inputHeight, channels,
				poolSize, stride, outputWidth, outputHeight);
		}

		lastForwardCuda = false;
		int outputSize = outputWidth * outputHeight * channels;
		lastMaxIndices = new int[outputSize][lastBatchSize];
		Matrix output = new Matrix(outputSize, lastBatchSize);
		for (int sample = 0; sample < lastBatchSize; sample++) {
			for (int channel = 0; channel < channels; channel++) {
				for (int outputY = 0; outputY < outputHeight; outputY++) {
					for (int outputX = 0; outputX < outputWidth; outputX++) {
						int firstInput = inputIndex(channel, outputY * stride, outputX * stride);
						float max = input.values[firstInput][sample];
						int maxIndex = firstInput;
						for (int poolY = 0; poolY < poolSize; poolY++) {
							for (int poolX = 0; poolX < poolSize; poolX++) {
								int inputIndex = inputIndex(channel, outputY * stride + poolY, outputX * stride + poolX);
								float value = input.values[inputIndex][sample];
								if (value > max) {
									max = value;
									maxIndex = inputIndex;
								}
							}
						}
						int outputIndex = outputIndex(channel, outputY, outputX);
						output.values[outputIndex][sample] = max;
						lastMaxIndices[outputIndex][sample] = maxIndex;
					}
				}
			}
		}
		return output;
	}

	@Override
	public Matrix backwardInput(Matrix outputGradient) {
		int outputSize = outputWidth * outputHeight * channels;
		if (lastInput == null)
			throw new IllegalStateException("MaxPool2D layer must run forward before backward");
		if (outputGradient.rows() != outputSize || outputGradient.columns() != lastBatchSize)
			throw new IllegalArgumentException("MaxPool2D output gradient dimensions do not match the last forward pass");
		if (lastForwardCuda && canUseCuda())
			return CudaCnnOps.maxPoolBackward(lastInput, outputGradient, inputWidth, inputHeight, channels,
				poolSize, stride, outputWidth, outputHeight);
		if (lastMaxIndices == null)
			throw new IllegalStateException("MaxPool2D layer must run forward before backward");
		Matrix inputGradient = new Matrix(inputWidth * inputHeight * channels, lastBatchSize);
		for (int output = 0; output < outputSize; output++)
			for (int sample = 0; sample < lastBatchSize; sample++)
				inputGradient.values[lastMaxIndices[output][sample]][sample] += outputGradient.values[output][sample];
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

	private boolean canUseCuda() {
		return Synapse.backend() instanceof CudaBackend && CudaCnnOps.isAvailable();
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

	/** Returns the input width.
	 * @return the input width
	 */
	public int getInputWidth() { return inputWidth; }
	/** Returns the input height.
	 * @return the input height
	 */
	public int getInputHeight() { return inputHeight; }
	/** Returns the number of channels.
	 * @return the number of channels
	 */
	public int getChannels() { return channels; }
	/** Returns the pooling-region size.
	 * @return the pooling-region size
	 */
	public int getPoolSize() { return poolSize; }
	/** Returns the pooling stride.
	 * @return the pooling stride
	 */
	public int getStride() { return stride; }
	/** Returns the output width.
	 * @return the output width
	 */
	public int getOutputWidth() { return outputWidth; }
	/** Returns the output height.
	 * @return the output height
	 */
	public int getOutputHeight() { return outputHeight; }
}
