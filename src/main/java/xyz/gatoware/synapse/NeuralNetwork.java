package xyz.gatoware.synapse;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.activation.ELU;
import xyz.gatoware.synapse.activation.GELU;
import xyz.gatoware.synapse.activation.LeakyReLU;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.SiLU;
import xyz.gatoware.synapse.activation.Sigmoid;
import xyz.gatoware.synapse.activation.Softmax;
import xyz.gatoware.synapse.activation.Swish;
import xyz.gatoware.synapse.activation.Tanh;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.layer.AvgPool2DLayer;
import xyz.gatoware.synapse.layer.Conv2DLayer;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.layer.MaxPool2DLayer;
import xyz.gatoware.synapse.layer.Padding;
import xyz.gatoware.synapse.loss.LossFunction;
import xyz.gatoware.synapse.loss.SparseCategoricalCrossEntropy;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Adam;
import xyz.gatoware.synapse.optimizer.Optimizer;

/** A lightweight neural network composed of layers. */
public class NeuralNetwork {
	private static final int FILE_MAGIC = 0x534E4E31; // SNN1
	private static final int LEGACY_FILE_VERSION = 1;
	private static final int FILE_VERSION = 2;
	private static final int MAX_LAYERS = 1_000_000;
	private static final int MAX_MATRIX_DIMENSION = 1_000_000;
	private static final int LAYER_DENSE = 1;
	private static final int LAYER_CONV2D = 2;
	private static final int LAYER_MAX_POOL_2D = 3;
	private static final int LAYER_AVG_POOL_2D = 4;

	private List<Layer> layers = new ArrayList<>();
	private float lastLoss = Float.NaN;

	public NeuralNetwork() {
		// here for compatibility
	}

	public NeuralNetwork(Layer[] layerList) {
		for (Layer l : layerList)
			addLayer(l);
	}

	public NeuralNetwork(int inputs, int layerSize, int layers, int outputs) {
		if (inputs <= 0)
			throw new IllegalArgumentException("inputs must be greater than 0");
		if (layerSize <= 0)
			throw new IllegalArgumentException("layerSize must be greater than 0");
		if (layers < 0)
			throw new IllegalArgumentException("layers cannot be negative");
		if (outputs <= 0)
			throw new IllegalArgumentException("outputs must be greater than 0");

		if (layers == 0) {
			addLayer(new DenseLayer(inputs, outputs, new Softmax()));
			return;
		}

		addLayer(new DenseLayer(inputs, layerSize, new ReLU()));
		for (int i = 1; i < layers; i++)
			addLayer(new DenseLayer(layerSize, layerSize, new ReLU()));
		addLayer(new DenseLayer(layerSize, outputs, new Softmax()));
	}

	public void addLayer(Layer layer) {
		layers.add(layer);
	}

	/** Runs the input through every layer and returns the output. */
	public Matrix forward(Matrix input) {
		Matrix output = input;
		for (int i = 0; i < layers.size(); i++) {
			Layer layer = layers.get(i);
			boolean hiddenLayer = i < layers.size() - 1;
			if (hiddenLayer && layer instanceof DenseLayer denseLayer && denseLayer.canForwardCudaResident())
				output = denseLayer.forwardCudaResident(output);
			else
				output = layer.forward(output);
		}
		return output;
	}

	/** Runs the input through the given zero-based layer and returns that layer's activations. */
	public Matrix forwardTo(Matrix input, int layerIndex) {
		validateLayerIndex(layerIndex);
		Matrix output = input;
		for (int i = 0; i <= layerIndex; i++)
			output = layers.get(i).forward(output);
		return output;
	}

	/** Returns the gradient with respect to the network input without updating parameters. */
	public Matrix inputGradient(Matrix input, int layerIndex, Matrix gradient) {
		Matrix activation = forwardTo(input, layerIndex);
		if (gradient == null)
			throw new IllegalArgumentException("Gradient cannot be null");
		if (gradient.rows() != activation.rows() || gradient.columns() != activation.columns())
			throw new IllegalArgumentException("Gradient dimensions must match the selected layer output");
		Matrix result = gradient;
		for (int i = layerIndex; i >= 0; i--)
			result = layers.get(i).backwardInput(result);
		return result;
	}

	private void validateLayerIndex(int layerIndex) {
		if (layerIndex < 0 || layerIndex >= layers.size())
			throw new IllegalArgumentException("Layer index is out of range");
	}

	private Matrix forwardTraining(Matrix input) {
		Matrix output = input;
		for (Layer layer : layers)
			output = layer.forward(output);
		return output;
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate) {
		fit(dataset, lossFunction, epochs, learningRate, new Adam(), false);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final int batchSize) {
		fit(dataset, lossFunction, epochs, learningRate, new Adam(), false, batchSize);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final Optimizer optimizer) {
		fit(dataset, lossFunction, epochs, learningRate, optimizer, false);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final Optimizer optimizer, final int batchSize) {
		fit(dataset, lossFunction, epochs, learningRate, optimizer, false, batchSize);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final boolean logging) {
		fit(dataset, lossFunction, epochs, learningRate, new Adam(), logging);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final boolean logging, final int batchSize) {
		fit(dataset, lossFunction, epochs, learningRate, new Adam(), logging, batchSize);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final Optimizer optimizer, final boolean logging) {
		validateTrainingArguments(dataset, lossFunction, epochs, learningRate, optimizer);
		if (canUseCudaTraining()) {
			fitCuda(dataset, lossFunction, epochs, learningRate, optimizer, logging, Synapse.getCudaBatchSize());
			return;
		}
		fitCpu(dataset, lossFunction, epochs, learningRate, optimizer, logging, 1);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final Optimizer optimizer, final boolean logging, final int batchSize) {
		validateTrainingArguments(dataset, lossFunction, epochs, learningRate, optimizer);
		if (batchSize <= 0)
			throw new IllegalArgumentException("Batch size must be positive");
		if (canUseCudaTraining()) {
			fitCuda(dataset, lossFunction, epochs, learningRate, optimizer, logging, batchSize);
			return;
		}
		fitCpu(dataset, lossFunction, epochs, learningRate, optimizer, logging, batchSize);
	}

	private void fitCpu(Dataset dataset, LossFunction lossFunction, int epochs, float learningRate,
			Optimizer optimizer, boolean logging, int batchSize) {
		int[] order = makeOrder(dataset.size());
		Random random = new Random();
		int featureCount = dataset.getInputs().columns();

		for (int epoch = 0; epoch < epochs; epoch++) {
			shuffle(order, random);
			float totalLoss = 0.0f;

			for (int start = 0; start < order.length; start += batchSize) {
				int currentBatchSize = Math.min(batchSize, order.length - start);
				Matrix batchInput = new Matrix(featureCount, currentBatchSize);
				for (int sample = 0; sample < currentBatchSize; sample++) {
					int index = order[start + sample];
					for (int feature = 0; feature < featureCount; feature++)
						batchInput.values[feature][sample] = dataset.getInputs().values[index][feature];
				}

				Matrix output = forwardTraining(batchInput);
				Matrix outputGradient = new Matrix(output.rows(), currentBatchSize);
				for (int sample = 0; sample < currentBatchSize; sample++) {
					int index = order[start + sample];
					Matrix predicted = rowFromColumn(output, sample);
					Matrix actual = rowVector(dataset.getTarget(index));
					totalLoss += lossFunction.calculate(predicted, actual);
					Matrix sampleGradient = lossFunction.gradient(predicted, actual);
					if (sampleGradient.rows() != 1 || sampleGradient.columns() != output.rows())
						throw new IllegalStateException("Loss gradient must match the network output width");
					for (int neuron = 0; neuron < output.rows(); neuron++)
						outputGradient.values[neuron][sample] = sampleGradient.values[0][neuron];
				}

				Matrix gradient = outputGradient;
				for (int layer = layers.size() - 1; layer >= 0; layer--)
					gradient = layers.get(layer).backward(gradient, learningRate, optimizer);
			}

			lastLoss = totalLoss / dataset.size();
			if (logging)
				System.out.printf("Epoch %d loss: %.6f accuracy: %.2f%%%n", epoch + 1, lastLoss,
					accuracy(dataset) * 100.0f);
		}
	}

	/** CUDA training keeps the standard Dense/ReLU/Softmax classifier path on the GPU.
	 * For Softmax + SparseCategoricalCrossEntropy, the final affine transform, bias,
	 * stable Softmax, cross-entropy derivative, final-layer backward, and optimizer
	 * update are fused into the CUDA backend without materializing probabilities.
	 */
	private void fitCuda(Dataset dataset, LossFunction lossFunction, int epochs, float learningRate,
			Optimizer optimizer, boolean logging, int cudaBatchSize) {
		int[] order = makeOrder(dataset.size());
		Random random = new Random();
		int featureCount = dataset.getInputs().columns();
		int finalLayerIndex = layers.size() - 1;
		boolean fusedSoftmaxCrossEntropy = lossFunction instanceof SparseCategoricalCrossEntropy
			&& layers.get(finalLayerIndex) instanceof DenseLayer finalDense
			&& finalDense.getActivationFunction() instanceof Softmax
			&& dataset.getTargets().columns() == 1;

		try {
			for (int epoch = 0; epoch < epochs; epoch++) {
				shuffle(order, random);
				float totalLoss = 0.0f;

				for (int start = 0; start < order.length; start += cudaBatchSize) {
					int batchSize = Math.min(cudaBatchSize, order.length - start);
					Matrix batchInput = new Matrix(featureCount, batchSize);
					for (int sample = 0; sample < batchSize; sample++) {
						int index = order[start + sample];
						for (int feature = 0; feature < featureCount; feature++)
							batchInput.values[feature][sample] = dataset.getInputs().values[index][feature];
					}

					if (fusedSoftmaxCrossEntropy) {
						Matrix hiddenOutput = batchInput;
						for (int layerIndex = 0; layerIndex < finalLayerIndex; layerIndex++)
							hiddenOutput = ((DenseLayer) layers.get(layerIndex)).forwardCudaResident(hiddenOutput);

						int[] targets = new int[batchSize];
						for (int sample = 0; sample < batchSize; sample++) {
							int index = order[start + sample];
							float classId = dataset.getTargets().values[index][0];
							int target = (int) classId;
							if (classId != target)
								throw new IllegalArgumentException("Actual matrix must contain integer class IDs");
							targets[sample] = target;
						}

						DenseLayer finalLayer = (DenseLayer) layers.get(finalLayerIndex);
						CudaBackend cuda = (CudaBackend) Synapse.backend();
						CudaBackend.SoftmaxTrainingResult step = cuda.denseSoftmaxCrossEntropyBackwardUpdate(
							finalLayer.getWeights(), finalLayer.getBiases(), hiddenOutput,
							targets, optimizer, learningRate);
						totalLoss += step.loss() * batchSize;
						Matrix gradient = step.inputGradient();
						for (int layerIndex = finalLayerIndex - 1; layerIndex >= 0; layerIndex--)
							gradient = layers.get(layerIndex).backward(gradient, learningRate, optimizer);
					} else {
						Matrix output = batchInput;
						for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
							DenseLayer dense = (DenseLayer) layers.get(layerIndex);
							if (layerIndex < finalLayerIndex)
								output = dense.forwardCudaResident(output);
							else
								output = dense.forward(output);
						}

						Matrix outputGradient = new Matrix(output.rows(), batchSize);
						for (int sample = 0; sample < batchSize; sample++) {
							int index = order[start + sample];
							Matrix predicted = rowFromColumn(output, sample);
							Matrix actual = rowVector(dataset.getTarget(index));
							totalLoss += lossFunction.calculate(predicted, actual);
							Matrix sampleGradient = lossFunction.gradient(predicted, actual);
							if (sampleGradient.rows() != 1 || sampleGradient.columns() != output.rows())
								throw new IllegalStateException("Loss gradient must match the network output width");
							for (int neuron = 0; neuron < output.rows(); neuron++)
								outputGradient.values[neuron][sample] = sampleGradient.values[0][neuron];
						}
						Matrix gradient = outputGradient;
						for (int layerIndex = finalLayerIndex; layerIndex >= 0; layerIndex--)
							gradient = layers.get(layerIndex).backward(gradient, learningRate, optimizer);
					}
				}

				lastLoss = totalLoss / dataset.size();
				if (logging)
					System.out.printf("Epoch %d loss: %.6f accuracy: %.2f%%%n", epoch + 1, lastLoss,
						accuracy(dataset) * 100.0f);
			}
		} finally {
			materializeCudaParameters();
		}
	}

	private boolean canUseCudaTraining() {
		if (!(Synapse.backend() instanceof CudaBackend) || layers.isEmpty())
			return false;
		for (int i = 0; i < layers.size(); i++) {
			if (!(layers.get(i) instanceof DenseLayer dense))
				return false;
			if (i < layers.size() - 1 && !dense.canForwardCudaResident())
				return false;
		}
		return true;
	}

	private void materializeCudaParameters() {
		for (Layer layer : layers)
			if (layer instanceof DenseLayer dense)
				dense.materializeParameters();
	}

	private static int[] makeOrder(int size) {
		int[] order = new int[size];
		for (int i = 0; i < size; i++)
			order[i] = i;
		return order;
	}

	private void validateTrainingArguments(Dataset dataset, LossFunction lossFunction, int epochs,
			float learningRate, Optimizer optimizer) {
		if (dataset == null)
			throw new IllegalArgumentException("Dataset cannot be null");
		if (lossFunction == null)
			throw new IllegalArgumentException("Loss function cannot be null");
		if (optimizer == null)
			throw new IllegalArgumentException("Optimizer cannot be null");
		if (dataset.size() == 0)
			throw new IllegalArgumentException("Dataset cannot be empty");
		if (layers.isEmpty())
			throw new IllegalStateException("Neural network must contain at least one layer");
		if (epochs <= 0)
			throw new IllegalArgumentException("Epochs must be positive");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, new Adam(), false);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final int batchSize) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, new Adam(), false, batchSize);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final Optimizer optimizer) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, false);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate,
			final Optimizer optimizer, final int batchSize) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, false, batchSize);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final boolean logging) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, new Adam(), logging);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate,
			final boolean logging, final int batchSize) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, new Adam(), logging, batchSize);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final Optimizer optimizer,
			final boolean logging) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, logging);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final Optimizer optimizer,
			final boolean logging, final int batchSize) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, logging, batchSize);
	}

	public float getLastLoss() {
		return lastLoss;
	}

	public int predict(Matrix input) {
		Matrix output = forward(input);
		if (output.rows() == 0 || output.columns() == 0 || (output.rows() != 1 && output.columns() != 1))
			throw new IllegalStateException("Network output must be a non-empty vector");

		int prediction = 0;
		float highest = output.values[0][0];
		int length = Math.max(output.rows(), output.columns());
		for (int i = 1; i < length; i++) {
			float value = output.rows() == 1 ? output.values[0][i] : output.values[i][0];
			if (value > highest) {
				highest = value;
				prediction = i;
			}
		}
		return prediction;
	}

	public float accuracy(Dataset dataset) {
		if (dataset == null)
			throw new IllegalArgumentException("Dataset cannot be null");
		if (dataset.size() == 0)
			throw new IllegalArgumentException("Dataset cannot be empty");

		int correct = 0;
		for (int i = 0; i < dataset.size(); i++) {
			int target = (int) dataset.getTarget(i).values[0][0];
			if (predict(dataset.getInput(i)) == target)
				correct++;
		}
		return (float) correct / dataset.size();
	}

	private static Matrix rowVector(Matrix vector) {
		if (vector.rows() == 1)
			return vector.copy();
		if (vector.columns() != 1)
			throw new IllegalArgumentException("Expected a vector");
		Matrix result = new Matrix(1, vector.rows());
		for (int i = 0; i < vector.rows(); i++)
			result.values[0][i] = vector.values[i][0];
		return result;
	}

	private static Matrix rowFromColumn(Matrix matrix, int column) {
		Matrix result = new Matrix(1, matrix.rows());
		for (int row = 0; row < matrix.rows(); row++)
			result.values[0][row] = matrix.values[row][column];
		return result;
	}

	private static Matrix columnVector(Matrix vector) {
		if (vector.columns() == 1)
			return vector.copy();
		if (vector.rows() != 1)
			throw new IllegalArgumentException("Expected a vector");
		Matrix result = new Matrix(vector.columns(), 1);
		for (int i = 0; i < vector.columns(); i++)
			result.values[i][0] = vector.values[0][i];
		return result;
	}

	private static void shuffle(int[] values, Random random) {
		for (int i = values.length - 1; i > 0; i--) {
			int other = random.nextInt(i + 1);
			int value = values[i];
			values[i] = values[other];
			values[other] = value;
		}
	}

	public void save(String filename) throws IOException {
		save(Path.of(filename));
	}

	public void save(Path path) throws IOException {
		materializeCudaParameters();
		try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
			output.writeInt(FILE_MAGIC);
			output.writeInt(FILE_VERSION);
			output.writeInt(layers.size());
			for (Layer layer : layers)
				writeLayer(output, layer);
		}
	}

	public static NeuralNetwork load(String filename) throws IOException {
		return load(Path.of(filename));
	}

	public static NeuralNetwork load(Path path) throws IOException {
		try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
			if (input.readInt() != FILE_MAGIC)
				throw new IOException("Not a Synapse neural network file");
			int version = input.readInt();
			if (version != LEGACY_FILE_VERSION && version != FILE_VERSION)
				throw new IOException("Unsupported Synapse neural network file version");

			int layerCount = input.readInt();
			if (layerCount < 0 || layerCount > MAX_LAYERS)
				throw new IOException("Invalid layer count");

			NeuralNetwork network = new NeuralNetwork();
			for (int i = 0; i < layerCount; i++) {
				if (version == LEGACY_FILE_VERSION)
					network.addLayer(readLegacyDenseLayer(input));
				else
					network.addLayer(readLayer(input));
			}
			return network;
		}
	}

	private static void writeLayer(DataOutputStream output, Layer layer) throws IOException {
		if (layer instanceof DenseLayer dense) {
			output.writeByte(LAYER_DENSE);
			writeMatrix(output, dense.getWeights());
			writeMatrix(output, dense.getBiases());
			writeActivation(output, dense.getActivationFunction());
			return;
		}
		if (layer instanceof Conv2DLayer conv) {
			output.writeByte(LAYER_CONV2D);
			output.writeInt(conv.getInputWidth());
			output.writeInt(conv.getInputHeight());
			output.writeInt(conv.getInputChannels());
			output.writeInt(conv.getFilters());
			output.writeInt(conv.getKernelSize());
			output.writeInt(conv.getStride());
			output.writeByte(conv.getPadding() == Padding.VALID ? 0 : 1);
			writeMatrix(output, conv.getKernels());
			writeMatrix(output, conv.getBiases());
			writeActivation(output, conv.getActivationFunction());
			return;
		}
		if (layer instanceof MaxPool2DLayer pool) {
			output.writeByte(LAYER_MAX_POOL_2D);
			writePool(output, pool.getInputWidth(), pool.getInputHeight(), pool.getChannels(), pool.getPoolSize(), pool.getStride());
			return;
		}
		if (layer instanceof AvgPool2DLayer pool) {
			output.writeByte(LAYER_AVG_POOL_2D);
			writePool(output, pool.getInputWidth(), pool.getInputHeight(), pool.getChannels(), pool.getPoolSize(), pool.getStride());
			return;
		}
		throw new IOException("Unsupported layer type: " + layer.getClass().getName());
	}

	private static Layer readLayer(DataInputStream input) throws IOException {
		switch (input.readUnsignedByte()) {
			case LAYER_DENSE:
				return readLegacyDenseLayer(input);
			case LAYER_CONV2D:
				return readConv2DLayer(input);
			case LAYER_MAX_POOL_2D: {
				int[] values = readPool(input);
				try {
					return new MaxPool2DLayer(values[0], values[1], values[2], values[3], values[4]);
				} catch (IllegalArgumentException exception) {
					throw new IOException("Invalid MaxPool2D layer", exception);
				}
			}
			case LAYER_AVG_POOL_2D: {
				int[] values = readPool(input);
				try {
					return new AvgPool2DLayer(values[0], values[1], values[2], values[3], values[4]);
				} catch (IllegalArgumentException exception) {
					throw new IOException("Invalid AvgPool2D layer", exception);
				}
			}
			default:
				throw new IOException("Unsupported layer type");
		}
	}

	private static DenseLayer readLegacyDenseLayer(DataInputStream input) throws IOException {
		Matrix weights = readMatrix(input);
		Matrix biases = readMatrix(input);
		if (weights.rows() != biases.rows() || biases.columns() != 1)
			throw new IOException("Invalid dense layer dimensions");
		return new DenseLayer(weights, biases, readActivation(input));
	}

	private static Conv2DLayer readConv2DLayer(DataInputStream input) throws IOException {
		int inputWidth = input.readInt();
		int inputHeight = input.readInt();
		int inputChannels = input.readInt();
		int filters = input.readInt();
		int kernelSize = input.readInt();
		int stride = input.readInt();
		int paddingId = input.readUnsignedByte();
		Padding padding;
		if (paddingId == 0)
			padding = Padding.VALID;
		else if (paddingId == 1)
			padding = Padding.SAME;
		else
			throw new IOException("Invalid convolution padding");
		Matrix kernels = readMatrix(input);
		Matrix biases = readMatrix(input);
		ActivationFunction activation = readActivation(input);
		try {
			Conv2DLayer layer = new Conv2DLayer(inputWidth, inputHeight, inputChannels, filters, kernelSize, stride, padding, activation);
			copyMatrix(kernels, layer.getKernels(), "convolution kernels");
			copyMatrix(biases, layer.getBiases(), "convolution biases");
			return layer;
		} catch (IllegalArgumentException exception) {
			throw new IOException("Invalid Conv2D layer", exception);
		}
	}

	private static void writePool(DataOutputStream output, int width, int height, int channels, int poolSize, int stride)
			throws IOException {
		output.writeInt(width);
		output.writeInt(height);
		output.writeInt(channels);
		output.writeInt(poolSize);
		output.writeInt(stride);
	}

	private static int[] readPool(DataInputStream input) throws IOException {
		return new int[] { input.readInt(), input.readInt(), input.readInt(), input.readInt(), input.readInt() };
	}

	private static void copyMatrix(Matrix source, Matrix target, String name) throws IOException {
		if (source.rows() != target.rows() || source.columns() != target.columns())
			throw new IOException("Invalid " + name + " dimensions");
		for (int row = 0; row < source.rows(); row++)
			System.arraycopy(source.values[row], 0, target.values[row], 0, source.columns());
	}

	private static void writeMatrix(DataOutputStream output, Matrix matrix) throws IOException {
		output.writeInt(matrix.rows());
		output.writeInt(matrix.columns());
		for (int row = 0; row < matrix.rows(); row++)
			for (int column = 0; column < matrix.columns(); column++)
				output.writeFloat(matrix.values[row][column]);
	}

	private static Matrix readMatrix(DataInputStream input) throws IOException {
		int rows = input.readInt();
		int columns = input.readInt();
		if (rows <= 0 || columns <= 0 || rows > MAX_MATRIX_DIMENSION || columns > MAX_MATRIX_DIMENSION
				|| (long) rows * columns > MAX_MATRIX_DIMENSION)
			throw new IOException("Invalid matrix dimensions");

		Matrix matrix = new Matrix(rows, columns);
		for (int row = 0; row < rows; row++)
			for (int column = 0; column < columns; column++)
				matrix.values[row][column] = input.readFloat();
		return matrix;
	}

	private static void writeActivation(DataOutputStream output, ActivationFunction activation) throws IOException {
		if (activation instanceof ReLU) {
			output.writeByte(1);
		} else if (activation instanceof Sigmoid) {
			output.writeByte(2);
		} else if (activation instanceof Tanh) {
			output.writeByte(3);
		} else if (activation instanceof SiLU) {
			output.writeByte(4);
		} else if (activation instanceof Swish) {
			output.writeByte(5);
		} else if (activation instanceof GELU) {
			output.writeByte(6);
		} else if (activation instanceof Softmax) {
			output.writeByte(7);
		} else if (activation instanceof ELU) {
			output.writeByte(8);
			output.writeFloat(((ELU) activation).getAlpha());
		} else if (activation instanceof LeakyReLU) {
			output.writeByte(9);
			output.writeFloat(((LeakyReLU) activation).getSlope());
		} else {
			throw new IOException("Unsupported activation function: " + activation.getClass().getName());
		}
	}

	private static ActivationFunction readActivation(DataInputStream input) throws IOException {
		switch (input.readByte()) {
			case 1: return new ReLU();
			case 2: return new Sigmoid();
			case 3: return new Tanh();
			case 4: return new SiLU();
			case 5: return new Swish();
			case 6: return new GELU();
			case 7: return new Softmax();
			case 8: return new ELU(input.readFloat());
			case 9: return new LeakyReLU(input.readFloat());
			default: throw new IOException("Unsupported activation function");
		}
	}
}
