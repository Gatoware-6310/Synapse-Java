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
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.loss.LossFunction;
import xyz.gatoware.synapse.loss.SparseCategoricalCrossEntropy;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Adam;
import xyz.gatoware.synapse.optimizer.Optimizer;

/** A lightweight neural network composed of layers. */
public class NeuralNetwork {
	private static final int FILE_MAGIC = 0x534E4E31; // SNN1
	private static final int FILE_VERSION = 1;
	private static final int MAX_LAYERS = 1_000_000;
	private static final int MAX_MATRIX_DIMENSION = 1_000_000;
	private static final int CUDA_BATCH_SIZE = 32;

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
			final Optimizer optimizer) {
		fit(dataset, lossFunction, epochs, learningRate, optimizer, false);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final boolean logging) {
		fit(dataset, lossFunction, epochs, learningRate, new Adam(), logging);
	}

	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final Optimizer optimizer, final boolean logging) {
		validateTrainingArguments(dataset, lossFunction, epochs, learningRate, optimizer);
		if (canUseCudaTraining()) {
			fitCuda(dataset, lossFunction, epochs, learningRate, optimizer, logging);
			return;
		}
		fitCpu(dataset, lossFunction, epochs, learningRate, optimizer, logging);
	}

	private void fitCpu(Dataset dataset, LossFunction lossFunction, int epochs, float learningRate,
			Optimizer optimizer, boolean logging) {
		int[] order = makeOrder(dataset.size());
		Random random = new Random();
		for (int epoch = 0; epoch < epochs; epoch++) {
			shuffle(order, random);
			float totalLoss = 0.0f;
			for (int index : order) {
				Matrix output = forwardTraining(dataset.getInput(index));
				Matrix predicted = rowVector(output);
				Matrix actual = rowVector(dataset.getTarget(index));
				totalLoss += lossFunction.calculate(predicted, actual);
				Matrix gradient = columnVector(lossFunction.gradient(predicted, actual));
				for (int layer = layers.size() - 1; layer >= 0; layer--)
					gradient = layers.get(layer).backward(gradient, learningRate, optimizer);
			}
			lastLoss = totalLoss / dataset.size();
			if (logging)
				System.out.printf("Epoch %d loss: %.6f accuracy: %.2f%%%n", epoch + 1, lastLoss,
					accuracy(dataset) * 100.0f);
		}
	}

	/** CUDA training uses mini-batches and keeps all hidden dense/ReLU activations,
	 * backprop GEMMs, gradients and optimizer state on the GPU. The final
	 * activation/loss boundary is materialized so Softmax and arbitrary LossFunction
	 * implementations keep their existing Java API.
	 */
	private void fitCuda(Dataset dataset, LossFunction lossFunction, int epochs, float learningRate,
			Optimizer optimizer, boolean logging) {
		int[] order = makeOrder(dataset.size());
		Random random = new Random();
		int featureCount = dataset.getInputs().columns();

		try {
			for (int epoch = 0; epoch < epochs; epoch++) {
				shuffle(order, random);
				float totalLoss = 0.0f;

				for (int start = 0; start < order.length; start += CUDA_BATCH_SIZE) {
					int batchSize = Math.min(CUDA_BATCH_SIZE, order.length - start);
					Matrix batchInput = new Matrix(featureCount, batchSize);
					for (int sample = 0; sample < batchSize; sample++) {
						int index = order[start + sample];
						for (int feature = 0; feature < featureCount; feature++)
							batchInput.values[feature][sample] = dataset.getInputs().values[index][feature];
					}

					Matrix output = batchInput;
					for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
						DenseLayer dense = (DenseLayer) layers.get(layerIndex);
						if (layerIndex < layers.size() - 1)
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
					for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--)
						gradient = layers.get(layerIndex).backward(gradient, learningRate, optimizer);
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
		for (int i = 0; i < size; i++) order[i] = i;
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

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final Optimizer optimizer) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, false);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final boolean logging) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, new Adam(), logging);
	}

	public void fit(final Dataset dataset, final int epochs, final float learningRate, final Optimizer optimizer,
			final boolean logging) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, optimizer, logging);
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
			for (Layer layer : layers) {
				if (!(layer instanceof DenseLayer))
					throw new IOException("Only DenseLayer instances can be saved");
				DenseLayer denseLayer = (DenseLayer) layer;
				writeMatrix(output, denseLayer.getWeights());
				writeMatrix(output, denseLayer.getBiases());
				writeActivation(output, denseLayer.getActivationFunction());
			}
		}
	}

	public static NeuralNetwork load(String filename) throws IOException {
		return load(Path.of(filename));
	}

	public static NeuralNetwork load(Path path) throws IOException {
		try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
			if (input.readInt() != FILE_MAGIC)
				throw new IOException("Not a Synapse neural network file");
			if (input.readInt() != FILE_VERSION)
				throw new IOException("Unsupported Synapse neural network file version");

			int layerCount = input.readInt();
			if (layerCount < 0 || layerCount > MAX_LAYERS)
				throw new IOException("Invalid layer count");

			NeuralNetwork network = new NeuralNetwork();
			for (int i = 0; i < layerCount; i++) {
				Matrix weights = readMatrix(input);
				Matrix biases = readMatrix(input);
				if (weights.rows() != biases.rows() || biases.columns() != 1)
					throw new IOException("Invalid dense layer dimensions");
				network.addLayer(new DenseLayer(weights, biases, readActivation(input)));
			}
			return network;
		}
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
