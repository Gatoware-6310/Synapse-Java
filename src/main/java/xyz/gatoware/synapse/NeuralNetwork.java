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
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.loss.LossFunction;
import xyz.gatoware.synapse.loss.SparseCategoricalCrossEntropy;
import xyz.gatoware.synapse.matrix.Matrix;

/** A lightweight neural network composed of layers. */
public class NeuralNetwork {
	private static final int FILE_MAGIC = 0x534E4E31; // SNN1
	private static final int FILE_VERSION = 1;
	private static final int MAX_LAYERS = 1_000_000;
	private static final int MAX_MATRIX_DIMENSION = 1_000_000;

	private List<Layer> layers = new ArrayList<>();
	private float lastLoss = Float.NaN;

	/** Creates an empty neural network. */
	public NeuralNetwork() {
		// here for compatibility
	}

	/** Creates a neural network from a list of layers.
	 * @param layerList the layers in the network
	 */
	public NeuralNetwork(Layer[] layerList) {
		for (Layer l : layerList) {
			addLayer(l);
		}
	}

	/** Adds a layer to the neural network.
	 * @param layer the layer to add
	 */
	public void addLayer(Layer layer) {
		layers.add(layer);
	}

	/** Runs the input through every layer and returns the output.
	 * @param input the network input
	 * @return the network output
	 */
	public Matrix forward(Matrix input) {
		Matrix output = input;
		for (Layer layer : layers)
			output = layer.forward(output);

		return output;
	}

	/** Trains the network on a dataset using the given loss function.
	 * @param dataset the dataset to train on
	 * @param lossFunction the loss function to use
	 * @param epochs the amount of training epochs
	 * @param learningRate the training learning rate
	 */
	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate) {
		fit(dataset, lossFunction, epochs, learningRate, false);
	}

	/** Trains the network on a dataset using the given loss function, optionally logging the average loss and classification accuracy after every epoch.
	 * @param dataset the dataset to train on
	 * @param lossFunction the loss function to use
	 * @param epochs the amount of training epochs
	 * @param learningRate the training learning rate
	 * @param logging whether to log after every epoch
	 */
	public void fit(final Dataset dataset, final LossFunction lossFunction, final int epochs, final float learningRate,
			final boolean logging) {
		if (dataset == null)
			throw new IllegalArgumentException("Dataset cannot be null");
		if (lossFunction == null)
			throw new IllegalArgumentException("Loss function cannot be null");
		if (dataset.size() == 0)
			throw new IllegalArgumentException("Dataset cannot be empty");
		if (layers.isEmpty())
			throw new IllegalStateException("Neural network must contain at least one layer");
		if (epochs <= 0)
			throw new IllegalArgumentException("Epochs must be positive");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");

		int[] order = new int[dataset.size()];
		for (int i = 0; i < order.length; i++)
			order[i] = i;
		Random random = new Random();

		for (int epoch = 0; epoch < epochs; epoch++) {
			shuffle(order, random);
			float totalLoss = 0.0f;

			for (int index : order) {
				Matrix output = forward(dataset.getInput(index));
				Matrix predicted = rowVector(output);
				Matrix actual = rowVector(dataset.getTarget(index));
				totalLoss += lossFunction.calculate(predicted, actual);

				Matrix gradient = columnVector(lossFunction.gradient(predicted, actual));
				for (int layer = layers.size() - 1; layer >= 0; layer--)
					gradient = layers.get(layer).backward(gradient, learningRate);
			}

			lastLoss = totalLoss / dataset.size();

			if (logging) {
				System.out.printf("Epoch %d loss: %.6f accuracy: %.2f%%%n", epoch + 1, lastLoss,
					accuracy(dataset) * 100.0f);
			}
		}
	}

	/** Trains the network using SparseCategoricalCrossEntropy by default.
	 * @param dataset the dataset to train on
	 * @param epochs the amount of training epochs
	 * @param learningRate the training learning rate
	 */
	public void fit(final Dataset dataset, final int epochs, final float learningRate) {
		fit(dataset, epochs, learningRate, false);
	}

	/** Trains the network using SparseCategoricalCrossEntropy by default, optionally logging after every epoch.
	 * @param dataset the dataset to train on
	 * @param epochs the amount of training epochs
	 * @param learningRate the training learning rate
	 * @param logging whether to log after every epoch
	 */
	public void fit(final Dataset dataset, final int epochs, final float learningRate, final boolean logging) {
		fit(dataset, new SparseCategoricalCrossEntropy(), epochs, learningRate, logging);
	}

	/** Returns the most recent average loss after training.
	 * @return the most recent average loss
	 */
	public float getLastLoss() {
		return lastLoss;
	}

	/** Returns the index of the largest value in the network output.
	 * @param input the network input
	 * @return the index of the largest output value
	 */
	public int predict(Matrix input) {
		Matrix output = forward(input);
		if (output.rows() == 0 || output.columns() == 0 || (output.rows() != 1 && output.columns() != 1)) {
			throw new IllegalStateException("Network output must be a non-empty vector");
		}

		int prediction = 0;
		float highest = output.rows() == 1 ? output.values[0][0] : output.values[0][0];
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

	/** Returns the fraction of dataset samples classified correctly.
	 * @param dataset the dataset to evaluate
	 * @return the fraction classified correctly
	 */
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

	/** Saves the model to a file.
	 * @param filename the file to save to
	 * @throws IOException if the model cannot be saved
	 */
	public void save(String filename) throws IOException {
		save(Path.of(filename));
	}

	/** Saves the model to a path.
	 * @param path the path to save to
	 * @throws IOException if the model cannot be saved
	 */
	public void save(Path path) throws IOException {
		try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
			output.writeInt(FILE_MAGIC);
			output.writeInt(FILE_VERSION);
			output.writeInt(layers.size());

			for (Layer layer : layers) {
				if (!(layer instanceof DenseLayer)) {
					throw new IOException("Only DenseLayer instances can be saved");
				}

				DenseLayer denseLayer = (DenseLayer) layer;
				writeMatrix(output, denseLayer.getWeights());
				writeMatrix(output, denseLayer.getBiases());
				writeActivation(output, denseLayer.getActivationFunction());
			}
		}
	}

	/** Loads a model from a file.
	 * @param filename the file to load
	 * @return the loaded neural network
	 * @throws IOException if the model cannot be loaded
	 */
	public static NeuralNetwork load(String filename) throws IOException {
		return load(Path.of(filename));
	}

	/** Loads a model from a path.
	 * @param path the path to load
	 * @return the loaded neural network
	 * @throws IOException if the model cannot be loaded
	 */
	public static NeuralNetwork load(Path path) throws IOException {
		try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
			if (input.readInt() != FILE_MAGIC) {
				throw new IOException("Not a Synapse neural network file");
			}
			if (input.readInt() != FILE_VERSION) {
				throw new IOException("Unsupported Synapse neural network file version");
			}

			int layerCount = input.readInt();
			if (layerCount < 0 || layerCount > MAX_LAYERS) {
				throw new IOException("Invalid layer count");
			}

			NeuralNetwork network = new NeuralNetwork();
			for (int i = 0; i < layerCount; i++) {
				Matrix weights = readMatrix(input);
				Matrix biases = readMatrix(input);
				if (weights.rows() != biases.rows() || biases.columns() != 1) {
					throw new IOException("Invalid dense layer dimensions");
				}
				network.addLayer(new DenseLayer(weights, biases, readActivation(input)));
			}

			return network;
		}
	}

	private static void writeMatrix(DataOutputStream output, Matrix matrix) throws IOException {
		output.writeInt(matrix.rows());
		output.writeInt(matrix.columns());
		for (int row = 0; row < matrix.rows(); row++) {
			for (int column = 0; column < matrix.columns(); column++) {
				output.writeFloat(matrix.values[row][column]);
			}
		}
	}

	private static Matrix readMatrix(DataInputStream input) throws IOException {
		int rows = input.readInt();
		int columns = input.readInt();
		if (rows <= 0 || columns <= 0 || rows > MAX_MATRIX_DIMENSION || columns > MAX_MATRIX_DIMENSION
				|| (long) rows * columns > MAX_MATRIX_DIMENSION) {
			throw new IOException("Invalid matrix dimensions");
		}

		Matrix matrix = new Matrix(rows, columns);
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				matrix.values[row][column] = input.readFloat();
			}
		}
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
			case 1:
				return new ReLU();
			case 2:
				return new Sigmoid();
			case 3:
				return new Tanh();
			case 4:
				return new SiLU();
			case 5:
				return new Swish();
			case 6:
				return new GELU();
			case 7:
				return new Softmax();
			case 8:
				return new ELU(input.readFloat());
			case 9:
				return new LeakyReLU(input.readFloat());
			default:
				throw new IOException("Unsupported activation function");
		}
	}
}
