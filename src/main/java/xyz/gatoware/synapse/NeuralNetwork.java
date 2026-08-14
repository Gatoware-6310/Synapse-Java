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
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.matrix.Matrix;

public class NeuralNetwork {
	private static final int FILE_MAGIC = 0x534E4E31; // SNN1
	private static final int FILE_VERSION = 1;
	private static final int MAX_LAYERS = 1_000_000;
	private static final int MAX_MATRIX_DIMENSION = 1_000_000;

	private List<Layer> layers = new ArrayList<>();

	public NeuralNetwork() {
		// here for compatibility
	}

	public NeuralNetwork(Layer[] layerList) {
		for (Layer l : layerList) {
			addLayer(l);
		}
	}

	public void addLayer(Layer layer) {
		layers.add(layer);
	}

	public Matrix forward(Matrix input) {
		Matrix output = input;
		for (Layer layer : layers)
			output = layer.forward(output);

		return output;
	}

	public void save(String filename) throws IOException {
		save(Path.of(filename));
	}

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

	public static NeuralNetwork load(String filename) throws IOException {
		return load(Path.of(filename));
	}

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
