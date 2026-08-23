import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.layer.Conv2DLayer;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.layer.MaxPool2DLayer;
import xyz.gatoware.synapse.matrix.Matrix;

public class CnnNetworkTest {
	@Test
	void forwardToAndInputGradientSupportFeatureVisualization() {
		Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1, 2, new ReLU());
		for (int i = 0; i < 4; i++)
			layer.getKernels().values[0][i] = 1.0f;
		NeuralNetwork network = new NeuralNetwork(new Layer[] { layer });
		Matrix input = column(1, 2, 3, 4, 5, 6, 7, 8, 9);
		Matrix kernelsBefore = layer.getKernels().copy();

		Matrix activation = network.forwardTo(input, 0);
		Matrix gradient = network.inputGradient(input, 0, column(1, 1, 1, 1));

		assertArrayEquals(new float[] { 12, 16, 24, 28 }, columnValues(activation), 1e-6f);
		assertArrayEquals(new float[] { 1, 2, 1, 2, 4, 2, 1, 2, 1 }, columnValues(gradient), 1e-6f);
		assertArrayEquals(kernelsBefore.values[0], layer.getKernels().values[0], 1e-6f);
	}

	@Test
	void saveAndLoadPreserveCnnLayers() throws Exception {
		Conv2DLayer convolution = new Conv2DLayer(4, 4, 1, 1, 2, new ReLU());
		for (int i = 0; i < 4; i++)
			convolution.getKernels().values[0][i] = 0.5f;
		DenseLayer dense = new DenseLayer(new Matrix(new float[][] { { 2.0f } }),
			new Matrix(new float[][] { { 0.25f } }), new ReLU());
		NeuralNetwork network = new NeuralNetwork(new Layer[] {
			convolution,
			new MaxPool2DLayer(3, 3, 1, 3),
			dense
		});
		Matrix input = column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
		Matrix expected = network.forward(input);
		Path file = Files.createTempFile("synapse-cnn", ".snn");
		try {
			network.save(file);
			Matrix actual = NeuralNetwork.load(file).forward(input);
			assertArrayEquals(expected.values[0], actual.values[0], 1e-6f);
		} finally {
			Files.deleteIfExists(file);
		}
	}

	@Test
	void loadStillAcceptsVersionOneDenseModels() throws Exception {
		Path file = Files.createTempFile("synapse-v1", ".snn");
		try {
			try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
				output.writeInt(0x534E4E31);
				output.writeInt(1);
				output.writeInt(1);
				writeSingleValueMatrix(output, 2.0f);
				writeSingleValueMatrix(output, 1.0f);
				output.writeByte(1); // ReLU
			}
			Matrix result = NeuralNetwork.load(file).forward(column(3.0f));
			assertEquals(7.0f, result.values[0][0], 1e-6f);
		} finally {
			Files.deleteIfExists(file);
		}
	}

	private static void writeSingleValueMatrix(DataOutputStream output, float value) throws Exception {
		output.writeInt(1);
		output.writeInt(1);
		output.writeFloat(value);
	}

	private static Matrix column(float... values) {
		Matrix matrix = new Matrix(values.length, 1);
		for (int i = 0; i < values.length; i++)
			matrix.values[i][0] = values[i];
		return matrix;
	}

	private static float[] columnValues(Matrix matrix) {
		float[] result = new float[matrix.rows()];
		for (int i = 0; i < result.length; i++)
			result[i] = matrix.values[i][0];
		return result;
	}
}
