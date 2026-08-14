import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.Sigmoid;
import xyz.gatoware.synapse.activation.Softmax;
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.loss.SparseCategoricalCrossEntropy;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.layer.*;

public class NeuralNetworkTest {

	@Test
	void forwardPassesTheOutputThroughLayersInInsertionOrder() {
		NeuralNetwork network = new NeuralNetwork();
		network.addLayer(input -> new Matrix(new float[][] {
				{ input.values[0][0] + 2.0f }
		}));
		network.addLayer(input -> new Matrix(new float[][] {
				{ input.values[0][0] * 3.0f }
		}));

		Matrix output = network.forward(new Matrix(new float[][] { { 1.0f } }));

		assertArrayEquals(new float[] { 9.0f }, output.values[0]);
	}

	@Test
	void testAutomaticLayerConstruction() {
		NeuralNetwork network = new NeuralNetwork(784, 128, 3, 10);

		Matrix output = network.forward(new Matrix(784, 1));

		assertEquals(10, output.rows());
		assertEquals(1, output.columns());
	}

	@Test
	void testZeroHiddenLayers() {
		NeuralNetwork network = new NeuralNetwork(784, 128, 0, 10);

		Matrix output = network.forward(new Matrix(784, 1));

		assertEquals(10, output.rows());
		assertEquals(1, output.columns());
	}

	@Test
	void testInvalidArguments() {
		assertThrows(IllegalArgumentException.class,
				() -> new NeuralNetwork(0, 128, 2, 10));

		assertThrows(IllegalArgumentException.class,
				() -> new NeuralNetwork(784, 0, 2, 10));

		assertThrows(IllegalArgumentException.class,
				() -> new NeuralNetwork(784, 128, -1, 10));

		assertThrows(IllegalArgumentException.class,
				() -> new NeuralNetwork(784, 128, 2, 0));
	}

	@Test
	void createNetworkWithDenseLayerList() {
		@SuppressWarnings("unused")
		NeuralNetwork network = new NeuralNetwork(new Layer[] {
				new DenseLayer(2, 5, new ReLU()),
				new DenseLayer(2, 5, new ReLU()),
				new DenseLayer(2, 5, new ReLU()),
				new DenseLayer(2, 5, new ReLU()),
		});
	}

	@Test
	void forwardReturnsTheFinalDenseLayerOutputDimensions() {
		NeuralNetwork network = new NeuralNetwork();
		network.addLayer(new DenseLayer(2, 3, new ReLU()));
		network.addLayer(new DenseLayer(3, 1, new Sigmoid()));

		Matrix output = network.forward(new Matrix(new float[][] {
				{ 0.25f },
				{ 0.75f }
		}));

		assertEquals(1, output.rows());
		assertEquals(1, output.columns());
	}

	@Test
	void predictReturnsIndexOfLargestOutput() {
		NeuralNetwork network = new NeuralNetwork(new Layer[] {
				new DenseLayer(new Matrix(new float[][] {
						{ 1.0f },
						{ 3.0f },
						{ 2.0f }
				}), new Matrix(
						new float[][] {
								{ 0.0f },
								{ 0.0f },
								{ 0.0f }
						}), new ReLU())
		});

		assertEquals(1, network.predict(new Matrix(new float[][] { { 1.0f } })));
	}

	@Test
	void fitLearnsIntegerClassLabels() {
		Dataset dataset = new Dataset(
				new Matrix(new float[][] {
						{ 1.0f, 0.0f },
						{ 0.0f, 1.0f }
				}),
				new Matrix(new float[][] {
						{ 0.0f },
						{ 1.0f }
				}));
		DenseLayer hidden = new DenseLayer(
				new Matrix(new float[][] {
						{ 1.0f, 0.0f },
						{ 0.0f, 1.0f }
				}),
				new Matrix(new float[][] {
						{ 0.1f },
						{ 0.1f }
				}),
				new ReLU());
		DenseLayer output = new DenseLayer(
				new Matrix(new float[][] {
						{ 0.0f, 0.0f },
						{ 0.0f, 0.0f }
				}),
				new Matrix(new float[][] {
						{ 0.0f },
						{ 0.0f }
				}),
				new Softmax());
		NeuralNetwork network = new NeuralNetwork(new Layer[] { hidden, output });
		Matrix hiddenWeightsBefore = hidden.getWeights().copy();

		network.fit(dataset, new SparseCategoricalCrossEntropy(), 100, 0.1f);

		assertEquals(0, network.predict(dataset.getInput(0)));
		assertEquals(1, network.predict(dataset.getInput(1)));
		assertTrue(network.getLastLoss() < 0.1f);
		assertTrue(hidden.getWeights().values[0][0] != hiddenWeightsBefore.values[0][0]);
	}

	@Test
	void fitLogsLossWhenLoggingIsEnabled() {
		Dataset dataset = new Dataset(
				new Matrix(new float[][] {
						{ 1.0f, 0.0f },
						{ 0.0f, 1.0f }
				}),
				new Matrix(new float[][] {
						{ 0.0f },
						{ 1.0f }
				}));
		NeuralNetwork network = new NeuralNetwork(new Layer[] {
				new DenseLayer(2, 2, new Softmax())
		});
		ByteArrayOutputStream output = new ByteArrayOutputStream();
		PrintStream originalOutput = System.out;

		try {
			System.setOut(new PrintStream(output));
			network.fit(dataset, 1, 0.1f, true);
		} finally {
			System.setOut(originalOutput);
		}

		String log = output.toString();
		assertTrue(log.contains("loss:"));
		assertTrue(log.contains("accuracy:"));
	}

	@Test
	void saveAndLoadPreserveDenseLayers() throws IOException {
		NeuralNetwork network = new NeuralNetwork();
		DenseLayer firstLayer = new DenseLayer(2, 2, new ReLU());
		firstLayer.getWeights().values[0] = new float[] { 1.0f, -2.0f };
		firstLayer.getWeights().values[1] = new float[] { 0.5f, 3.0f };
		firstLayer.getBiases().values[0][0] = 0.25f;
		firstLayer.getBiases().values[1][0] = -0.5f;
		network.addLayer(firstLayer);

		DenseLayer secondLayer = new DenseLayer(2, 1, new Sigmoid());
		secondLayer.getWeights().values[0] = new float[] { 1.5f, -0.75f };
		secondLayer.getBiases().values[0][0] = 0.1f;
		network.addLayer(secondLayer);

		Matrix input = new Matrix(new float[][] { { 0.4f }, { -0.2f } });
		Matrix expected = network.forward(input);
		Path file = Files.createTempFile("synapse-network", ".snn");

		try {
			network.save(file);
			Matrix actual = NeuralNetwork.load(file).forward(input);
			assertArrayEquals(expected.values[0], actual.values[0]);
		} finally {
			Files.deleteIfExists(file);
		}
	}
}
