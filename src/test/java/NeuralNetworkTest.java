import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.Sigmoid;
import xyz.gatoware.synapse.layer.DenseLayer;
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
