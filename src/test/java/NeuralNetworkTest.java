import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.Sigmoid;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.matrix.Matrix;

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
}
