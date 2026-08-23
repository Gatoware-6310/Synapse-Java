import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.layer.Conv2DLayer;
import xyz.gatoware.synapse.layer.Padding;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.SGD;

public class Conv2DLayerTest {
	@Test
	void forwardComputesConvolution() {
		Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1, 2, new ReLU());
		for (int i = 0; i < 4; i++)
			layer.getKernels().values[0][i] = 1.0f;
		layer.getBiases().values[0][0] = 0.0f;

		Matrix output = layer.forward(column(1, 2, 3, 4, 5, 6, 7, 8, 9));

		assertArrayEquals(new float[] { 12, 16, 24, 28 }, columnValues(output), 1e-6f);
	}

	@Test
	void backwardInputDoesNotUpdateKernels() {
		Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1, 2, new ReLU());
		for (int i = 0; i < 4; i++)
			layer.getKernels().values[0][i] = 1.0f;
		layer.forward(column(1, 2, 3, 4, 5, 6, 7, 8, 9));
		Matrix before = layer.getKernels().copy();

		Matrix inputGradient = layer.backwardInput(column(1, 1, 1, 1));

		assertArrayEquals(new float[] { 1, 2, 1, 2, 4, 2, 1, 2, 1 }, columnValues(inputGradient), 1e-6f);
		assertArrayEquals(before.values[0], layer.getKernels().values[0], 1e-6f);
	}

	@Test
	void backwardUpdatesConvolutionParameters() {
		Conv2DLayer layer = new Conv2DLayer(3, 3, 1, 1, 2, new ReLU());
		for (int i = 0; i < 4; i++)
			layer.getKernels().values[0][i] = 1.0f;
		layer.forward(column(1, 2, 3, 4, 5, 6, 7, 8, 9));

		layer.backward(column(1, 1, 1, 1), 0.01f, new SGD());

		assertArrayEquals(new float[] { 0.88f, 0.84f, 0.76f, 0.72f }, layer.getKernels().values[0], 1e-6f);
		assertEquals(-0.04f, layer.getBiases().values[0][0], 1e-6f);
	}

	@Test
	void samePaddingPreservesSpatialSizeAtStrideOne() {
		Conv2DLayer layer = new Conv2DLayer(5, 4, 3, 7, 3, Padding.SAME, new ReLU());
		Matrix output = layer.forward(new Matrix(5 * 4 * 3, 2));

		assertEquals(5, layer.getOutputWidth());
		assertEquals(4, layer.getOutputHeight());
		assertEquals(5 * 4 * 7, output.rows());
		assertEquals(2, output.columns());
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
