import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.layer.AvgPool2DLayer;
import xyz.gatoware.synapse.layer.MaxPool2DLayer;
import xyz.gatoware.synapse.matrix.Matrix;

public class PoolingLayerTest {
	@Test
	void maxPoolForwardAndBackward() {
		MaxPool2DLayer layer = new MaxPool2DLayer(4, 4, 1, 2);
		Matrix output = layer.forward(column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
		assertArrayEquals(new float[] { 6, 8, 14, 16 }, columnValues(output), 1e-6f);

		Matrix gradient = layer.backwardInput(column(1, 1, 1, 1));
		assertArrayEquals(new float[] { 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 },
			columnValues(gradient), 1e-6f);
	}

	@Test
	void averagePoolForwardAndBackward() {
		AvgPool2DLayer layer = new AvgPool2DLayer(4, 4, 1, 2);
		Matrix output = layer.forward(column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
		assertArrayEquals(new float[] { 3.5f, 5.5f, 11.5f, 13.5f }, columnValues(output), 1e-6f);

		Matrix gradient = layer.backwardInput(column(1, 1, 1, 1));
		float[] expected = new float[16];
		java.util.Arrays.fill(expected, 0.25f);
		assertArrayEquals(expected, columnValues(gradient), 1e-6f);
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
