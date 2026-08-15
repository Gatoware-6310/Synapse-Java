import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.reflect.Field;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.matrix.Matrix;

public class DenseLayerTest {

	@Test
	void constructorCreatesParametersWithCorrectDimensions() {
		DenseLayer layer = new DenseLayer(3, 2, new ReLU());

		assertEquals(2, weights(layer).rows());
		assertEquals(3, weights(layer).columns());
		assertEquals(2, biases(layer).rows());
		assertEquals(1, biases(layer).columns());
	}

	@Test
	void forwardReturnsOutputWithCorrectDimensionsWithoutChangingWeights() {
		DenseLayer layer = new DenseLayer(3, 2, new ReLU());
		Matrix weightsBefore = weights(layer).copy();

		Matrix output = layer.forward(new Matrix(new float[][] {
				{ 1.0f },
				{ 2.0f },
				{ 3.0f }
		}));

		assertEquals(2, output.rows());
		assertEquals(1, output.columns());
		assertMatrixEquals(weightsBefore, weights(layer));
	}

	@Test
	void forwardCalculatesWeightedInputBiasAndActivation() {
		DenseLayer layer = new DenseLayer(2, 2, new ReLU());
		weights(layer).values = new float[][] {
				{ 2.0f, -1.0f },
				{ -3.0f, 4.0f }
		};
		biases(layer).values = new float[][] {
				{ 1.0f },
				{ -2.0f }
		};

		Matrix output = layer.forward(new Matrix(new float[][] {
				{ 3.0f },
				{ 2.0f }
		}));

		assertArrayEquals(new float[] { 5.0f }, output.values[0]);
		assertArrayEquals(new float[] { 0.0f }, output.values[1]);
	}

	@Test
	void forwardRejectsInputThatIsNotAnInputSizeByOneColumnMatrix() {
		DenseLayer layer = new DenseLayer(3, 2, new ReLU());

		IllegalArgumentException wrongRows = assertThrows(
				IllegalArgumentException.class,
				() -> layer.forward(new Matrix(2, 1)));
		IllegalArgumentException wrongColumns = assertThrows(
				IllegalArgumentException.class,
				() -> layer.forward(new Matrix(3, 2)));

		assertEquals("Dense layer input must have dimensions 3 x 1", wrongRows.getMessage());
		assertEquals("Dense layer input must have dimensions 3 x 1", wrongColumns.getMessage());
	}

	@Test
	void backwardUpdatesWeightsAndReturnsInputGradient() {
		DenseLayer layer = new DenseLayer(
				new Matrix(new float[][] { { 2.0f, -1.0f } }),
				new Matrix(new float[][] { { 0.0f } }),
				new ReLU());
		layer.forward(new Matrix(new float[][] { { 3.0f }, { 4.0f } }));

		Matrix inputGradient = layer.backward(new Matrix(new float[][] { { 0.5f } }), 0.1f);

		assertArrayEquals(new float[] { 1.0f }, inputGradient.values[0]);
		assertArrayEquals(new float[] { -0.5f }, inputGradient.values[1]);
		assertArrayEquals(new float[] { 1.85f, -1.2f }, layer.getWeights().values[0], 0.0001f);
		assertEquals(-0.05f, layer.getBiases().values[0][0], 0.0001f);
	}

	@Test
	void denseLayerHonorsExistingVectorActivationOverrides() {
		ActivationFunction vectorActivation = new ActivationFunction() {
			@Override
			public float apply(float value) {
				return -1.0f;
			}

			@Override
			public float[] apply(float[] values) {
				float sum = values[0] + values[1];
				return new float[] { values[0] / sum, values[1] / sum };
			}

			@Override
			public float[] backward(float[] inputs, float[] outputs, float[] gradients) {
				return new float[] { gradients[1], gradients[0] };
			}
		};
		DenseLayer layer = new DenseLayer(
				new Matrix(new float[][] { { 1.0f, 0.0f }, { 0.0f, 1.0f } }),
				new Matrix(new float[][] { { 0.0f }, { 0.0f } }),
				vectorActivation);

		Matrix output = layer.forward(new Matrix(new float[][] { { 1.0f }, { 3.0f } }));
		Matrix inputGradient = layer.backward(new Matrix(new float[][] { { 2.0f }, { 5.0f } }), 0.1f);

		assertArrayEquals(new float[] { 0.25f }, output.values[0]);
		assertArrayEquals(new float[] { 0.75f }, output.values[1]);
		assertArrayEquals(new float[] { 5.0f }, inputGradient.values[0]);
		assertArrayEquals(new float[] { 2.0f }, inputGradient.values[1]);
	}

	private Matrix weights(DenseLayer layer) {
		return parameter(layer, "weights");
	}

	private Matrix biases(DenseLayer layer) {
		return parameter(layer, "biases");
	}

	private Matrix parameter(DenseLayer layer, String name) {
		try {
			Field field = DenseLayer.class.getDeclaredField(name);
			field.setAccessible(true);
			return (Matrix) field.get(layer);
		} catch (ReflectiveOperationException exception) {
			throw new AssertionError(exception);
		}
	}

	private void assertMatrixEquals(Matrix expected, Matrix actual) {
		assertEquals(expected.rows(), actual.rows());
		assertEquals(expected.columns(), actual.columns());
		for (int i = 0; i < expected.rows(); i++)
			assertArrayEquals(expected.values[i], actual.values[i]);
	}
}
