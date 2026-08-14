import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.loss.BinaryCrossEntropy;
import xyz.gatoware.synapse.loss.CategoricalCrossEntropy;
import xyz.gatoware.synapse.loss.HingeLoss;
import xyz.gatoware.synapse.loss.HuberLoss;
import xyz.gatoware.synapse.loss.KLDivergence;
import xyz.gatoware.synapse.loss.LossFunction;
import xyz.gatoware.synapse.loss.MeanAbsoluteError;
import xyz.gatoware.synapse.loss.MeanSquaredError;
import xyz.gatoware.synapse.loss.SparseCategoricalCrossEntropy;
import xyz.gatoware.synapse.matrix.Matrix;

public class LossFunctionTest {
	private static final float EPSILON = 0.0001f;

	@Test
	void regressionLossesCalculateMeanError() {
		Matrix predicted = new Matrix(new float[][] { { 2.0f, 4.0f } });
		Matrix actual = new Matrix(new float[][] { { 1.0f, 1.0f } });

		assertEquals(5.0f, new MeanSquaredError().calculate(predicted, actual), EPSILON);
		assertEquals(2.0f, new MeanAbsoluteError().calculate(predicted, actual), EPSILON);
	}

	@Test
	void binaryCrossEntropyCalculatesMeanProbabilityError() {
		Matrix predicted = new Matrix(new float[][] { { 0.9f, 0.2f } });
		Matrix actual = new Matrix(new float[][] { { 1.0f, 0.0f } });
		float expected = -((float) Math.log(0.9f) + (float) Math.log(0.8f)) / 2.0f;

		assertEquals(expected, new BinaryCrossEntropy().calculate(predicted, actual), EPSILON);
	}

	@Test
	void categoricalCrossEntropyCalculatesMeanBatchLoss() {
		Matrix predicted = new Matrix(new float[][] {
				{ 0.1f, 0.7f, 0.2f },
				{ 0.8f, 0.1f, 0.1f }
		});
		Matrix actual = new Matrix(new float[][] {
				{ 0.0f, 1.0f, 0.0f },
				{ 1.0f, 0.0f, 0.0f }
		});
		float expected = -((float) Math.log(0.7f) + (float) Math.log(0.8f)) / 2.0f;

		assertEquals(expected, new CategoricalCrossEntropy().calculate(predicted, actual), EPSILON);
	}

	@Test
	void sparseCategoricalCrossEntropyUsesClassIds() {
		Matrix predicted = new Matrix(new float[][] {
				{ 0.1f, 0.7f, 0.2f },
				{ 0.8f, 0.1f, 0.1f }
		});
		Matrix actual = new Matrix(new float[][] { { 1.0f }, { 0.0f } });
		float expected = -((float) Math.log(0.7f) + (float) Math.log(0.8f)) / 2.0f;

		assertEquals(expected, new SparseCategoricalCrossEntropy().calculate(predicted, actual), EPSILON);
	}

	@Test
	void sparseCategoricalCrossEntropyGradientTargetsClassProbability() {
		Matrix gradient = new SparseCategoricalCrossEntropy().gradient(
				new Matrix(new float[][] { { 0.1f, 0.7f, 0.2f } }),
				new Matrix(new float[][] { { 1.0f } }));

		assertArrayEquals(new float[] { 0.0f, -1.0f / 0.7f, 0.0f }, gradient.values[0], EPSILON);
	}

	@Test
	void hingeAndHuberLossCalculateExpectedValues() {
		assertEquals(0.25f, new HingeLoss().calculate(
				new Matrix(new float[][] { { 2.0f, -0.5f } }),
				new Matrix(new float[][] { { 1.0f, -1.0f } })), EPSILON);
		assertEquals(1.3125f, new HuberLoss().calculate(
				new Matrix(new float[][] { { 0.5f, 3.0f } }),
				new Matrix(new float[][] { { 0.0f, 0.0f } })), EPSILON);
	}

	@Test
	void klDivergenceCalculatesDistributionDifference() {
		Matrix predicted = new Matrix(new float[][] { { 0.5f, 0.5f } });
		Matrix actual = new Matrix(new float[][] { { 0.75f, 0.25f } });
		float expected = 0.75f * (float) Math.log(1.5f) + 0.25f * (float) Math.log(0.5f);

		assertEquals(expected, new KLDivergence().calculate(predicted, actual), EPSILON);
	}

	@Test
	void lossesRejectMismatchedMatrixDimensions() {
		Matrix predicted = new Matrix(new float[][] { { 0.5f, 0.5f } });
		Matrix actual = new Matrix(new float[][] { { 1.0f }, { 0.0f } });

		LossFunction[] losses = {
				new MeanSquaredError(), new MeanAbsoluteError(), new BinaryCrossEntropy(),
				new CategoricalCrossEntropy(), new HingeLoss(), new HuberLoss(), new KLDivergence()
		};
		for (LossFunction loss : losses) {
			assertThrows(IllegalArgumentException.class, () -> loss.calculate(predicted, actual));
		}
	}

	@Test
	void sparseCategoricalCrossEntropyRejectsInvalidClassIds() {
		Matrix predicted = new Matrix(new float[][] { { 0.1f, 0.9f } });
		Matrix actual = new Matrix(new float[][] { { 2.0f } });

		assertThrows(IllegalArgumentException.class,
				() -> new SparseCategoricalCrossEntropy().calculate(predicted, actual));
	}
}
