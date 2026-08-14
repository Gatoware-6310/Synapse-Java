package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

/** The hinge loss function. */
public class HingeLoss implements LossFunction {
	/** Creates a hinge loss. */
	public HingeLoss() {
	}

	/** Calculates hinge loss. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				loss += Math.max(0.0f, 1.0f - actual.values[i][j] * predicted.values[i][j]);
			}
		}

		return loss / (predicted.rows() * predicted.columns());
	}

	/** Calculates the hinge-loss gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float scale = 1.0f / (predicted.rows() * predicted.columns());
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				if (1.0f - actual.values[i][j] * predicted.values[i][j] > 0.0f)
					gradient.values[i][j] = -actual.values[i][j] * scale;
			}
		}
		return gradient;
	}

	private void validateDimensions(Matrix predicted, Matrix actual) {
		if (predicted.rows() != actual.rows() || predicted.columns() != actual.columns()) {
			throw new IllegalArgumentException(
				"Predicted and actual matrices must have the same dimensions"
			);
		}
	}
}
