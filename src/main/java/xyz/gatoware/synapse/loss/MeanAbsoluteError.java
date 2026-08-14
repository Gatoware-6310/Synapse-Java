package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class MeanAbsoluteError implements LossFunction {
	/** Calculates mean absolute error. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				loss += Math.abs(predicted.values[i][j] - actual.values[i][j]);
			}
		}

		return loss / (predicted.rows() * predicted.columns());
	}

	/** Calculates the mean absolute error gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float scale = 1.0f / (predicted.rows() * predicted.columns());
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float difference = predicted.values[i][j] - actual.values[i][j];
				gradient.values[i][j] = Math.signum(difference) * scale;
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
