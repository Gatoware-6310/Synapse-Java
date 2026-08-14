package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

/** The mean squared error loss function. */
public class MeanSquaredError implements LossFunction {
	/** Creates a mean squared error loss. */
	public MeanSquaredError() {
	}

	/** Calculates mean squared error. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float difference = predicted.values[i][j] - actual.values[i][j];
				loss += difference * difference;
			}
		}

		return loss / (predicted.rows() * predicted.columns());
	}

	/** Calculates the mean squared error gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float scale = 2.0f / (predicted.rows() * predicted.columns());
		for (int i = 0; i < predicted.rows(); i++)
			for (int j = 0; j < predicted.columns(); j++)
				gradient.values[i][j] = scale * (predicted.values[i][j] - actual.values[i][j]);
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
