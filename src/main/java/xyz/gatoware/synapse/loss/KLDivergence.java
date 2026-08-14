package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class KLDivergence implements LossFunction {
	/** Calculates Kullback-Leibler divergence. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		float epsilon = 1e-7f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float target = actual.values[i][j];
				if (target == 0.0f) {
					continue;
				}

				float prediction = predicted.values[i][j];
				prediction = Math.max(epsilon, Math.min(1.0f - epsilon, prediction));
				loss += target * (float) Math.log(target / prediction);
			}
		}

		return loss / predicted.rows();
	}

	/** Calculates the Kullback-Leibler divergence gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float epsilon = 1e-7f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float prediction = Math.max(epsilon, Math.min(1.0f - epsilon, predicted.values[i][j]));
				gradient.values[i][j] = -actual.values[i][j] / prediction / predicted.rows();
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
