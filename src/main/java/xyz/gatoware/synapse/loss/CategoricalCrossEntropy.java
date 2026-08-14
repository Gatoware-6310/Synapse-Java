package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class CategoricalCrossEntropy implements LossFunction {
	/** Calculates categorical cross-entropy loss. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		if (predicted.rows() != actual.rows() || predicted.columns() != actual.columns()) {
			throw new IllegalArgumentException(
				"Predicted and actual matrices must have the same dimensions"
			);
		}

		float loss = 0.0f;
		float epsilon = 1e-7f;

		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float prediction = predicted.values[i][j];
				prediction = Math.max(epsilon, Math.min(1.0f - epsilon, prediction));
				loss += actual.values[i][j] * (float) Math.log(prediction);
			}
		}

		return -loss / predicted.rows();
	}

	/** Calculates the categorical cross-entropy gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		if (predicted.rows() != actual.rows() || predicted.columns() != actual.columns()) {
			throw new IllegalArgumentException("Predicted and actual matrices must have the same dimensions");
		}

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
}
