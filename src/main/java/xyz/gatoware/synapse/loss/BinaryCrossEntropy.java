package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class BinaryCrossEntropy implements LossFunction {
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		float epsilon = 1e-7f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float prediction = predicted.values[i][j];
				prediction = Math.max(epsilon, Math.min(1.0f - epsilon, prediction));
				float target = actual.values[i][j];
				loss += target * (float) Math.log(prediction)
					+ (1.0f - target) * (float) Math.log(1.0f - prediction);
			}
		}

		return -loss / (predicted.rows() * predicted.columns());
	}

	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float epsilon = 1e-7f;
		float scale = 1.0f / (predicted.rows() * predicted.columns());
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float prediction = Math.max(epsilon, Math.min(1.0f - epsilon, predicted.values[i][j]));
				float target = actual.values[i][j];
				gradient.values[i][j] = (prediction - target) / (prediction * (1.0f - prediction)) * scale;
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
