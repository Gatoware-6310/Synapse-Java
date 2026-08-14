package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class SparseCategoricalCrossEntropy implements LossFunction {
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		if (predicted.rows() != actual.rows() || actual.columns() != 1) {
			throw new IllegalArgumentException(
				"Predicted matrix rows must match actual class IDs"
			);
		}

		float loss = 0.0f;
		float epsilon = 1e-7f;
		for (int i = 0; i < predicted.rows(); i++) {
			float classId = actual.values[i][0];
			int targetClass = (int) classId;
			if (classId != targetClass || targetClass < 0 || targetClass >= predicted.columns()) {
				throw new IllegalArgumentException("Actual matrix must contain valid class IDs");
			}

			float prediction = predicted.values[i][targetClass];
			prediction = Math.max(epsilon, Math.min(1.0f - epsilon, prediction));
			loss += (float) Math.log(prediction);
		}

		return -loss / predicted.rows();
	}

	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validate(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float epsilon = 1e-7f;
		for (int i = 0; i < predicted.rows(); i++) {
			int targetClass = (int) actual.values[i][0];
			float prediction = Math.max(epsilon, Math.min(1.0f - epsilon, predicted.values[i][targetClass]));
			gradient.values[i][targetClass] = -1.0f / prediction / predicted.rows();
		}
		return gradient;
	}

	private void validate(Matrix predicted, Matrix actual) {
		if (predicted.rows() != actual.rows() || actual.columns() != 1)
			throw new IllegalArgumentException("Predicted matrix rows must match actual class IDs");
		for (int i = 0; i < actual.rows(); i++) {
			float classId = actual.values[i][0];
			int targetClass = (int) classId;
			if (classId != targetClass || targetClass < 0 || targetClass >= predicted.columns())
				throw new IllegalArgumentException("Actual matrix must contain valid class IDs");
		}
	}
}
