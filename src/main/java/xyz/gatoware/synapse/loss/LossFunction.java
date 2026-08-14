package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public interface LossFunction {
	float calculate(Matrix predicted, Matrix actual);

	default Matrix gradient(Matrix predicted, Matrix actual) {
		float epsilon = 1e-3f;
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				Matrix above = predicted.copy();
				Matrix below = predicted.copy();
				above.values[i][j] += epsilon;
				below.values[i][j] -= epsilon;
				gradient.values[i][j] = (calculate(above, actual) - calculate(below, actual)) / (2.0f * epsilon);
			}
		}
		return gradient;
	}
}
