package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

/** A function that calculates the difference between predicted and actual values. */
public interface LossFunction {
	/** Calculates the loss between predicted and actual values.
	 * @param predicted the predicted values
	 * @param actual the actual values
	 * @return the calculated loss
	 */
	float calculate(Matrix predicted, Matrix actual);

	/** Calculates the gradient of the loss with respect to the predicted values.
	 * @param predicted the predicted values
	 * @param actual the actual values
	 * @return the loss gradient
	 */
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
