package xyz.gatoware.synapse.loss;

import xyz.gatoware.synapse.matrix.Matrix;

public class HuberLoss implements LossFunction {
	private final float delta;

	/** Creates a Huber loss with delta equal to 1. */
	public HuberLoss() {
		this(1.0f);
	}

	/** Creates a Huber loss with the given delta. */
	public HuberLoss(float delta) {
		if (delta <= 0) {
			throw new IllegalArgumentException("Delta must be positive");
		}
		this.delta = delta;
	}

	/** Calculates Huber loss. */
	@Override
	public float calculate(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);

		float loss = 0.0f;
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float difference = Math.abs(predicted.values[i][j] - actual.values[i][j]);
				loss += difference <= delta
					? 0.5f * difference * difference
					: delta * (difference - 0.5f * delta);
			}
		}

		return loss / (predicted.rows() * predicted.columns());
	}

	/** Calculates the Huber-loss gradient. */
	@Override
	public Matrix gradient(Matrix predicted, Matrix actual) {
		validateDimensions(predicted, actual);
		Matrix gradient = new Matrix(predicted.rows(), predicted.columns());
		float scale = 1.0f / (predicted.rows() * predicted.columns());
		for (int i = 0; i < predicted.rows(); i++) {
			for (int j = 0; j < predicted.columns(); j++) {
				float difference = predicted.values[i][j] - actual.values[i][j];
				gradient.values[i][j] = (Math.abs(difference) <= delta
						? difference
						: delta * Math.signum(difference)) * scale;
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
