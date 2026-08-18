package xyz.gatoware.synapse.optimizer;

import xyz.gatoware.synapse.matrix.Matrix;

/** Stochastic gradient descent optimizer. */
public class SGD implements Optimizer {
	@Override
	public void update(Matrix parameters, Matrix gradients, float learningRate) {
		validate(parameters, gradients, learningRate);
		for (int row = 0; row < parameters.rows(); row++)
			for (int column = 0; column < parameters.columns(); column++)
				parameters.values[row][column] -= learningRate * gradients.values[row][column];
	}

	private static void validate(Matrix parameters, Matrix gradients, float learningRate) {
		if (parameters == null || gradients == null)
			throw new IllegalArgumentException("Parameters and gradients cannot be null");
		if (parameters.rows() != gradients.rows() || parameters.columns() != gradients.columns())
			throw new IllegalArgumentException("Parameter and gradient dimensions must match");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
	}
}
