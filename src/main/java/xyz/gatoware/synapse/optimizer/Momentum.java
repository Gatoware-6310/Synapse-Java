package xyz.gatoware.synapse.optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

import xyz.gatoware.synapse.matrix.Matrix;

/** Gradient descent with momentum. */
public class Momentum implements Optimizer {
	private final float momentum;
	private final Map<Matrix, Matrix> velocities = new IdentityHashMap<>();

	/** Creates a momentum optimizer using a momentum of 0.9. */
	public Momentum() {
		this(0.9f);
	}

	/** Creates a momentum optimizer.
	 * @param momentum the momentum coefficient in [0, 1)
	 */
	public Momentum(float momentum) {
		if (!Float.isFinite(momentum) || momentum < 0.0f || momentum >= 1.0f)
			throw new IllegalArgumentException("Momentum must be finite and in [0, 1)");
		this.momentum = momentum;
	}

	@Override
	public void update(Matrix parameters, Matrix gradients, float learningRate) {
		validate(parameters, gradients, learningRate);
		Matrix velocity = velocities.computeIfAbsent(parameters,
				parameter -> new Matrix(parameter.rows(), parameter.columns()));
		for (int row = 0; row < parameters.rows(); row++) {
			for (int column = 0; column < parameters.columns(); column++) {
				velocity.values[row][column] = momentum * velocity.values[row][column]
						+ gradients.values[row][column];
				parameters.values[row][column] -= learningRate * velocity.values[row][column];
			}
		}
	}

	@Override
	public void reset() {
		velocities.clear();
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
