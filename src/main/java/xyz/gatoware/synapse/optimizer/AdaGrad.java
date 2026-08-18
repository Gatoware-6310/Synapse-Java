package xyz.gatoware.synapse.optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

import xyz.gatoware.synapse.matrix.Matrix;

/** AdaGrad optimizer with per-parameter adaptive learning rates. */
public class AdaGrad implements Optimizer {
	private final float epsilon;
	private final Map<Matrix, Matrix> squaredGradients = new IdentityHashMap<>();

	/** Creates an AdaGrad optimizer using epsilon 1e-7. */
	public AdaGrad() {
		this(1e-7f);
	}

	/** Creates an AdaGrad optimizer.
	 * @param epsilon a small positive value used for numerical stability
	 */
	public AdaGrad(float epsilon) {
		if (!Float.isFinite(epsilon) || epsilon <= 0.0f)
			throw new IllegalArgumentException("Epsilon must be positive and finite");
		this.epsilon = epsilon;
	}

	@Override
	public void update(Matrix parameters, Matrix gradients, float learningRate) {
		validate(parameters, gradients, learningRate);
		Matrix accumulator = squaredGradients.computeIfAbsent(parameters,
				parameter -> new Matrix(parameter.rows(), parameter.columns()));
		for (int row = 0; row < parameters.rows(); row++) {
			for (int column = 0; column < parameters.columns(); column++) {
				float gradient = gradients.values[row][column];
				accumulator.values[row][column] += gradient * gradient;
				parameters.values[row][column] -= learningRate * gradient
						/ ((float) Math.sqrt(accumulator.values[row][column]) + epsilon);
			}
		}
	}

	@Override
	public void reset() {
		squaredGradients.clear();
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
