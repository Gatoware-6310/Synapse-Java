package xyz.gatoware.synapse.optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

import xyz.gatoware.synapse.matrix.Matrix;

/** RMSProp optimizer. */
public class RMSProp implements Optimizer {
	private final float decay;
	private final float epsilon;
	private final Map<Matrix, Matrix> squaredAverages = new IdentityHashMap<>();

	/** Creates an RMSProp optimizer using decay 0.9 and epsilon 1e-7. */
	public RMSProp() {
		this(0.9f, 1e-7f);
	}

	/** Creates an RMSProp optimizer.
	 * @param decay the squared-gradient decay coefficient in [0, 1)
	 * @param epsilon a small positive value used for numerical stability
	 */
	public RMSProp(float decay, float epsilon) {
		if (!Float.isFinite(decay) || decay < 0.0f || decay >= 1.0f)
			throw new IllegalArgumentException("Decay must be finite and in [0, 1)");
		if (!Float.isFinite(epsilon) || epsilon <= 0.0f)
			throw new IllegalArgumentException("Epsilon must be positive and finite");
		this.decay = decay;
		this.epsilon = epsilon;
	}

	@Override
	public void update(Matrix parameters, Matrix gradients, float learningRate) {
		validate(parameters, gradients, learningRate);
		Matrix average = squaredAverages.computeIfAbsent(parameters,
				parameter -> new Matrix(parameter.rows(), parameter.columns()));
		for (int row = 0; row < parameters.rows(); row++) {
			for (int column = 0; column < parameters.columns(); column++) {
				float gradient = gradients.values[row][column];
				average.values[row][column] = decay * average.values[row][column]
						+ (1.0f - decay) * gradient * gradient;
				parameters.values[row][column] -= learningRate * gradient
						/ ((float) Math.sqrt(average.values[row][column]) + epsilon);
			}
		}
	}

	@Override
	public void reset() {
		squaredAverages.clear();
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
