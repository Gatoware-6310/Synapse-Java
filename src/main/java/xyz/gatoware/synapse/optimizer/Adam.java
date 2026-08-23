package xyz.gatoware.synapse.optimizer;

import java.util.IdentityHashMap;
import java.util.Map;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.matrix.Matrix;

/** Adam optimizer using first and second gradient moments. */
public class Adam implements Optimizer {
	private final float beta1;
	private final float beta2;
	private final float epsilon;
	private final Map<Matrix, State> states = new IdentityHashMap<>();

	private static class State {
		private final Matrix firstMoment;
		private final Matrix secondMoment;
		private int step;
		private State(int rows, int columns) {
			firstMoment = new Matrix(rows, columns);
			secondMoment = new Matrix(rows, columns);
		}
	}

	/** Creates an Adam optimizer with the default beta and epsilon values. */
	public Adam() { this(0.9f, 0.999f, 1e-7f); }

	/** Creates an Adam optimizer.
	 * @param beta1 decay rate for the first gradient moment
	 * @param beta2 decay rate for the second gradient moment
	 * @param epsilon small positive value used to avoid division by zero
	 */
	public Adam(float beta1, float beta2, float epsilon) {
		if (!Float.isFinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f)
			throw new IllegalArgumentException("Beta1 must be finite and in [0, 1)");
		if (!Float.isFinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f)
			throw new IllegalArgumentException("Beta2 must be finite and in [0, 1)");
		if (!Float.isFinite(epsilon) || epsilon <= 0.0f)
			throw new IllegalArgumentException("Epsilon must be positive and finite");
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}

	/** Returns the first-moment decay rate.
	 * @return the beta1 value
	 */
	public float getBeta1() { return beta1; }

	/** Returns the second-moment decay rate.
	 * @return the beta2 value
	 */
	public float getBeta2() { return beta2; }

	/** Returns the epsilon value used by this optimizer.
	 * @return the epsilon value
	 */
	public float getEpsilon() { return epsilon; }

	@Override
	public void update(Matrix parameters, Matrix gradients, float learningRate) {
		validate(parameters, gradients, learningRate);
		State state = states.computeIfAbsent(parameters, parameter -> new State(parameter.rows(), parameter.columns()));
		state.step++;
		float firstCorrection = 1.0f - (float) Math.pow(beta1, state.step);
		float secondCorrection = 1.0f - (float) Math.pow(beta2, state.step);
		for (int row = 0; row < parameters.rows(); row++) {
			for (int column = 0; column < parameters.columns(); column++) {
				float gradient = gradients.values[row][column];
				state.firstMoment.values[row][column] = beta1 * state.firstMoment.values[row][column] + (1.0f - beta1) * gradient;
				state.secondMoment.values[row][column] = beta2 * state.secondMoment.values[row][column] + (1.0f - beta2) * gradient * gradient;
				float first = state.firstMoment.values[row][column] / firstCorrection;
				float second = state.secondMoment.values[row][column] / secondCorrection;
				parameters.values[row][column] -= learningRate * first / ((float) Math.sqrt(second) + epsilon);
			}
		}
	}

	@Override
	public void reset() {
		states.clear();
		if (Synapse.backend() instanceof CudaBackend cuda)
			cuda.resetOptimizer(this);
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
