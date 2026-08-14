package xyz.gatoware.synapse.activation;

public class Tanh implements ActivationFunction {
	/** Applies the hyperbolic tangent activation to a number. */
	@Override
	public float apply(float num) {
		return (float) Math.tanh(num);
	}

	/** Returns the hyperbolic tangent derivative. */
	@Override
	public float derivative(float input, float output) {
		return 1.0f - output * output;
	}
}
