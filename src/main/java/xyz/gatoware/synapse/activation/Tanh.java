package xyz.gatoware.synapse.activation;

/** The hyperbolic tangent activation function. */
public class Tanh implements ActivationFunction {
	/** Creates a hyperbolic tangent activation. */
	public Tanh() {
	}
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
