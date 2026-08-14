package xyz.gatoware.synapse.activation;

public class Sigmoid implements ActivationFunction {
	/** Applies the Sigmoid activation to a number. */
	@Override
	public float apply(float num) {
		return (float) (1 / (1 + Math.exp(-num)));
	}

	/** Returns the Sigmoid derivative. */
	@Override
	public float derivative(float input, float output) {
		return output * (1.0f - output);
	}
}
