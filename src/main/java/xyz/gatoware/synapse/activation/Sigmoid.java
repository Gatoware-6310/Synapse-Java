package xyz.gatoware.synapse.activation;

/** The sigmoid activation function. */
public class Sigmoid implements ActivationFunction {
	/** Creates a Sigmoid activation. */
	public Sigmoid() {
	}
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
