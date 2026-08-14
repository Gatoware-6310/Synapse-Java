package xyz.gatoware.synapse.activation;

/** The Swish activation function. */
public class Swish implements ActivationFunction {
	/** Creates a Swish activation. */
	public Swish() {
	}
	/** Applies the Swish activation to a number. */
	@Override
	public float apply(float num) {
		return (float) (num / (1 + Math.exp(-num)));
	}

	/** Returns the Swish derivative. */
	@Override
	public float derivative(float input, float output) {
		float sigmoid = (float) (1.0 / (1.0 + Math.exp(-input)));
		return sigmoid + input * sigmoid * (1.0f - sigmoid);
	}
}
