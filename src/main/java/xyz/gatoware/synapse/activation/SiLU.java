package xyz.gatoware.synapse.activation;

/** The sigmoid linear unit activation function. */
public class SiLU implements ActivationFunction {
	/** Creates a SiLU activation. */
	public SiLU() {
	}
	/** Applies the SiLU activation to a number. */
	@Override
	public float apply(float num) {
		return (float) (num / (1 + Math.exp(-num)));
	}

	/** Returns the SiLU derivative. */
	@Override
	public float derivative(float input, float output) {
		float sigmoid = (float) (1.0 / (1.0 + Math.exp(-input)));
		return sigmoid + input * sigmoid * (1.0f - sigmoid);
	}
}
