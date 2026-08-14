package xyz.gatoware.synapse.activation;

/** The rectified linear unit activation function. */
public class ReLU implements ActivationFunction {
	/** Creates a ReLU activation. */
	public ReLU() {
	}

	/** Applies the ReLU activation to a number. */
	@Override
	public float apply(float num) {
		return num < 0 ? 0 : num;
	}

	/** Returns the ReLU derivative. */
	@Override
	public float derivative(float input, float output) {
		return input > 0 ? 1.0f : 0.0f;
	}
}
