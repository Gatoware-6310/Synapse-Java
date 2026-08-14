package xyz.gatoware.synapse.activation;

public class LeakyReLU implements ActivationFunction {
	private final float slope;

	/** Creates a LeakyReLU activation with a slope of 0.01. */
	public LeakyReLU() {
		this(0.01f);
	}

	/** Creates a LeakyReLU activation with the given slope. */
	public LeakyReLU(float slope) {
		if (slope < 0) {
			throw new IllegalArgumentException("Slope cannot be negative");
		}
		this.slope = slope;
	}

	/** Applies the LeakyReLU activation to a number. */
	@Override
	public float apply(float num) {
		return num < 0 ? slope * num : num;
	}

	/** Returns the LeakyReLU derivative. */
	@Override
	public float derivative(float input, float output) {
		return input < 0 ? slope : 1.0f;
	}

	/** Returns the negative-input slope. */
	public float getSlope() {
		return slope;
	}
}
