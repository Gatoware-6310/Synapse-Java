package xyz.gatoware.synapse.activation;

public class LeakyReLU implements ActivationFunction {
	private final float slope;

	public LeakyReLU() {
		this(0.01f);
	}

	public LeakyReLU(float slope) {
		if (slope < 0) {
			throw new IllegalArgumentException("Slope cannot be negative");
		}
		this.slope = slope;
	}

	@Override
	public float apply(float num) {
		return num < 0 ? slope * num : num;
	}

	@Override
	public float derivative(float input, float output) {
		return input < 0 ? slope : 1.0f;
	}

	public float getSlope() {
		return slope;
	}
}
