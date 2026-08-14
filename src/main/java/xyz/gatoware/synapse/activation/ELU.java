package xyz.gatoware.synapse.activation;

public class ELU implements ActivationFunction {
	private final float alpha;

	public ELU() {
		this(1.0f);
	}

	public ELU(float alpha) {
		if (alpha <= 0) {
			throw new IllegalArgumentException("Alpha must be positive");
		}
		this.alpha = alpha;
	}

	@Override
	public float apply(float num) {
		return num >= 0 ? num : (float) (alpha * Math.expm1(num));
	}

	@Override
	public float derivative(float input, float output) {
		return input >= 0 ? 1.0f : output + alpha;
	}

	public float getAlpha() {
		return alpha;
	}
}
