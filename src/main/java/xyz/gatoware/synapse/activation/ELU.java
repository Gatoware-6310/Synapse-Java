package xyz.gatoware.synapse.activation;

/** The exponential linear unit activation function. */
public class ELU implements ActivationFunction {
	private final float alpha;

	/** Creates an ELU activation with alpha equal to 1. */
	public ELU() {
		this(1.0f);
	}

	/** Creates an ELU activation with the given alpha.
	 * @param alpha the ELU alpha value
	 */
	public ELU(float alpha) {
		if (alpha <= 0) {
			throw new IllegalArgumentException("Alpha must be positive");
		}
		this.alpha = alpha;
	}

	/** Applies the ELU activation to a number. */
	@Override
	public float apply(float num) {
		return num >= 0 ? num : (float) (alpha * Math.expm1(num));
	}

	/** Returns the ELU derivative. */
	@Override
	public float derivative(float input, float output) {
		return input >= 0 ? 1.0f : output + alpha;
	}

	/** Returns the alpha value.
	 * @return the alpha value
	 */
	public float getAlpha() {
		return alpha;
	}
}
