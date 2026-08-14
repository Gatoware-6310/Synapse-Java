package xyz.gatoware.synapse.activation;

public class GELU implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) (0.5 * num * (1 + erf(num / Math.sqrt(2))));
	}

	@Override
	public float derivative(float input, float output) {
		double cdf = 0.5 * (1.0 + erf(input / Math.sqrt(2.0)));
		double density = Math.exp(-0.5 * input * input) / Math.sqrt(2.0 * Math.PI);
		return (float) (cdf + input * density);
	}

	private double erf(double value) {
		double sign = value < 0 ? -1 : 1;
		value = Math.abs(value);
		double t = 1.0 / (1.0 + 0.3275911 * value);
		double polynomial = (((((1.061405429 * t - 1.453152027) * t)
				+ 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
		return sign * (1 - polynomial * Math.exp(-value * value));
	}
}
