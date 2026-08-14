package xyz.gatoware.synapse.activation;

public class Sigmoid implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) (1 / (1 + Math.exp(-num)));
	}

	@Override
	public float derivative(float input, float output) {
		return output * (1.0f - output);
	}
}
