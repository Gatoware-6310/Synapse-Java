package xyz.gatoware.synapse.activation;

public class Tanh implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) Math.tanh(num);
	}

	@Override
	public float derivative(float input, float output) {
		return 1.0f - output * output;
	}
}
