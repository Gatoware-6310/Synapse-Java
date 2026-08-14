package xyz.gatoware.synapse.activation;

public class ReLU implements ActivationFunction {

	@Override
	public float apply(float num) {
		return num < 0 ? 0 : num;
	}

	@Override
	public float derivative(float input, float output) {
		return input > 0 ? 1.0f : 0.0f;
	}
}
