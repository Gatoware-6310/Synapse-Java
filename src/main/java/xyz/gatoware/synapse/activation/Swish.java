package xyz.gatoware.synapse.activation;

public class Swish implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) (num / (1 + Math.exp(-num)));
	}

	@Override
	public float derivative(float input, float output) {
		float sigmoid = (float) (1.0 / (1.0 + Math.exp(-input)));
		return sigmoid + input * sigmoid * (1.0f - sigmoid);
	}
}
