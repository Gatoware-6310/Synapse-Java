package xyz.gatoware.synapse.activation;

public class Tanh implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) Math.tanh(num);
	}
}
