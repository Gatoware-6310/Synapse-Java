package xyz.gatoware.synapse.activation;

public class Swish implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) (num / (1 + Math.exp(-num)));
	}
}
