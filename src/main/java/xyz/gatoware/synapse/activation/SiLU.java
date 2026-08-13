package xyz.gatoware.synapse.activation;

public class SiLU implements ActivationFunction {
	@Override
	public float apply(float num) {
		return (float) (num / (1 + Math.exp(-num)));
	}
}
