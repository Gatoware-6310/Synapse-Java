package xyz.gatoware.synapse.activation;

public class ReLU implements ActivationFunction {

	@Override
	public float apply(float num) {
		return num < 0 ? 0 : num;
	}
}
