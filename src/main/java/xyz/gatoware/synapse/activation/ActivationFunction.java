package xyz.gatoware.synapse.activation;

public interface ActivationFunction {
	/* Apply the activation on any number */
	float apply(float num);

	default float[] apply(float[] values) {
		float[] result = values.clone();
		for (int i = 0; i < result.length; i++)
			result[i] = apply(result[i]);
		return result;
	}
}
