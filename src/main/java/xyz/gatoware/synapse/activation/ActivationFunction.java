package xyz.gatoware.synapse.activation;

public interface ActivationFunction {
	/* Apply the activation on any number */
	float apply(float num);

	default float derivative(float input, float output) {
		float epsilon = 1e-3f;
		return (apply(input + epsilon) - apply(input - epsilon)) / (2.0f * epsilon);
	}

	default float[] apply(float[] values) {
		float[] result = values.clone();
		for (int i = 0; i < result.length; i++)
			result[i] = apply(result[i]);
		return result;
	}

	default float[] backward(float[] inputs, float[] outputs, float[] gradients) {
		if (inputs.length != outputs.length || outputs.length != gradients.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");

		float[] result = new float[gradients.length];
		for (int i = 0; i < result.length; i++)
			result[i] = gradients[i] * derivative(inputs[i], outputs[i]);
		return result;
	}
}
