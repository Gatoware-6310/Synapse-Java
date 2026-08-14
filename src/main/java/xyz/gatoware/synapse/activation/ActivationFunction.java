package xyz.gatoware.synapse.activation;

/** An activation function used by a neural network layer. */
public interface ActivationFunction {
	/** Applies the activation to a number.
	 * @param num the number to activate
	 * @return the activated number
	 */
	float apply(float num);

	/** Returns the derivative of the activation for an input and its output.
	 * @param input the activation input
	 * @param output the activation output
	 * @return the activation derivative
	 */
	default float derivative(float input, float output) {
		float epsilon = 1e-3f;
		return (apply(input + epsilon) - apply(input - epsilon)) / (2.0f * epsilon);
	}

	/** Applies the activation to every value in an array.
	 * @param values the values to activate
	 * @return the activated values
	 */
	default float[] apply(float[] values) {
		float[] result = values.clone();
		for (int i = 0; i < result.length; i++)
			result[i] = apply(result[i]);
		return result;
	}

	/** Applies the activation derivative to a vector of gradients.
	 * @param inputs the activation inputs
	 * @param outputs the activation outputs
	 * @param gradients the output gradients
	 * @return the input gradients
	 */
	default float[] backward(float[] inputs, float[] outputs, float[] gradients) {
		if (inputs.length != outputs.length || outputs.length != gradients.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");

		float[] result = new float[gradients.length];
		for (int i = 0; i < result.length; i++)
			result[i] = gradients[i] * derivative(inputs[i], outputs[i]);
		return result;
	}
}
