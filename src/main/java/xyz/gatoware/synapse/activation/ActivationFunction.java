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
		float[] result = new float[values.length];
		apply(values, result);
		return result;
	}

	/** Applies the activation into an existing result array.
	 * @param values the values to activate
	 * @param result storage for the activated values
	 */
	default void apply(float[] values, float[] result) {
		if (values.length != result.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");
		for (int i = 0; i < result.length; i++)
			result[i] = apply(values[i]);
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
		backward(inputs, outputs, gradients, result);
		return result;
	}

	/** Applies the activation derivative into an existing result array.
	 * @param inputs the activation inputs
	 * @param outputs the activation outputs
	 * @param gradients the output gradients
	 * @param result storage for the input gradients
	 */
	default void backward(float[] inputs, float[] outputs, float[] gradients, float[] result) {
		if (inputs.length != outputs.length || outputs.length != gradients.length || gradients.length != result.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");

		for (int i = 0; i < result.length; i++)
			result[i] = gradients[i] * derivative(inputs[i], outputs[i]);
	}
}
