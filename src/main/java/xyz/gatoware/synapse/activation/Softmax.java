package xyz.gatoware.synapse.activation;

/** The Softmax activation function. */
public class Softmax implements ActivationFunction {
	/** Creates a Softmax activation. */
	public Softmax() {
	}
	/** Returns 1 for scalar application; Softmax is applied to vectors. */
	@Override
	public float apply(float num) {
		return 1.0f;
	}

	/** Applies Softmax to a vector of values. */
	public float[] apply(float[] values) {
		float[] result = new float[values.length];
		apply(values, result);
		return result;
	}

	/** Applies Softmax into an existing result array. */
	@Override
	public void apply(float[] values, float[] result) {
		if (values.length != result.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");
		if (values.length == 0)
			return;
		float max = values[0];
		for (float value : values) max = Math.max(max, value);
		float sum = 0;
		for (int i = 0; i < values.length; i++) sum += result[i] = (float) Math.exp(values[i] - max);
		for (int i = 0; i < result.length; i++) result[i] /= sum;
	}

	/** Applies the Softmax Jacobian to a vector of gradients. */
	@Override
	public float[] backward(float[] inputs, float[] outputs, float[] gradients) {
		if (inputs.length != outputs.length || outputs.length != gradients.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");

		float[] result = new float[outputs.length];
		backward(inputs, outputs, gradients, result);
		return result;
	}

	/** Applies the Softmax Jacobian into an existing result array. */
	@Override
	public void backward(float[] inputs, float[] outputs, float[] gradients, float[] result) {
		if (inputs.length != outputs.length || outputs.length != gradients.length || gradients.length != result.length)
			throw new IllegalArgumentException("Activation vectors must have the same length");

		float dot = 0.0f;
		for (int i = 0; i < outputs.length; i++)
			dot += outputs[i] * gradients[i];

		for (int i = 0; i < result.length; i++)
			result[i] = outputs[i] * (gradients[i] - dot);
	}
}
