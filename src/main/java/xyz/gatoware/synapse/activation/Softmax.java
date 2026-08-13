package xyz.gatoware.synapse.activation;

public class Softmax implements ActivationFunction {
	@Override
	public float apply(float num) {
		return 1.0f;
	}

	public float[] apply(float[] values) {
		if (values.length == 0) {
			return new float[0];
		}
		float max = values[0];
		for (float value : values) max = Math.max(max, value);
		float sum = 0;
		float[] result = new float[values.length];
		for (int i = 0; i < values.length; i++) sum += result[i] = (float) Math.exp(values[i] - max);
		for (int i = 0; i < result.length; i++) result[i] /= sum;
		return result;
	}
}
