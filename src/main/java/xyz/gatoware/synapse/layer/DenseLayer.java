package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.matrix.Matrix;

public class DenseLayer implements Layer {
	
	private Matrix weights;
	private Matrix biases;
	private ActivationFunction activationFunction;

	public DenseLayer (int inputSize, int outputSize, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
		this.weights = new Matrix(outputSize, inputSize);
		this.biases = new Matrix(outputSize, 1);

		float scale = (float) (1.0 / Math.sqrt(inputSize));
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				this.weights.values[i][j] = (float) ((Math.random() * 2 - 1) * scale);
			}
		}
	}
	
	@Override
	public Matrix forward(Matrix input) {
		if (input.rows() != weights.columns() || input.columns() != 1) {
			throw new IllegalArgumentException("Dense layer input must have dimensions " + weights.columns() + " x 1");
		}

		return weights.copy().multiply(input).add(biases).apply(activationFunction);
	}

}
