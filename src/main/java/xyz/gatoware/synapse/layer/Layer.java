package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;

public interface Layer {
	/** Runs the layer on the given input. */
	Matrix forward(Matrix input);

	/** Runs backpropagation for the layer and returns the input gradient. */
	default Matrix backward(Matrix outputGradient, float learningRate) {
		throw new UnsupportedOperationException("Layer does not support training");
	}
}
