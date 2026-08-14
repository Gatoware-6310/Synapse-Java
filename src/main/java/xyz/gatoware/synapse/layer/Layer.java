package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;

public interface Layer {
	Matrix forward(Matrix input);

	default Matrix backward(Matrix outputGradient, float learningRate) {
		throw new UnsupportedOperationException("Layer does not support training");
	}
}
