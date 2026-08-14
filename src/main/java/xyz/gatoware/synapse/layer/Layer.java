package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;

/** A layer in a neural network. */
public interface Layer {
	/** Runs the layer on the given input.
	 * @param input the layer input
	 * @return the layer output
	 */
	Matrix forward(Matrix input);

	/** Runs backpropagation for the layer and returns the input gradient.
	 * @param outputGradient the gradient at the layer output
	 * @param learningRate the training learning rate
	 * @return the gradient at the layer input
	 */
	default Matrix backward(Matrix outputGradient, float learningRate) {
		throw new UnsupportedOperationException("Layer does not support training");
	}
}
