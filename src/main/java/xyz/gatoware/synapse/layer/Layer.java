package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;

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

	/** Runs backpropagation using the given optimizer and returns the input gradient.
	 * Layers that do not override this method fall back to their existing backward implementation.
	 * @param outputGradient the gradient at the layer output
	 * @param learningRate the training learning rate
	 * @param optimizer the optimizer used to update trainable parameters
	 * @return the gradient at the layer input
	 */
	default Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		return backward(outputGradient, learningRate);
	}
}
