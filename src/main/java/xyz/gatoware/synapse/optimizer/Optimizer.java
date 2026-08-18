package xyz.gatoware.synapse.optimizer;

import xyz.gatoware.synapse.matrix.Matrix;

/** Updates trainable parameters using their gradients. */
public interface Optimizer {
	/** Updates a parameter matrix in place.
	 * @param parameters the parameters to update
	 * @param gradients the gradients for the parameters
	 * @param learningRate the training learning rate
	 */
	void update(Matrix parameters, Matrix gradients, float learningRate);

	/** Clears any optimizer state accumulated for parameters. */
	default void reset() {
	}
}
