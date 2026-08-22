package xyz.gatoware.synapse.backend;

/** Internal compute backend used by matrix and layer operations. */
public interface Backend extends AutoCloseable {
	/** Multiplies two row-major matrices.
	 * @param left left matrix values
	 * @param right right matrix values
	 * @param rows amount of rows in the left matrix
	 * @param shared shared dimension
	 * @param columns amount of columns in the right matrix
	 * @return the matrix product
	 */
	float[][] multiply(float[][] left, float[][] right, int rows, int shared, int columns);

	/** Releases backend resources. */
	@Override
	default void close() {
		// Most backends do not own resources.
	}
}
