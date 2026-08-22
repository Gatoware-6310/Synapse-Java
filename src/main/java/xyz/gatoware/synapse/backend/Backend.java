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

	/** Invalidates any backend-side cached representation of the given host matrix.
	 * CPU backends do not need to do anything.
	 * @param matrix host matrix values that were modified
	 */
	default void invalidate(float[][] matrix) {
		// Most backends do not cache host matrices.
	}

	/** Releases backend resources. */
	@Override
	default void close() {
		// Most backends do not own resources.
	}
}
