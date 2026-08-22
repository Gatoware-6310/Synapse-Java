package xyz.gatoware.synapse.backend;

/** Default pure-Java CPU backend. */
public final class CpuBackend implements Backend {
	/** Shared stateless CPU backend instance. */
	public static final CpuBackend INSTANCE = new CpuBackend();

	private CpuBackend() {
	}

	@Override
	public float[][] multiply(float[][] left, float[][] right, int rows, int shared, int columns) {
		float[][] product = new float[rows][columns];

		if (columns == 1) {
			for (int i = 0; i < rows; i++) {
				float sum = 0.0f;
				float[] leftRow = left[i];
				for (int k = 0; k < shared; k++)
					sum += leftRow[k] * right[k][0];
				product[i][0] = sum;
			}
			return product;
		}

		for (int i = 0; i < rows; i++) {
			float[] leftRow = left[i];
			float[] productRow = product[i];
			for (int k = 0; k < shared; k++) {
				float leftValue = leftRow[k];
				float[] rightRow = right[k];
				for (int j = 0; j < columns; j++)
					productRow[j] += leftValue * rightRow[j];
			}
		}

		return product;
	}
}
