package xyz.gatoware.synapse.utils;

import xyz.gatoware.synapse.matrix.Matrix;

/** Utility functions for matrices. */
public class MatrixUtils {
	/** Creates matrix utilities. */
	public MatrixUtils() {
	}

	/** Prints a matrix to standard output.
	 * @param m the matrix to print
	 */
	public static void printMatrix(Matrix m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.columns(); j++) {
				System.out.print(m.values[i][j] + " ");
			}
			System.out.println();
		}
	}
}
