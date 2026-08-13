package xyz.gatoware.synapse.utils;

import xyz.gatoware.synapse.matrix.Matrix;

public class MatrixUtils {
	public static void printMatrix(Matrix m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.columns(); j++) {
				System.out.print(m.values[i][j] + " ");
			}
			System.out.println();
		}
	}
}
