package xyz.gatoware.synapse.matrix;

public class Matrix {
	private int rows;
	private int columns;
	private float[][] values;

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
	}

	public Matrix(float[][] values) {
		this.values = values;

		this.rows = values.length;
		this.columns = values[0].length;
	}

	public float[][] getValues() {
		return values;
	}

	public Matrix transpose(Matrix matrix) {
		float[][] m = matrix.getValues();

		float[][] temp = new float[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];

		return new Matrix(temp);
	}

	/* WORK IN PROGRESS */
	public Matrix multiply(Matrix m1, Matrix m2) {
		if (m1.columns() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		return m1;
	}

	public int rows() {
		return rows;
	}

	public int columns() {
		return columns;
	}

}
