package xyz.gatoware.synapse.matrix;

public class Matrix {
	private int rows;
	private int columns;
	public float[][] values;

	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;

		this.values = new float[rows][columns];
	}

	public Matrix(float[][] values) {
		this.values = values;

		this.rows = values.length;
		this.columns = values[0].length;
	}

	public Matrix transpose() {
		float[][] m = this.values;

		float[][] temp = new float[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];

		return new Matrix(temp);
	}

	public Matrix add(Matrix m2) {
		if (this.columns() != m2.columns() || this.rows() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		int rows = this.rows();
		int cols = this.columns();

		float[][] sum = new float[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				sum[i][j] = this.values[i][j] + m2.values[i][j];
			}
		}

		return new Matrix(sum);
	}

	public Matrix multiply(Matrix m2) {
		if (this.columns() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		final Matrix m3 = new Matrix(this.rows, m2.columns);

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < m2.columns; j++) {
				for (int k = 0; k < this.columns; k++) {
					m3.values[i][j] += this.values[i][k] * m2.values[k][j];
				}
			}
		}

		return m3;
	}

	public int rows() {
		return rows;
	}

	public int columns() {
		return columns;
	}

}
