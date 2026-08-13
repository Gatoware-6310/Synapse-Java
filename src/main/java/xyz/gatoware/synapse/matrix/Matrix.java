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

		this.values = temp;
		return this;
	}

	public Matrix add(Matrix m2) {
		if (this.columns() != m2.columns() || this.rows() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		int rows = this.rows();
		int cols = this.columns();

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.values[i][j] += m2.values[i][j];
			}
		}

		return this;
	}

	public Matrix subtract(Matrix m2) {
		if (this.columns() != m2.columns() || this.rows() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		int rows = this.rows();
		int cols = this.columns();

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.values[i][j] -= m2.values[i][j];
			}
		}

		return this;
	}

	public Matrix multiply(Matrix m2) {
		if (this.columns() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		float[][] product = new float[this.rows][m2.columns];
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < m2.columns; j++) {
				for (int k = 0; k < this.columns; k++) {
					product[i][j] += this.values[i][k] * m2.values[k][j];
				}
			}
		}

		this.values = product;
		this.columns = m2.columns;
		return this;
	}

	public Matrix multiply(float scalar) {
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.values[i][j] *= scalar;
			}
		}
		return this;
	}

	public int rows() {
		return rows;
	}

	public int columns() {
		return columns;
	}

}
