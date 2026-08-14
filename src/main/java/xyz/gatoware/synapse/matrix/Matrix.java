package xyz.gatoware.synapse.matrix;

import xyz.gatoware.synapse.activation.ActivationFunction;

public class Matrix {
	private int rows;
	private int columns;
	public float[][] values;

	/** Creates a matrix with the given amount of rows and columns. */
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;

		this.values = new float[rows][columns];
	}

	/** Creates a matrix from floating point values. */
	public Matrix(float[][] values) {
		this.values = values;

		this.rows = values.length;
		this.columns = values[0].length;
	}

	/** Returns a copy of this matrix. */
	public Matrix copy() {
		float[][] copy = new float[this.rows][this.columns];
		for (int i = 0; i < this.rows; i++)
			System.arraycopy(this.values[i], 0, copy[i], 0, this.columns);

		return new Matrix(copy);
	}

	/** Transposes this matrix and returns it. */
	public Matrix transpose() {
		float[][] m = this.values;

		float[][] temp = new float[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];

		this.values = temp;
		return this;
	}

	/** Adds another matrix to this matrix and returns it. */
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

	/** Subtracts another matrix from this matrix and returns it. */
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

	/** Multiplies this matrix by another matrix and returns it. */
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

	/** Multiplies every value in this matrix by a scalar and returns it. */
	public Matrix multiply(float scalar) {
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.values[i][j] *= scalar;
			}
		}
		return this;
	}

	/** Multiplies this matrix element by element with another matrix and returns it. */
	public Matrix multiply_hadamard(Matrix m2) {
		if (this.columns() != m2.columns() || this.rows() != m2.rows()) {
			throw new IllegalArgumentException("Matrix dimensions are incompatible!");
		}

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {
				this.values[i][j] *= m2.values[i][j];
			}
		}

		return this;
	}

	/** Applies an activation function to every value in this matrix and returns it. */
	public Matrix apply(ActivationFunction func) {
		if (rows == 1 || columns == 1) {
			float[] vector = new float[Math.max(rows, columns)];
			for (int i = 0; i < vector.length; i++)
				vector[i] = rows == 1 ? values[0][i] : values[i][0];

			vector = func.apply(vector);
			for (int i = 0; i < vector.length; i++) {
				if (rows == 1)
					values[0][i] = vector[i];
				else
					values[i][0] = vector[i];
			}
			return this;
		}

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				values[i][j] = func.apply(values[i][j]);

		return this;
	}

	/** Returns the amount of rows in this matrix. */
	public int rows() {
		return rows;
	}

	/** Returns the amount of columns in this matrix. */
	public int columns() {
		return columns;
	}

}
