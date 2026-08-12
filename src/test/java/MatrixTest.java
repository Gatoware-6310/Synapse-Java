import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import xyz.gatoware.synapse.matrix.Matrix;

public class MatrixTest {

	@Test
	void valuesConstructorStoresValuesAndDimensions() {
		float[][] values = {
				{1.0f, 2.0f, 3.0f},
				{4.0f, 5.0f, 6.0f}
		};

		Matrix matrix = new Matrix(values);

		assertEquals(2, matrix.rows());
		assertEquals(3, matrix.columns());
		assertArrayEquals(values[0], matrix.values[0]);
		assertArrayEquals(values[1], matrix.values[1]);
	}

	@Test
	void dimensionsConstructorCreatesMatrixWithRequestedDimensions() {
		Matrix matrix = new Matrix(2, 3);

		assertEquals(2, matrix.rows());
		assertEquals(3, matrix.columns());
	}

	@Test
	void transposeSwapsRowsAndColumns() {
		Matrix matrix = new Matrix(new float[][] {
				{1.0f, 2.0f, 3.0f},
				{4.0f, 5.0f, 6.0f}
		});

		Matrix transposed = matrix.transpose();

		assertEquals(3, transposed.rows());
		assertEquals(2, transposed.columns());
		assertArrayEquals(new float[] {1.0f, 4.0f}, transposed.values[0]);
		assertArrayEquals(new float[] {2.0f, 5.0f}, transposed.values[1]);
		assertArrayEquals(new float[] {3.0f, 6.0f}, transposed.values[2]);
	}

	@Test
	void multiplyCalculatesMatrixProduct() {
		Matrix left = new Matrix(new float[][] {
				{1.0f, 2.0f, 3.0f},
				{4.0f, 5.0f, 6.0f}
		});
		Matrix right = new Matrix(new float[][] {
				{7.0f, 8.0f},
				{9.0f, 10.0f},
				{11.0f, 12.0f}
		});

		Matrix product = left.multiply(right);

		assertEquals(2, product.rows());
		assertEquals(2, product.columns());
		assertArrayEquals(new float[] {58.0f, 64.0f}, product.values[0]);
		assertArrayEquals(new float[] {139.0f, 154.0f}, product.values[1]);
	}

	@Test
	void multiplyRejectsIncompatibleDimensions() {
		Matrix left = new Matrix(new float[][] {{1.0f, 2.0f}});
		Matrix right = new Matrix(new float[][] {{3.0f, 4.0f}});

		IllegalArgumentException exception = assertThrows(
				IllegalArgumentException.class,
				() -> left.multiply(right));

		assertEquals("Matrix dimensions are incompatible!", exception.getMessage());
	}
}
