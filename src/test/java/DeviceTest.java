import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.matrix.Matrix;

public class DeviceTest {
	@AfterEach
	void resetDevice() {
		Synapse.useDevice(Devices.CPU);
	}

	@Test
	void cpuIsDefaultAndAvailable() {
		assertEquals(Devices.CPU, Synapse.getDevice());
		assertTrue(Synapse.isDeviceAvailable(Devices.CPU));
	}

	@Test
	void cpuBackendMultipliesMatrices() {
		Synapse.useDevice(Devices.CPU);
		Matrix left = new Matrix(new float[][] {
			{1.0f, 2.0f, 3.0f},
			{4.0f, 5.0f, 6.0f}
		});
		Matrix right = new Matrix(new float[][] {
			{7.0f, 8.0f},
			{9.0f, 10.0f},
			{11.0f, 12.0f}
		});

		left.multiply(right);

		assertEquals(58.0f, left.values[0][0], 0.0001f);
		assertEquals(64.0f, left.values[0][1], 0.0001f);
		assertEquals(139.0f, left.values[1][0], 0.0001f);
		assertEquals(154.0f, left.values[1][1], 0.0001f);
	}

	@Test
	void cudaBackendMatchesCpuWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		float[][] leftValues = {
			{1.0f, 2.0f, 3.0f},
			{4.0f, 5.0f, 6.0f}
		};
		float[][] rightValues = {
			{7.0f, 8.0f},
			{9.0f, 10.0f},
			{11.0f, 12.0f}
		};

		Synapse.useDevice(Devices.CPU);
		Matrix cpu = new Matrix(copy(leftValues)).multiply(new Matrix(copy(rightValues)));
		Synapse.useDevice(Devices.CUDA);
		Matrix cuda = new Matrix(copy(leftValues)).multiply(new Matrix(copy(rightValues)));
		assertMatrixEquals(cpu, cuda);
	}

	@Test
	void cudaCacheRefreshesAfterExplicitHostMutationWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		Synapse.useDevice(Devices.CUDA);

		Matrix right = new Matrix(new float[][] {
			{1.0f, 0.0f},
			{0.0f, 1.0f}
		});
		Matrix first = new Matrix(new float[][] {{2.0f, 3.0f}}).multiply(right);
		assertEquals(2.0f, first.values[0][0], 0.001f);
		assertEquals(3.0f, first.values[0][1], 0.001f);

		right.values[0][0] = 4.0f;
		right.markDirty();
		Matrix second = new Matrix(new float[][] {{2.0f, 3.0f}}).multiply(right);
		assertEquals(8.0f, second.values[0][0], 0.001f);
		assertEquals(3.0f, second.values[0][1], 0.001f);
	}

	@Test
	void cudaCachedResultCanFeedAnotherMultiplyWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		Synapse.useDevice(Devices.CUDA);
		Matrix value = new Matrix(new float[][] {
			{1.0f, 2.0f},
			{3.0f, 4.0f}
		});
		Matrix identity = new Matrix(new float[][] {
			{1.0f, 0.0f},
			{0.0f, 1.0f}
		});
		value.multiply(identity).multiply(identity);
		assertEquals(1.0f, value.values[0][0], 0.001f);
		assertEquals(2.0f, value.values[0][1], 0.001f);
		assertEquals(3.0f, value.values[1][0], 0.001f);
		assertEquals(4.0f, value.values[1][1], 0.001f);
	}

	@Test
	void batchedDenseCudaMatchesCpuWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		DenseLayer cpuLayer = new DenseLayer(3, 2, new ReLU());
		DenseLayer cudaLayer = new DenseLayer(cpuLayer.getWeights().copy(), cpuLayer.getBiases().copy(), new ReLU());
		Matrix input = new Matrix(new float[][] {
			{1.0f, 2.0f, 3.0f, 4.0f},
			{0.5f, 1.5f, 2.5f, 3.5f},
			{-1.0f, -2.0f, -3.0f, -4.0f}
		});

		Synapse.useDevice(Devices.CPU);
		Matrix cpu = cpuLayer.forward(input);
		Synapse.useDevice(Devices.CUDA);
		Matrix cuda = cudaLayer.forward(input);
		assertMatrixEquals(cpu, cuda);
	}

	private static void assertMatrixEquals(Matrix expected, Matrix actual) {
		assertEquals(expected.rows(), actual.rows());
		assertEquals(expected.columns(), actual.columns());
		for (int row = 0; row < expected.rows(); row++)
			for (int column = 0; column < expected.columns(); column++)
				assertEquals(expected.values[row][column], actual.values[row][column], 0.001f);
	}

	private static float[][] copy(float[][] values) {
		float[][] result = new float[values.length][];
		for (int row = 0; row < values.length; row++)
			result[row] = values[row].clone();
		return result;
	}
}
