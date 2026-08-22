import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;
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

		assertEquals(cpu.rows(), cuda.rows());
		assertEquals(cpu.columns(), cuda.columns());
		for (int row = 0; row < cpu.rows(); row++)
			for (int column = 0; column < cpu.columns(); column++)
				assertEquals(cpu.values[row][column], cuda.values[row][column], 0.001f);
	}

	private static float[][] copy(float[][] values) {
		float[][] result = new float[values.length][];
		for (int row = 0; row < values.length; row++)
			result[row] = values[row].clone();
		return result;
	}
}
