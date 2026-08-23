import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.backend.CudaCnnOps;
import xyz.gatoware.synapse.layer.AvgPool2DLayer;
import xyz.gatoware.synapse.layer.Conv2DLayer;
import xyz.gatoware.synapse.layer.MaxPool2DLayer;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.SGD;

public class CudaCnnTest {
	@AfterEach
	void resetDevice() {
		Synapse.useDevice(Devices.CPU);
	}

	@Test
	void cudaConvAndPoolingMatchCpuWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		Synapse.useDevice(Devices.CUDA);
		Assumptions.assumeTrue(CudaCnnOps.isAvailable());

		Conv2DLayer cudaConv = new Conv2DLayer(4, 4, 1, 2, 2, new ReLU());
		for (int filter = 0; filter < cudaConv.getFilters(); filter++) {
			for (int value = 0; value < cudaConv.getKernels().columns(); value++)
				cudaConv.getKernels().values[filter][value] = (filter + 1) * (value + 1) * 0.05f;
			cudaConv.getBiases().values[filter][0] = filter * 0.1f;
		}
		Matrix input = new Matrix(new float[][] {
			{0.0f, 1.0f}, {0.1f, 0.9f}, {0.2f, 0.8f}, {0.3f, 0.7f},
			{0.4f, 0.6f}, {0.5f, 0.5f}, {0.6f, 0.4f}, {0.7f, 0.3f},
			{0.8f, 0.2f}, {0.9f, 0.1f}, {1.0f, 0.0f}, {0.9f, 0.1f},
			{0.8f, 0.2f}, {0.7f, 0.3f}, {0.6f, 0.4f}, {0.5f, 0.5f}
		});
		Matrix cudaConvOutput = cudaConv.forward(input);
		Matrix cudaMaxOutput = new MaxPool2DLayer(3, 3, 2, 2, 1).forward(cudaConvOutput);
		Matrix cudaAvgOutput = new AvgPool2DLayer(3, 3, 2, 2, 1).forward(cudaConvOutput);

		float[][] kernelValues = copy(cudaConv.getKernels().values);
		float[][] biasValues = copy(cudaConv.getBiases().values);
		Synapse.useDevice(Devices.CPU);
		Conv2DLayer cpuConv = new Conv2DLayer(4, 4, 1, 2, 2, new ReLU());
		copyInto(kernelValues, cpuConv.getKernels().values);
		copyInto(biasValues, cpuConv.getBiases().values);
		Matrix cpuConvOutput = cpuConv.forward(input);
		Matrix cpuMaxOutput = new MaxPool2DLayer(3, 3, 2, 2, 1).forward(cpuConvOutput);
		Matrix cpuAvgOutput = new AvgPool2DLayer(3, 3, 2, 2, 1).forward(cpuConvOutput);

		assertMatrixEquals(cpuConvOutput, cudaConvOutput);
		assertMatrixEquals(cpuMaxOutput, cudaMaxOutput);
		assertMatrixEquals(cpuAvgOutput, cudaAvgOutput);
	}

	@Test
	void cudaConvBackwardUpdatesKernelsWhenAvailable() {
		Assumptions.assumeTrue(Synapse.isDeviceAvailable(Devices.CUDA));
		Synapse.useDevice(Devices.CUDA);
		Assumptions.assumeTrue(CudaCnnOps.isAvailable());
		Conv2DLayer conv = new Conv2DLayer(4, 4, 1, 1, 2, new ReLU());
		Matrix input = new Matrix(new float[][] {
			{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1},
			{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}
		});
		Matrix output = conv.forward(input);
		Matrix gradient = new Matrix(output.rows(), output.columns());
		for (int row = 0; row < gradient.rows(); row++)
			for (int column = 0; column < gradient.columns(); column++)
				gradient.values[row][column] = 1.0f;
		float before = conv.getKernels().values[0][0];
		conv.backward(gradient, 0.01f, new SGD());
		assertTrue(Float.isFinite(conv.getKernels().values[0][0]));
		assertNotEquals(before, conv.getKernels().values[0][0]);
	}

	private static void assertMatrixEquals(Matrix expected, Matrix actual) {
		assertEquals(expected.rows(), actual.rows());
		assertEquals(expected.columns(), actual.columns());
		for (int row = 0; row < expected.rows(); row++)
			for (int column = 0; column < expected.columns(); column++)
				assertEquals(expected.values[row][column], actual.values[row][column], 0.002f);
	}

	private static float[][] copy(float[][] values) {
		float[][] result = new float[values.length][];
		for (int row = 0; row < values.length; row++) result[row] = values[row].clone();
		return result;
	}

	private static void copyInto(float[][] source, float[][] target) {
		for (int row = 0; row < source.length; row++)
			System.arraycopy(source[row], 0, target[row], 0, source[row].length);
	}
}
