import java.util.Random;

import xyz.gatoware.synapse.Devices;
import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.matrix.Matrix;

/** Simple manual benchmark for comparing Synapse CPU and CUDA backends. */
public final class CudaBenchmark {
	private static final int WARMUP = 10;
	private static final int ITERATIONS = 50;
	private static final Random RANDOM = new Random(1234L);

	private CudaBenchmark() {
	}

	public static void main(String[] args) {
		System.out.println("Synapse CUDA benchmark");
		boolean available = Synapse.isDeviceAvailable(Devices.CUDA);
		System.out.println("CUDA available: " + available);
		if (!available) {
			System.out.println("CUDA backend unavailable.");
			System.out.println("Attempting explicit initialization to show the cause...");
			try {
				Synapse.useDevice(Devices.CUDA);
				System.out.println("Unexpectedly initialized CUDA successfully on the second attempt.");
				Synapse.useDevice(Devices.CPU);
			} catch (Throwable error) {
				printError(error);
			}
			return;
		}

		benchmarkRawMultiply(512);
		benchmarkCachedMultiply(512);
		for (int batch : new int[] {1, 8, 32, 64, 128, 256})
			benchmarkDenseForward(784, 1024, batch);
		Synapse.useDevice(Devices.CPU);
	}

	private static void benchmarkRawMultiply(int size) {
		float[][] aValues = randomValues(size, size);
		float[][] bValues = randomValues(size, size);
		double cpuMs = timeMs(Devices.CPU, () -> {
			Matrix a = new Matrix(copy(aValues));
			Matrix b = new Matrix(copy(bValues));
			a.multiply(b);
		});
		double cudaMs = timeMs(Devices.CUDA, () -> {
			Matrix a = new Matrix(copy(aValues));
			Matrix b = new Matrix(copy(bValues));
			a.multiply(b);
		});
		printResult("Raw " + size + "x" + size + " multiply", cpuMs, cudaMs);
	}

	private static void benchmarkCachedMultiply(int size) {
		float[][] left = randomValues(size, size);
		float[][] right = randomValues(size, size);
		double cpuMs = timeBackendMultiply(Devices.CPU, left, right, size);
		double cudaMs = timeBackendMultiply(Devices.CUDA, left, right, size);
		printResult("Fully cached inputs " + size + "x" + size + " multiply", cpuMs, cudaMs);
	}

	private static double timeBackendMultiply(Devices device, float[][] left, float[][] right, int size) {
		Synapse.useDevice(device);
		for (int i = 0; i < WARMUP; i++)
			Synapse.backend().multiply(left, right, size, size, size);
		long start = System.nanoTime();
		for (int i = 0; i < ITERATIONS; i++)
			Synapse.backend().multiply(left, right, size, size, size);
		return (System.nanoTime() - start) / 1_000_000.0 / ITERATIONS;
	}

	private static void benchmarkDenseForward(int inputSize, int hiddenSize, int batchSize) {
		DenseLayer cpuLayer = new DenseLayer(inputSize, hiddenSize, new ReLU());
		DenseLayer cudaLayer = new DenseLayer(cpuLayer.getWeights().copy(), cpuLayer.getBiases().copy(), new ReLU());
		Matrix input = new Matrix(randomValues(inputSize, batchSize));

		double cpuMs = timeMs(Devices.CPU, () -> cpuLayer.forward(input));
		double cudaMs = timeMs(Devices.CUDA, () -> cudaLayer.forward(input));
		printResult("Dense forward " + inputSize + " -> " + hiddenSize + " batch " + batchSize, cpuMs, cudaMs);
	}

	private static double timeMs(Devices device, Runnable task) {
		Synapse.useDevice(device);
		for (int i = 0; i < WARMUP; i++)
			task.run();
		long start = System.nanoTime();
		for (int i = 0; i < ITERATIONS; i++)
			task.run();
		return (System.nanoTime() - start) / 1_000_000.0 / ITERATIONS;
	}

	private static void printResult(String name, double cpuMs, double cudaMs) {
		System.out.printf("%n%s%n", name);
		System.out.printf("  CPU : %.3f ms%n", cpuMs);
		System.out.printf("  CUDA: %.3f ms%n", cudaMs);
		System.out.printf("  Speedup: %.2fx%n", cpuMs / cudaMs);
	}

	private static void printError(Throwable error) {
		System.out.println("Initialization error chain:");
		Throwable current = error;
		int depth = 0;
		while (current != null) {
			System.out.printf("  [%d] %s: %s%n", depth, current.getClass().getName(), current.getMessage());
			current = current.getCause();
			depth++;
		}
	}

	private static float[][] randomValues(int rows, int columns) {
		float[][] values = new float[rows][columns];
		for (int row = 0; row < rows; row++)
			for (int column = 0; column < columns; column++)
				values[row][column] = RANDOM.nextFloat() * 2.0f - 1.0f;
		return values;
	}

	private static float[][] copy(float[][] values) {
		float[][] result = new float[values.length][];
		for (int row = 0; row < values.length; row++)
			result[row] = values[row].clone();
		return result;
	}
}
