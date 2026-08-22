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
		benchmarkDenseForward(784, 1024, 10);
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
		Matrix cpuLeft = new Matrix(randomValues(size, size));
		Matrix cpuRight = new Matrix(randomValues(size, size));
		Matrix cudaLeft = new Matrix(copy(cpuLeft.values));
		Matrix cudaRight = new Matrix(copy(cpuRight.values));

		double cpuMs = timeMs(Devices.CPU, () -> {
			Matrix working = cpuLeft.copy();
			working.multiply(cpuRight);
		});

		Synapse.useDevice(Devices.CUDA);
		for (int i = 0; i < WARMUP; i++) {
			Matrix working = cudaLeft.copy();
			working.multiply(cudaRight);
		}
		long start = System.nanoTime();
		for (int i = 0; i < ITERATIONS; i++) {
			Matrix working = cudaLeft.copy();
			working.multiply(cudaRight);
		}
		double cudaMs = (System.nanoTime() - start) / 1_000_000.0 / ITERATIONS;

		printResult("Cached/reused " + size + "x" + size + " multiply", cpuMs, cudaMs);
	}

	private static void benchmarkDenseForward(int inputSize, int hiddenSize, int passes) {
		DenseLayer cpuLayer = new DenseLayer(inputSize, hiddenSize, new ReLU());
		DenseLayer cudaLayer = new DenseLayer(cpuLayer.getWeights().copy(), cpuLayer.getBiases().copy(), new ReLU());
		Matrix input = new Matrix(randomValues(inputSize, 1));

		double cpuMs = timeMs(Devices.CPU, () -> {
			for (int i = 0; i < passes; i++)
				cpuLayer.forward(input);
		}) / passes;

		double cudaMs = timeMs(Devices.CUDA, () -> {
			for (int i = 0; i < passes; i++)
				cudaLayer.forward(input);
		}) / passes;

		printResult("Dense forward " + inputSize + " -> " + hiddenSize, cpuMs, cudaMs);
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
