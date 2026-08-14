package xyz.gatoware.synapse.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xyz.gatoware.synapse.matrix.Matrix;

public class CSVLoader {

	/** Loads a CSV dataset, treating the first value in each row as an integer label or scalar target. */
	public static Dataset loadDataset(final String filename) {
		return loadDataset(filename, 1);
	}

	/** Loads a CSV dataset, treating the first targetSize values in each row as the target vector and the remaining values as input data. */
	public static Dataset loadDataset(final String filename, final int targetSize) {
		if (targetSize <= 0)
			throw new IllegalArgumentException("Target size must be positive");

		ArrayList<float[]> inputRows = new ArrayList<>();
		ArrayList<float[]> targetRows = new ArrayList<>();
		int inputSize = -1;

		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {

			String line;

			while ((line = reader.readLine()) != null) {
				String[] values = line.split(",");
				if (values.length <= targetSize)
					throw new IllegalArgumentException("CSV row must contain targets and input data");

				float[] targets = new float[targetSize];
				for (int i = 0; i < targetSize; i++)
					targets[i] = Float.parseFloat(values[i]);

				float[] inputs = new float[values.length - targetSize];

				for (int i = targetSize; i < values.length; i++)
					inputs[i - targetSize] = Float.parseFloat(values[i]);

				if (inputSize == -1)
					inputSize = inputs.length;
				else if (inputs.length != inputSize)
					throw new IllegalArgumentException("CSV rows must contain the same amount of input data");

				targetRows.add(targets);
				inputRows.add(inputs);
			}

		} catch (IOException e) {
			throw new RuntimeException("Failed to load CSV: " + filename, e);
		}

		float[][] inputs = new float[inputRows.size()][];
		float[][] targets = new float[targetRows.size()][targetSize];

		for (int i = 0; i < inputRows.size(); i++) {
			inputs[i] = inputRows.get(i);
			targets[i] = targetRows.get(i);
		}

		return new Dataset(new Matrix(inputs), new Matrix(targets));
	}
}
