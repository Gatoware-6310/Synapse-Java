package xyz.gatoware.synapse.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import xyz.gatoware.synapse.matrix.Matrix;

public class CSVLoader {

	public static Dataset loadDataset(final String filename) {
		ArrayList<float[]> inputRows = new ArrayList<>();
		ArrayList<Float> targetRows = new ArrayList<>();

		try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {

			String line;

			while ((line = reader.readLine()) != null) {
				String[] values = line.split(",");

				float target = Float.parseFloat(values[0]);

				float[] inputs = new float[values.length - 1];

				for (int i = 1; i < values.length; i++) {
					inputs[i - 1] = Float.parseFloat(values[i]);
				}

				targetRows.add(target);
				inputRows.add(inputs);
			}

		} catch (IOException e) {
			throw new RuntimeException("Failed to load CSV: " + filename, e);
		}

		float[][] inputs = new float[inputRows.size()][];
		float[][] targets = new float[targetRows.size()][1];

		for (int i = 0; i < inputRows.size(); i++) {
			inputs[i] = inputRows.get(i);
			targets[i][0] = targetRows.get(i);
		}

		return new Dataset(new Matrix(inputs), new Matrix(targets));
	}
}
