import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.dataset.CSVLoader;
import xyz.gatoware.synapse.dataset.Dataset;

public class DatasetTest {
	@Test
	void csvDatasetTest() {
		Dataset dataset = CSVLoader.loadDataset("src/test/java/resources/data.csv");

		assertEquals(2, dataset.size());
		assertEquals(2, dataset.getInputs().rows());
		assertEquals(14, dataset.getInputs().columns());
		assertEquals(2, dataset.getTargets().rows());
		assertEquals(1, dataset.getTargets().columns());

		assertArrayEquals(new float[] {
				1, 2, 3, 4, 5, 6, 7,
				8, 9, 10, 11, 12, 13, 14
		}, dataset.getInputs().values[0]);
		assertArrayEquals(new float[] {
				2, 3, 4, 5, 6, 7, 8,
				9, 10, 11, 12, 13, 14, 15
		}, dataset.getInputs().values[1]);

		assertArrayEquals(new float[] { 0 }, dataset.getTargets().values[0]);
		assertArrayEquals(new float[] { 1 }, dataset.getTargets().values[1]);
	}

	@Test
	void sampleAccessorsReturnLayerColumnVectors() {
		Dataset dataset = CSVLoader.loadDataset("src/test/java/resources/data.csv");

		assertEquals(14, dataset.getInput(0).rows());
		assertEquals(1, dataset.getInput(0).columns());
		for (int i = 0; i < 14; i++)
			assertEquals(i + 1, dataset.getInput(0).values[i][0]);
		assertEquals(0, dataset.getTarget(0).values[0][0]);
	}

	@Test
	void mnistCsvLoadsNormalizedInputsAndIntegerLabels() {
		Dataset dataset = CSVLoader.loadDataset("src/test/java/resources/MNIST.csv");

		assertEquals(10_000, dataset.size());
		assertEquals(784, dataset.getInputs().columns());
		assertEquals(1, dataset.getTargets().columns());

		for (int sample = 0; sample < dataset.size(); sample++) {
			float label = dataset.getTargets().values[sample][0];
			assertEquals(Math.round(label), label);
			assertTrue(label >= 0 && label <= 9);

			for (float pixel : dataset.getInputs().values[sample])
				assertTrue(pixel >= 0.0f && pixel <= 1.0f);
		}
	}

	@Test
	void csvLoaderCanReadMultipleTargetValues() {
		Dataset dataset = CSVLoader.loadDataset("src/test/java/resources/data.csv", 2);

		assertEquals(2, dataset.getTargets().rows());
		assertEquals(2, dataset.getTargets().columns());
		assertEquals(13, dataset.getInputs().columns());
		assertArrayEquals(new float[] { 0, 1 }, dataset.getTargets().values[0]);
		assertArrayEquals(new float[] { 1, 2 }, dataset.getTargets().values[1]);
	}
}
