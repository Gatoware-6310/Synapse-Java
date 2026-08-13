import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.dataset.CSVLoader;
import xyz.gatoware.synapse.dataset.Dataset;

public class DatasetTest {
	@Test
	void csvDatasetTest() {
		Dataset dataset = CSVLoader.loadDataset("src/test/java/data.csv");

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
}
