import xyz.gatoware.synapse.dataset.CSVLoader;
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.utils.MatrixUtils;

public class Main {
	public static void main(String[] args) {
		Dataset d = CSVLoader.loadDataset("src/test/java/resources/data.csv");
		MatrixUtils.printMatrix(d.getInputs());
	}
}
