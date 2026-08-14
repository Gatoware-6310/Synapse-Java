import java.io.IOException;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.Softmax;
import xyz.gatoware.synapse.dataset.CSVLoader;
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.layer.Layer;

public class Main {

	private static final String DATASET = "src/test/java/resources/MNIST.csv";
	private static final String MODEL = "MNIST_example.snn";
	private static final int EPOCHS = 10;
	private static final float LEARNING_RATE = 0.01f;

	public static void main(String[] args) throws IOException {
		System.out.println("Loading MNIST dataset...");
		Dataset dataset = CSVLoader.loadDataset(DATASET);

		NeuralNetwork network = new NeuralNetwork(new Layer[] {
				new DenseLayer(784, 128, new ReLU()),
				new DenseLayer(128, 64, new ReLU()),
				new DenseLayer(64, 10, new Softmax())
		});

		System.out.println("Training for " + EPOCHS + " epochs...");
		network.fit(dataset, EPOCHS, LEARNING_RATE, true);

		network.save(MODEL);
		System.out.printf("Saved %s with final accuracy: %.2f%%%n", MODEL, network.accuracy(dataset) * 100.0f);
	}
}
