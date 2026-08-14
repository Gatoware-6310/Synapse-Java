import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.activation.*;
import xyz.gatoware.synapse.matrix.*;
import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.utils.MatrixUtils;;

public class Main {
	public static void main(String[] args) {
		NeuralNetwork network = new NeuralNetwork(new Layer[] {
				new DenseLayer(2, 5, new ReLU()),
				new DenseLayer(5, 5, new ReLU()),
				new DenseLayer(5, 5, new ReLU()),
				new DenseLayer(5, 3, new Softmax()),
		});

		final Matrix output = network.forward(new Matrix(new float[][] {
				{ 1 },
				{ 2 }
		}));

		MatrixUtils.printMatrix(output);
	}
}
