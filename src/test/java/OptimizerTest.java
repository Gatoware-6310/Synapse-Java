import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.NeuralNetwork;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.dataset.Dataset;
import xyz.gatoware.synapse.layer.DenseLayer;
import xyz.gatoware.synapse.loss.LossFunction;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.AdaGrad;
import xyz.gatoware.synapse.optimizer.Adam;
import xyz.gatoware.synapse.optimizer.Momentum;
import xyz.gatoware.synapse.optimizer.Optimizer;
import xyz.gatoware.synapse.optimizer.RMSProp;
import xyz.gatoware.synapse.optimizer.SGD;

public class OptimizerTest {

	@Test
	void includedOptimizersUpdateParameters() {
		List<Optimizer> optimizers = List.of(
				new SGD(),
				new Momentum(),
				new AdaGrad(),
				new RMSProp(),
				new Adam());

		for (Optimizer optimizer : optimizers) {
			Matrix parameters = new Matrix(new float[][] { { 1.0f, -1.0f } });
			Matrix gradients = new Matrix(new float[][] { { 0.5f, -0.25f } });

			optimizer.update(parameters, gradients, 0.1f);

			assertTrue(Float.isFinite(parameters.values[0][0]));
			assertTrue(Float.isFinite(parameters.values[0][1]));
			assertTrue(parameters.values[0][0] < 1.0f);
			assertTrue(parameters.values[0][1] > -1.0f);
		}
	}

	@Test
	void momentumAccumulatesVelocity() {
		Momentum optimizer = new Momentum(0.9f);
		Matrix parameters = new Matrix(new float[][] { { 1.0f } });
		Matrix gradients = new Matrix(new float[][] { { 2.0f } });

		optimizer.update(parameters, gradients, 0.1f);
		optimizer.update(parameters, gradients, 0.1f);

		assertEquals(0.42f, parameters.values[0][0], 0.0001f);
	}

	@Test
	void fitUsesAdamByDefault() {
		DenseLayer layer = singleLayer();
		NeuralNetwork network = new NeuralNetwork(new DenseLayer[] { layer });

		network.fit(singleSampleDataset(), constantGradientLoss(), 1, 0.1f);

		assertEquals(0.9f, layer.getWeights().values[0][0], 0.0001f);
	}

	@Test
	void fitUsesExplicitOptimizerWhenProvided() {
		DenseLayer layer = singleLayer();
		NeuralNetwork network = new NeuralNetwork(new DenseLayer[] { layer });

		network.fit(singleSampleDataset(), constantGradientLoss(), 1, 0.1f, new SGD());

		assertEquals(0.8f, layer.getWeights().values[0][0], 0.0001f);
	}

	private DenseLayer singleLayer() {
		return new DenseLayer(
				new Matrix(new float[][] { { 1.0f } }),
				new Matrix(new float[][] { { 0.0f } }),
				new ReLU());
	}

	private Dataset singleSampleDataset() {
		return new Dataset(
				new Matrix(new float[][] { { 4.0f } }),
				new Matrix(new float[][] { { 0.0f } }));
	}

	private LossFunction constantGradientLoss() {
		return new LossFunction() {
			@Override
			public float calculate(Matrix predicted, Matrix actual) {
				return 0.0f;
			}

			@Override
			public Matrix gradient(Matrix predicted, Matrix actual) {
				return new Matrix(new float[][] { { 0.5f } });
			}
		};
	}
}
