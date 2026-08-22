package xyz.gatoware.synapse.layer;

import java.util.Arrays;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.Optimizer;
import xyz.gatoware.synapse.optimizer.SGD;

/** A fully connected neural network layer. */
public class DenseLayer implements Layer {

	private Matrix weights;
	private Matrix biases;
	private ActivationFunction activationFunction;
	private Matrix lastInput;
	private Matrix lastWeightedInput;
	private Matrix lastOutput;
	private Matrix weightGradients;
	private Matrix biasGradients;
	private float[] weightedInputValues;
	private float[] outputValues;
	private float[] outputGradientValues;
	private float[] weightedGradient;
	private float[] inputGradientValues;

	/** Creates a dense layer with randomly initialized weights.
	 * @param inputSize the amount of inputs
	 * @param outputSize the amount of outputs
	 * @param activationFunction the activation function
	 */
	public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
		this.weights = new Matrix(outputSize, inputSize);
		this.biases = new Matrix(outputSize, 1);

		float scale = (float) (1.0 / Math.sqrt(inputSize));
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				this.weights.values[i][j] = (float) ((Math.random() * 2 - 1) * scale);
			}
		}
	}

	/** Creates a dense layer from weights, biases, and an activation function.
	 * @param weights the layer weights
	 * @param biases the layer biases
	 * @param activationFunction the activation function
	 */
	public DenseLayer(Matrix weights, Matrix biases, ActivationFunction activationFunction) {
		if (weights.rows() != biases.rows() || biases.columns() != 1) {
			throw new IllegalArgumentException("Dense layer biases must have dimensions " + weights.rows() + " x 1");
		}

		this.weights = weights;
		this.biases = biases;
		this.activationFunction = activationFunction;
	}

	@Override
	/** Runs the input through the dense layer.
	 * @param input the layer input
	 * @return the layer output
	 */
	public Matrix forward(Matrix input) {
		if (input.rows() != weights.columns() || input.columns() != 1) {
			throw new IllegalArgumentException("Dense layer input must have dimensions " + weights.columns() + " x 1");
		}

		int inputSize = weights.columns();
		int outputSize = weights.rows();
		ensureTrainingBuffers(inputSize, outputSize);

		for (int i = 0; i < inputSize; i++)
			lastInput.values[i][0] = input.values[i][0];

		float[][] product = Synapse.backend().multiply(weights.values, input.values, outputSize, inputSize, 1);
		for (int neuron = 0; neuron < outputSize; neuron++) {
			float sum = product[neuron][0] + biases.values[neuron][0];
			lastWeightedInput.values[neuron][0] = sum;
			weightedInputValues[neuron] = sum;
		}

		activationFunction.apply(weightedInputValues, outputValues);
		for (int neuron = 0; neuron < outputSize; neuron++)
			lastOutput.values[neuron][0] = outputValues[neuron];
		return lastOutput.copy();
	}

	@Override
	/** Updates the layer using backpropagation and stochastic gradient descent, then returns the input gradient.
	 * @param outputGradient the gradient at the layer output
	 * @param learningRate the training learning rate
	 * @return the gradient at the layer input
	 */
	public Matrix backward(Matrix outputGradient, float learningRate) {
		return backward(outputGradient, learningRate, new SGD());
	}

	@Override
	/** Updates the layer using backpropagation and the given optimizer, then returns the input gradient.
	 * @param outputGradient the gradient at the layer output
	 * @param learningRate the training learning rate
	 * @param optimizer the optimizer used to update weights and biases
	 * @return the gradient at the layer input
	 */
	public Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		if (lastInput == null)
			throw new IllegalStateException("Dense layer must run forward before backward");
		if (outputGradient.rows() != weights.rows() || outputGradient.columns() != 1)
			throw new IllegalArgumentException("Dense layer output gradient must have dimensions " + weights.rows() + " x 1");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
		if (optimizer == null)
			throw new IllegalArgumentException("Optimizer cannot be null");

		for (int i = 0; i < weights.rows(); i++) {
			weightedInputValues[i] = lastWeightedInput.values[i][0];
			outputValues[i] = lastOutput.values[i][0];
			outputGradientValues[i] = outputGradient.values[i][0];
		}
		activationFunction.backward(weightedInputValues, outputValues, outputGradientValues, weightedGradient);

		Matrix inputGradient = new Matrix(weights.columns(), 1);
		Arrays.fill(inputGradientValues, 0.0f);
		for (int neuron = 0; neuron < weights.rows(); neuron++) {
			float gradient = weightedGradient[neuron];
			float[] neuronWeights = weights.values[neuron];
			for (int input = 0; input < weights.columns(); input++) {
				inputGradientValues[input] += neuronWeights[input] * gradient;
				weightGradients.values[neuron][input] = gradient * lastInput.values[input][0];
			}
			biasGradients.values[neuron][0] = gradient;
		}
		for (int input = 0; input < weights.columns(); input++)
			inputGradient.values[input][0] = inputGradientValues[input];

		optimizer.update(weights, weightGradients, learningRate);
		optimizer.update(biases, biasGradients, learningRate);
		return inputGradient;
	}

	private void ensureTrainingBuffers(int inputSize, int outputSize) {
		if (lastInput == null || lastInput.rows() != inputSize) {
			lastInput = new Matrix(inputSize, 1);
			inputGradientValues = new float[inputSize];
		}
		if (lastWeightedInput == null || lastWeightedInput.rows() != outputSize) {
			lastWeightedInput = new Matrix(outputSize, 1);
			lastOutput = new Matrix(outputSize, 1);
			weightGradients = new Matrix(outputSize, inputSize);
			biasGradients = new Matrix(outputSize, 1);
			weightedInputValues = new float[outputSize];
			outputValues = new float[outputSize];
			outputGradientValues = new float[outputSize];
			weightedGradient = new float[outputSize];
		}
	}

	/** Returns the layer weights.
	 * @return the layer weights
	 */
	public Matrix getWeights() {
		return weights;
	}

	/** Returns the layer biases.
	 * @return the layer biases
	 */
	public Matrix getBiases() {
		return biases;
	}

	/** Returns the layer activation function.
	 * @return the activation function
	 */
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

}
