package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.matrix.Matrix;

public class DenseLayer implements Layer {

	private Matrix weights;
	private Matrix biases;
	private ActivationFunction activationFunction;
	private Matrix lastInput;
	private Matrix lastWeightedInput;
	private Matrix lastOutput;

	/** Creates a dense layer with randomly initialized weights. */
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

	/** Creates a dense layer from weights, biases, and an activation function. */
	public DenseLayer(Matrix weights, Matrix biases, ActivationFunction activationFunction) {
		if (weights.rows() != biases.rows() || biases.columns() != 1) {
			throw new IllegalArgumentException("Dense layer biases must have dimensions " + weights.rows() + " x 1");
		}

		this.weights = weights;
		this.biases = biases;
		this.activationFunction = activationFunction;
	}

	@Override
	/** Runs the input through the dense layer. */
	public Matrix forward(Matrix input) {
		if (input.rows() != weights.columns() || input.columns() != 1) {
			throw new IllegalArgumentException("Dense layer input must have dimensions " + weights.columns() + " x 1");
		}

		lastInput = input.copy();
		lastWeightedInput = weights.copy().multiply(input).add(biases);
		lastOutput = lastWeightedInput.copy().apply(activationFunction);
		return lastOutput.copy();
	}

	@Override
	/** Updates the layer using backpropagation and returns the input gradient. */
	public Matrix backward(Matrix outputGradient, float learningRate) {
		if (lastInput == null)
			throw new IllegalStateException("Dense layer must run forward before backward");
		if (outputGradient.rows() != weights.rows() || outputGradient.columns() != 1)
			throw new IllegalArgumentException("Dense layer output gradient must have dimensions " + weights.rows() + " x 1");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");

		float[] weightedInput = new float[weights.rows()];
		float[] output = new float[weights.rows()];
		float[] gradient = new float[weights.rows()];
		for (int i = 0; i < weights.rows(); i++) {
			weightedInput[i] = lastWeightedInput.values[i][0];
			output[i] = lastOutput.values[i][0];
			gradient[i] = outputGradient.values[i][0];
		}
		float[] weightedGradient = activationFunction.backward(weightedInput, output, gradient);

		Matrix inputGradient = new Matrix(weights.columns(), 1);
		for (int input = 0; input < weights.columns(); input++) {
			for (int neuron = 0; neuron < weights.rows(); neuron++)
				inputGradient.values[input][0] += weights.values[neuron][input] * weightedGradient[neuron];
		}

		for (int neuron = 0; neuron < weights.rows(); neuron++) {
			for (int input = 0; input < weights.columns(); input++)
				weights.values[neuron][input] -= learningRate * weightedGradient[neuron] * lastInput.values[input][0];
			biases.values[neuron][0] -= learningRate * weightedGradient[neuron];
		}

		return inputGradient;
	}

	/** Returns the layer weights. */
	public Matrix getWeights() {
		return weights;
	}

	/** Returns the layer biases. */
	public Matrix getBiases() {
		return biases;
	}

	/** Returns the layer activation function. */
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

}
