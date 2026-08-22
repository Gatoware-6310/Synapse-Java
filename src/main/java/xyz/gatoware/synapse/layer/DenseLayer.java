package xyz.gatoware.synapse.layer;

import java.util.Arrays;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.activation.ActivationFunction;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.backend.CudaBackend;
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
	private boolean lastForwardWasBatch;

	public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
		this.weights = new Matrix(outputSize, inputSize);
		this.biases = new Matrix(outputSize, 1);
		float scale = (float) (1.0 / Math.sqrt(inputSize));
		for (int i = 0; i < outputSize; i++)
			for (int j = 0; j < inputSize; j++)
				this.weights.values[i][j] = (float) ((Math.random() * 2 - 1) * scale);
	}

	public DenseLayer(Matrix weights, Matrix biases, ActivationFunction activationFunction) {
		if (weights.rows() != biases.rows() || biases.columns() != 1)
			throw new IllegalArgumentException("Dense layer biases must have dimensions " + weights.rows() + " x 1");
		this.weights = weights;
		this.biases = biases;
		this.activationFunction = activationFunction;
	}

	@Override
	public Matrix forward(Matrix input) {
		if (input.rows() != weights.columns() || input.columns() <= 0)
			throw new IllegalArgumentException("Dense layer input must have " + weights.columns() + " rows");

		int inputSize = weights.columns();
		int outputSize = weights.rows();
		int batchSize = input.columns();
		float[][] product = Synapse.backend().multiply(weights.values, input.values,
			outputSize, inputSize, batchSize);

		if (batchSize > 1) {
			lastForwardWasBatch = true;
			Matrix result = new Matrix(outputSize, batchSize);
			float[] weighted = new float[outputSize];
			float[] activated = new float[outputSize];
			for (int sample = 0; sample < batchSize; sample++) {
				for (int neuron = 0; neuron < outputSize; neuron++)
					weighted[neuron] = product[neuron][sample] + biases.values[neuron][0];
				activationFunction.apply(weighted, activated);
				for (int neuron = 0; neuron < outputSize; neuron++)
					result.values[neuron][sample] = activated[neuron];
			}
			return result;
		}

		lastForwardWasBatch = false;
		ensureTrainingBuffers(inputSize, outputSize);
		lastInput.markDirty();
		lastWeightedInput.markDirty();
		lastOutput.markDirty();
		for (int i = 0; i < inputSize; i++)
			lastInput.values[i][0] = input.values[i][0];
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

	/** Returns true when this layer can use the no-readback CUDA inference path. */
	public boolean canForwardCudaResident() {
		return activationFunction instanceof ReLU
			&& Synapse.backend() instanceof CudaBackend cuda
			&& cuda.supportsResidentRelu();
	}

	/** Runs this ReLU dense layer without materializing its result on the CPU.
	 * Intended for internal network inference chains only.
	 */
	public Matrix forwardCudaResident(Matrix input) {
		if (!(activationFunction instanceof ReLU))
			throw new IllegalStateException("Resident CUDA forward currently supports ReLU layers only");
		if (!(Synapse.backend() instanceof CudaBackend cuda) || !cuda.supportsResidentRelu())
			return forward(input);
		if (input.rows() != weights.columns() || input.columns() <= 0)
			throw new IllegalArgumentException("Dense layer input must have " + weights.columns() + " rows");
		lastForwardWasBatch = input.columns() > 1;
		return cuda.denseReluResident(weights, biases, input);
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate) {
		return backward(outputGradient, learningRate, new SGD());
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		if (lastForwardWasBatch)
			throw new IllegalStateException("Backward after a batched forward is not supported yet");
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
		weights.markDirty();
		biases.markDirty();
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

	public Matrix getWeights() {
		return weights;
	}

	public Matrix getBiases() {
		return biases;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}
}
