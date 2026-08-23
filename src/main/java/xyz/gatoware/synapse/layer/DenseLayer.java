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
	private boolean lastCudaResident;
	private Matrix lastBatchInput;
	private Matrix lastBatchWeighted;
	private Matrix lastBatchOutput;

	/** Creates a dense layer with randomly initialized weights and zero biases.
	 * @param inputSize number of input values
	 * @param outputSize number of output neurons
	 * @param activationFunction activation function applied to the layer output
	 */
	public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
		this.weights = new Matrix(outputSize, inputSize);
		this.biases = new Matrix(outputSize, 1);
		float scale = (float) (1.0 / Math.sqrt(inputSize));
		for (int i = 0; i < outputSize; i++)
			for (int j = 0; j < inputSize; j++)
				this.weights.values[i][j] = (float) ((Math.random() * 2 - 1) * scale);
	}

	/** Creates a dense layer from existing parameters.
	 * @param weights weight matrix, with one row per output neuron
	 * @param biases bias column vector, with one value per output neuron
	 * @param activationFunction activation function applied to the layer output
	 */
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
		float[][] product = Synapse.backend().multiply(weights.values, input.values, outputSize, inputSize, batchSize);
		lastCudaResident = false;

		boolean cudaMaterialized = Synapse.backend() instanceof CudaBackend;
		if (cudaMaterialized)
			((CudaBackend) Synapse.backend()).materialize(biases);
		if (batchSize > 1 || cudaMaterialized) {
			lastForwardWasBatch = true;
			lastBatchInput = input;
			lastBatchWeighted = new Matrix(outputSize, batchSize);
			lastBatchOutput = new Matrix(outputSize, batchSize);
			float[] weighted = new float[outputSize];
			float[] activated = new float[outputSize];
			for (int sample = 0; sample < batchSize; sample++) {
				for (int neuron = 0; neuron < outputSize; neuron++) {
					weighted[neuron] = product[neuron][sample] + biases.values[neuron][0];
					lastBatchWeighted.values[neuron][sample] = weighted[neuron];
				}
				activationFunction.apply(weighted, activated);
				for (int neuron = 0; neuron < outputSize; neuron++)
					lastBatchOutput.values[neuron][sample] = activated[neuron];
			}
			return lastBatchOutput;
		}

		lastForwardWasBatch = false;
		lastBatchInput = null;
		lastBatchWeighted = null;
		lastBatchOutput = null;
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

	/** Returns whether this layer can use the resident CUDA ReLU fast path.
	 * @return true when the active backend and activation support resident CUDA execution
	 */
	public boolean canForwardCudaResident() {
		return activationFunction instanceof ReLU
			&& Synapse.backend() instanceof CudaBackend cuda
			&& cuda.supportsResidentRelu();
	}

	/** Runs the dense ReLU forward pass while keeping compatible values resident on CUDA.
	 * @param input input matrix
	 * @return the layer output
	 */
	public Matrix forwardCudaResident(Matrix input) {
		if (!(activationFunction instanceof ReLU))
			throw new IllegalStateException("Resident CUDA forward currently supports ReLU layers only");
		if (!(Synapse.backend() instanceof CudaBackend cuda) || !cuda.supportsResidentRelu())
			return forward(input);
		if (input.rows() != weights.columns() || input.columns() <= 0)
			throw new IllegalArgumentException("Dense layer input must have " + weights.columns() + " rows");
		Matrix result = cuda.denseReluResident(weights, biases, input);
		lastForwardWasBatch = true;
		lastCudaResident = true;
		lastBatchInput = input;
		lastBatchWeighted = null;
		lastBatchOutput = result;
		return result;
	}

	@Override
	public Matrix backwardInput(Matrix outputGradient) {
		if (lastCudaResident)
			throw new IllegalStateException("Input gradients require a standard forward pass");
		materializeParameters();

		if (lastForwardWasBatch) {
			if (lastBatchInput == null || lastBatchWeighted == null || lastBatchOutput == null)
				throw new IllegalStateException("Dense layer must run forward before backward");
			if (outputGradient.rows() != weights.rows() || outputGradient.columns() != lastBatchInput.columns())
				throw new IllegalArgumentException("Dense layer output gradient dimensions do not match the last forward pass");
			return inputGradientFromWeighted(activationBackwardBatch(outputGradient), lastBatchInput.columns());
		}

		if (lastInput == null)
			throw new IllegalStateException("Dense layer must run forward before backward");
		if (outputGradient.rows() != weights.rows() || outputGradient.columns() != 1)
			throw new IllegalArgumentException("Dense layer output gradient must have dimensions " + weights.rows() + " x 1");

		for (int i = 0; i < weights.rows(); i++) {
			weightedInputValues[i] = lastWeightedInput.values[i][0];
			outputValues[i] = lastOutput.values[i][0];
			outputGradientValues[i] = outputGradient.values[i][0];
		}
		activationFunction.backward(weightedInputValues, outputValues, outputGradientValues, weightedGradient);
		Matrix inputGradient = new Matrix(weights.columns(), 1);
		for (int neuron = 0; neuron < weights.rows(); neuron++) {
			float gradient = weightedGradient[neuron];
			for (int input = 0; input < weights.columns(); input++)
				inputGradient.values[input][0] += weights.values[neuron][input] * gradient;
		}
		return inputGradient;
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate) {
		return backward(outputGradient, learningRate, new SGD());
	}

	@Override
	public Matrix backward(Matrix outputGradient, float learningRate, Optimizer optimizer) {
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
		if (optimizer == null)
			throw new IllegalArgumentException("Optimizer cannot be null");

		if (Synapse.backend() instanceof CudaBackend cuda && cuda.supportsResidentRelu()) {
			if (lastCudaResident) {
				if (lastBatchInput == null || lastBatchOutput == null)
					throw new IllegalStateException("Dense layer must run forward before backward");
				if (outputGradient.rows() != weights.rows() || outputGradient.columns() != lastBatchInput.columns())
					throw new IllegalArgumentException("Dense layer output gradient dimensions do not match the last forward pass");
				return cuda.denseReluBackwardUpdate(weights, biases, lastBatchInput, lastBatchOutput,
					outputGradient, optimizer, learningRate);
			}

			if (lastForwardWasBatch) {
				if (lastBatchInput == null || lastBatchWeighted == null || lastBatchOutput == null)
					throw new IllegalStateException("Dense layer must run forward before backward");
				int outputSize = weights.rows();
				int batch = lastBatchInput.columns();
				if (outputGradient.rows() != outputSize || outputGradient.columns() != batch)
					throw new IllegalArgumentException("Dense layer output gradient dimensions do not match the last forward pass");

				Matrix weightedGradients = activationBackwardBatch(outputGradient);
				return cuda.denseBackwardUpdate(weights, biases, lastBatchInput, weightedGradients,
					optimizer, learningRate);
			}
		}

		if (lastForwardWasBatch) {
			if (lastBatchInput == null || lastBatchWeighted == null || lastBatchOutput == null)
				throw new IllegalStateException("Dense layer must run forward before backward");
			int outputSize = weights.rows();
			int inputSize = weights.columns();
			int batch = lastBatchInput.columns();
			if (outputGradient.rows() != outputSize || outputGradient.columns() != batch)
				throw new IllegalArgumentException("Dense layer output gradient dimensions do not match the last forward pass");

			Matrix weightedGradients = activationBackwardBatch(outputGradient);
			Matrix inputGradient = new Matrix(inputSize, batch);
			Matrix batchWeightGradients = new Matrix(outputSize, inputSize);
			Matrix batchBiasGradients = new Matrix(outputSize, 1);
			float scale = 1.0f / batch;

			for (int neuron = 0; neuron < outputSize; neuron++) {
				float biasSum = 0.0f;
				for (int sample = 0; sample < batch; sample++) {
					float gradient = weightedGradients.values[neuron][sample];
					biasSum += gradient;
					for (int input = 0; input < inputSize; input++) {
						inputGradient.values[input][sample] += weights.values[neuron][input] * gradient;
						batchWeightGradients.values[neuron][input] += gradient * lastBatchInput.values[input][sample];
					}
				}
				batchBiasGradients.values[neuron][0] = biasSum * scale;
				for (int input = 0; input < inputSize; input++)
					batchWeightGradients.values[neuron][input] *= scale;
			}

			optimizer.update(weights, batchWeightGradients, learningRate);
			optimizer.update(biases, batchBiasGradients, learningRate);
			weights.markDirty();
			biases.markDirty();
			return inputGradient;
		}

		if (lastInput == null)
			throw new IllegalStateException("Dense layer must run forward before backward");
		if (outputGradient.rows() != weights.rows() || outputGradient.columns() != 1)
			throw new IllegalArgumentException("Dense layer output gradient must have dimensions " + weights.rows() + " x 1");

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

	private Matrix activationBackwardBatch(Matrix outputGradient) {
		int outputSize = weights.rows();
		int batch = lastBatchInput.columns();
		Matrix weightedGradients = new Matrix(outputSize, batch);
		float[] weighted = new float[outputSize];
		float[] output = new float[outputSize];
		float[] upstream = new float[outputSize];
		float[] result = new float[outputSize];
		for (int sample = 0; sample < batch; sample++) {
			for (int neuron = 0; neuron < outputSize; neuron++) {
				weighted[neuron] = lastBatchWeighted.values[neuron][sample];
				output[neuron] = lastBatchOutput.values[neuron][sample];
				upstream[neuron] = outputGradient.values[neuron][sample];
			}
			activationFunction.backward(weighted, output, upstream, result);
			for (int neuron = 0; neuron < outputSize; neuron++)
				weightedGradients.values[neuron][sample] = result[neuron];
		}
		return weightedGradients;
	}

	private Matrix inputGradientFromWeighted(Matrix weightedGradients, int batch) {
		Matrix inputGradient = new Matrix(weights.columns(), batch);
		for (int neuron = 0; neuron < weights.rows(); neuron++)
			for (int sample = 0; sample < batch; sample++) {
				float gradient = weightedGradients.values[neuron][sample];
				for (int input = 0; input < weights.columns(); input++)
					inputGradient.values[input][sample] += weights.values[neuron][input] * gradient;
			}
		return inputGradient;
	}

	/**
	 * CUDA fast path for callers that already have the gradient with respect to
	 * this layer's pre-activation logits. This is especially useful for the
	 * mathematically fused Softmax + cross-entropy derivative (probabilities - one-hot),
	 * avoiding the full Softmax Jacobian on the CPU.
	 *
	 * @param weightedGradient gradient with respect to this layer's pre-activation values
	 * @param learningRate training learning rate
	 * @param optimizer optimizer used to update the weights and biases
	 * @return the gradient with respect to this layer's input
	 */
	public Matrix backwardCudaPreactivated(Matrix weightedGradient, float learningRate, Optimizer optimizer) {
		if (!(Synapse.backend() instanceof CudaBackend cuda))
			throw new IllegalStateException("Pre-activated CUDA backward requires the CUDA backend");
		if (lastBatchInput == null)
			throw new IllegalStateException("Dense layer must run forward before backward");
		if (weightedGradient.rows() != weights.rows() || weightedGradient.columns() != lastBatchInput.columns())
			throw new IllegalArgumentException("Dense layer gradient dimensions do not match the last forward pass");
		if (!Float.isFinite(learningRate) || learningRate <= 0.0f)
			throw new IllegalArgumentException("Learning rate must be positive and finite");
		if (optimizer == null)
			throw new IllegalArgumentException("Optimizer cannot be null");
		if (!cuda.supportsResidentRelu())
			throw new IllegalStateException("CUDA training kernels are unavailable");
		return cuda.denseBackwardUpdate(weights, biases, lastBatchInput, weightedGradient, optimizer, learningRate);
	}

	/** Copies CUDA-resident weights and biases back to their host matrices when needed. */
	public void materializeParameters() {
		if (Synapse.backend() instanceof CudaBackend cuda) {
			cuda.materialize(weights);
			cuda.materialize(biases);
		}
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

	/** Returns this layer's weight matrix.
	 * @return the weight matrix
	 */
	public Matrix getWeights() { return weights; }

	/** Returns this layer's bias vector.
	 * @return the bias vector
	 */
	public Matrix getBiases() { return biases; }

	/** Returns this layer's activation function.
	 * @return the activation function
	 */
	public ActivationFunction getActivationFunction() { return activationFunction; }
}
