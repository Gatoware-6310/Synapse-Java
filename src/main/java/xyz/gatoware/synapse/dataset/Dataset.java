package xyz.gatoware.synapse.dataset;

import xyz.gatoware.synapse.matrix.Matrix;

public class Dataset {
	private Matrix inputs;
	private Matrix targets;
	private int size;

	public Dataset(Matrix inputs, Matrix targets) {
		if (inputs.rows() != targets.rows())
			throw new IllegalArgumentException("Inputs and targets must have the same amount of rows!");

		this.inputs = inputs;
		this.targets = targets;
		this.size = inputs.rows(); /* Either inputs.rows or targets.rows works - we verified they're the same. */
	}

	public Matrix getInputs() {
		return inputs;
	}

	public void setInputs(Matrix inputs) {
		this.inputs = inputs;
	}

	public Matrix getTargets() {
		return targets;
	}

	public int size() {
		return size;
	}

	public void setTargets(Matrix targets) {
		this.targets = targets;
	}
}
