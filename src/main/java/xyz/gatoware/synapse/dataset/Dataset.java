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

	/** Returns one row-oriented dataset input as a column vector for a layer. */
	public Matrix getInput(int index) {
		if (index < 0 || index >= size)
			throw new IndexOutOfBoundsException("Dataset input index: " + index);

		Matrix input = new Matrix(inputs.columns(), 1);
		for (int column = 0; column < inputs.columns(); column++)
			input.values[column][0] = inputs.values[index][column];
		return input;
	}

	public void setInputs(Matrix inputs) {
		this.inputs = inputs;
	}

	public Matrix getTargets() {
		return targets;
	}

	/** Returns one row-oriented dataset target. */
	public Matrix getTarget(int index) {
		if (index < 0 || index >= size)
			throw new IndexOutOfBoundsException("Dataset target index: " + index);

		Matrix target = new Matrix(targets.columns(), 1);
		for (int column = 0; column < targets.columns(); column++)
			target.values[column][0] = targets.values[index][column];
		return target;
	}

	public int size() {
		return size;
	}

	public void setTargets(Matrix targets) {
		this.targets = targets;
	}
}
