package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;

public interface Layer {
	Matrix forward(Matrix input);
}
