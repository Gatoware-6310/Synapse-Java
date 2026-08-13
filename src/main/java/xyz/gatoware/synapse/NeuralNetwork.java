package xyz.gatoware.synapse;

import java.util.ArrayList;
import java.util.List;

import xyz.gatoware.synapse.layer.Layer;
import xyz.gatoware.synapse.matrix.Matrix;

public class NeuralNetwork {
	private List<Layer> layers = new ArrayList<>();

	public void addLayer(Layer layer) {
		layers.add(layer);
	}

	public Matrix forward(Matrix input) {
		Matrix output = input;
		for (Layer layer : layers)
			output = layer.forward(output);

		return output;
	}
}
