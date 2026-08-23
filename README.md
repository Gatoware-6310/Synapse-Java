# Synapse 0.2.0

Synapse is a lightweight Neural Network library in both C and Java, this repo containing the Java version.

## Creating Neural Networks
In Synapse, there are several ways you can instantiate a Neural Network, with varying degrees of simplicity;

```java
// 1. Empty network, adding layers manually
NeuralNetwork manual = new NeuralNetwork();
manual.addLayer(new DenseLayer(784, 128, new ReLU()));
manual.addLayer(new DenseLayer(128, 10, new Softmax()));

// 2. Define all layers immediately
NeuralNetwork defined = new NeuralNetwork(new Layer[] {
	new DenseLayer(784, 128, new ReLU()),
	new DenseLayer(128, 10, new Softmax())
});

// 3. Simple constructor:
// 784 inputs, 128 neurons/layer, 2 hidden layers, 10 outputs
NeuralNetwork simple = new NeuralNetwork(784, 128, 2, 10);
```

## Convolutional Neural Networks
CNNs are mainly useful for images. A convolution layer looks at small parts of an image at a time and learns useful patterns from them, such as edges and shapes.

Synapse uses the same `NeuralNetwork` and `Matrix` classes for CNNs as it does for normal neural networks.

For example, this creates a small CNN for 28x28 grayscale images:

```java
NeuralNetwork cnn = new NeuralNetwork(new Layer[] {
	new Conv2DLayer(28, 28, 1, 32, 3, new ReLU()),
	new MaxPool2DLayer(26, 26, 32, 2),
	new DenseLayer(13 * 13 * 32, 10, new Softmax())
});
```

The first layer:

```java
new Conv2DLayer(28, 28, 1, 32, 3, new ReLU())
```

means:

- `28, 28` - the image width and height
- `1` - the number of color channels (`1` for grayscale, `3` for RGB)
- `32` - the number of filters the layer learns (a filter being a small pattern detector that "slides" across the image)
- `3` - each filter looks at a 3x3 part of the image
- `new ReLU()` - the activation function

With no padding, a 3x3 filter changes a 28x28 image into 26x26 output for each filter.

The next layer:

```java
new MaxPool2DLayer(26, 26, 32, 2)
```

reduces each 26x26 result to 13x13 by taking the largest value from each 2x2 area. This makes the network smaller and faster while keeping the strongest features.

The final `DenseLayer` can use the result directly.

`Conv2DLayer` also supports stride and padding. Stride is how many pixels the filter moves each time, a stride of `1` moves one pixel at a time, while a stride of `2` moves two pixels at a time.

```java
new Conv2DLayer(28, 28, 1, 32, 3, 2, Padding.SAME, new ReLU());
```

- `Padding.VALID` means no padding.
- `Padding.SAME` adds empty space around the image so the output size is preserved when stride is 1.

Images can be loaded into a `Matrix` and saved again with `Images`:

```java
Matrix image = Images.load("image.png", 224, 224);
Images.save(image, 224, 224, 3, "output.png");
```

`Images.load` returns RGB image data with values from `0.0` to `1.0`.

## Datasets
Currently, CSV datasets are supported by Synapse, following this format:

```csv
class (target),data (input),data1,data2...
```

An example of a full CSV dataset to detect a string of even (0) or odd (1) numbers would be:

```csv
0,2,4,6,8
1,3,6,9,12
0,10,12,14,16
1,15,18,21,14
```

It should be noted that column headers are **not** supported.

By default, `CSVLoader` treats the first value in each row as an integer label or scalar target, and the rest as input data. To use multiple target values, pass the amount of target values as the second argument:

```java
Dataset labels = CSVLoader.loadDataset("dataset.csv");
Dataset vectors = CSVLoader.loadDataset("dataset.csv", 10);
```

In the second example, the first 10 values in each row are treated as the **target vector**, and the remaining values are input data.

### Dataset Implementation
The `Dataset` class is a purposefully generic container. The important part is that datasets are mainly comprised of two matrices; `inputs` and `targets`, a target essentially being the expected output or classification for a given input.

Each row in `inputs` corresponds to the same row in `targets`. For example, given the previous dataset, Synapse would interpret it roughly as:
```
inputs:  [2, 4, 6, 8] 
		 [3, 6, 9, 12]
		 [10, 12, 14, 16] 
		 [15, 18, 21, 14]
targets: [0]
		 [1]
		 [0]
		 [1]
```
Additionally, matrices are defined by the `Matrix` class, and are essentially just a wrapper for the type `float[][]` with extra functionality.
Crucially, a matrix's values can only ever be floating point numbers.

## Training models
For datasets containing integer class labels, the simplest form of `fit` uses `SparseCategoricalCrossEntropy` and the `Adam` optimizer by default:

```java
NeuralNetwork network = new NeuralNetwork(new Layer[] {
	new DenseLayer(4, 3, new ReLU()),
	new DenseLayer(3, 2, new Softmax())
});

network.fit(dataset, 10, 0.001f);
```

A batch size can optionally be passed to `fit`:

```java
network.fit(dataset, 10, 0.001f, 64); // 64 being the batch size
network.fit(dataset, 10, 0.001f, new Adam(), 64);
```

For simple fully-connected networks, Synapse can also construct the layers automatically by passing the input size, hidden layer size, amount of hidden layers, and output size:

```java
NeuralNetwork network = new NeuralNetwork(784, 128, 3, 10);
```

This creates three hidden layers with 128 neurons each using `ReLU`, followed by a 10-neuron `Softmax` output layer. Passing `0` hidden layers creates a direct input-to-output `Softmax` layer instead.

A custom loss function and optimizer can both be passed explicitly:

```java
network.fit(dataset, new MeanSquaredError(), 100, 0.001f, new RMSProp());
```

Or, while keeping the default loss function:

```java
network.fit(dataset, 100, 0.01f, new SGD());
```

Optimizers included with Synapse are:

- `Adam` (default)
- `SGD`
- `Momentum`
- `AdaGrad`
- `RMSProp`

Other loss functions that Synapse includes are:

- `BinaryCrossEntropy`
- `CategoricalCrossEntropy`
- `SparseCategoricalCrossEntropy`
- `MeanSquaredError`
- `MeanAbsoluteError`
- `HuberLoss`
- `HingeLoss`
- `KLDivergence`

Other activation functions that Synapse includes are:

- `ELU`
- `GELU`
- `LeakyReLU`
- `ReLU`
- `Sigmoid`
- `SiLU`
- `Softmax`
- `Swish`
- `Tanh`

Logging can be enabled by passing `true` as the final argument. This prints the average loss and classification accuracy after every epoch:

```java
network.fit(dataset, 10, 0.001f, true);
network.fit(dataset, 10, 0.01f, new SGD(), true);
```

The most recent average loss is also available after training with `getLastLoss()`, e.g,

```java
System.out.println("Final loss: " + network.getLastLoss());
```

## Working with a specific layer
`forwardTo` lets you stop at a specific layer and inspect its output:

```java
Matrix output = network.forwardTo(image, 4);
```

The layer number starts at `0`.

`inputGradient` tells you how changing the original input would change that layer's output, without changing the network's weights. This can be used for things like DeepDream.

A basic DeepDream-style update looks like this:

```java
Matrix output = network.forwardTo(image, 4);
Matrix change = network.inputGradient(image, 4, output.copy().multiply(2.0f));
image.add(change.multiply(0.01f));
```

Running that repeatedly changes the image in a direction that makes the selected layer respond more strongly.

## CUDA
CUDA acceleration is optional; CPU remains the default.

```java
Synapse.useDevice(Devices.CUDA);
```

Synapse 0.2.0 supports CUDA **12.0 through 12.9**.

If you directly modify `Matrix.values` after that matrix has been used on CUDA, call `matrix.markDirty()` before using it on CUDA again. For example,

```java
matrix.values[0][0] = 5;
matrix.markDirty();
```

To setup CUDA on Linux/Windows, see the setup [here](CUDA.md).

## Saving models
Saving a model and loading it again is straightforward. For example:

```java
NeuralNetwork network = new NeuralNetwork(new Layer[] {
	new DenseLayer(2, 3, new ReLU()),
	new DenseLayer(3, 1, new Sigmoid())
});

network.save("model.snn");

NeuralNetwork loadedNetwork = NeuralNetwork.load("model.snn");
```

CNN and pooling layers can be saved the same way as dense layers. Older dense-only `.snn` files are still supported.
