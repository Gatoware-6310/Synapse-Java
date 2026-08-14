# Synapse

Synapse is a lightweight Neural Network library in both C and Java, this repo containing the Java version.

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
For datasets containing integer class labels, the simplest form of `fit` uses `SparseCategoricalCrossEntropy` by default:

```java
NeuralNetwork network = new NeuralNetwork(new Layer[] {
	new DenseLayer(4, 3, new ReLU()),
	new DenseLayer(3, 2, new Softmax())
});

// `network.fit` takes arguments `Dataset dataset`, `int epochs`, `float learningRate`, and `boolean logging`

network.fit(dataset, 10, 0.01f);
```

```java
network.fit(dataset, new MeanSquaredError(), 100, 0.001f);
```

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
network.fit(dataset, 10, 0.01f, true);
```

The most recent average loss is also available after training with `getLastLoss()`, e.g,

```java
System.out.println("Final loss: " + network.getLastLoss());
```

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
