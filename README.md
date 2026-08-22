# Synapse 0.1.2

Synapse is a lightweight neural network library in both C and Java; this repository contains the Java version.

## Creating Neural Networks

```java
NeuralNetwork manual = new NeuralNetwork();
manual.addLayer(new DenseLayer(784, 128, new ReLU()));
manual.addLayer(new DenseLayer(128, 10, new Softmax()));

NeuralNetwork defined = new NeuralNetwork(new Layer[] {
	new DenseLayer(784, 128, new ReLU()),
	new DenseLayer(128, 10, new Softmax())
});

NeuralNetwork simple = new NeuralNetwork(784, 128, 2, 10);
```

## Datasets

CSV datasets use the first value in each row as the target by default, followed by input values:

```csv
0,2,4,6,8
1,3,6,9,12
```

```java
Dataset labels = CSVLoader.loadDataset("dataset.csv");
Dataset vectors = CSVLoader.loadDataset("dataset.csv", 10);
```

Column headers are not supported.

## Training

The simplest `fit` uses `SparseCategoricalCrossEntropy` and `Adam` by default:

```java
network.fit(dataset, 10, 0.001f);
```

You can optionally choose the batch size:

```java
network.fit(dataset, 10, 0.001f, 64);
network.fit(dataset, 10, 0.001f, new Adam(), 64);
```

Custom loss functions and optimizers are also supported:

```java
network.fit(dataset, new MeanSquaredError(), 100, 0.001f, new RMSProp());
```

Included optimizers:

- `Adam` (default)
- `SGD`
- `Momentum`
- `AdaGrad`
- `RMSProp`

Included loss functions:

- `BinaryCrossEntropy`
- `CategoricalCrossEntropy`
- `SparseCategoricalCrossEntropy`
- `MeanSquaredError`
- `MeanAbsoluteError`
- `HuberLoss`
- `HingeLoss`
- `KLDivergence`

Included activation functions:

- `ELU`
- `GELU`
- `LeakyReLU`
- `ReLU`
- `Sigmoid`
- `SiLU`
- `Softmax`
- `Swish`
- `Tanh`

Logging can be enabled with the boolean overload:

```java
network.fit(dataset, 10, 0.001f, true);
```

The latest average loss is available through `getLastLoss()`.

## CUDA

CPU is the default. To use an NVIDIA GPU:

```java
Synapse.useDevice(Devices.CUDA);
```

Synapse 0.1.2 supports **CUDA 12.0 through CUDA 12.9**. CUDA 13 is not supported by the current JCuda 12.x backend.

CUDA accelerates matrix multiplication, dense inference, mini-batched training, backpropagation, and the included optimizers. The standard `Softmax + SparseCategoricalCrossEntropy` training path is fused on the GPU.

For direct writes to a matrix that has already been used by CUDA, mark it dirty before reusing it:

```java
matrix.values[0][0] = 123.0f;
matrix.markDirty();
```

You can check availability before switching devices:

```java
if (Synapse.isDeviceAvailable(Devices.CUDA)) {
	Synapse.useDevice(Devices.CUDA);
}
```

## Saving Models

```java
network.save("model.snn");
NeuralNetwork loadedNetwork = NeuralNetwork.load("model.snn");
```
