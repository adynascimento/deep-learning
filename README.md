# Deep Feedforward Neural Network (DNN) Code from Scratch Using Golang

A comprehensive deep learning library written in Go from scratch, featuring support for Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Hyperparameter Optimization, and Natural Language Processing (NLP).

## 📋 Features

### 🧠 Artificial Neural Networks (ANN)
- Fully connected customizable architecture
- Multiple activation functions: Tanh, ReLU, Sigmoid, ELU
- Three training modes:
  - **Regression**: Continuous value prediction
  - **Multiclass Classification**: Multi-class classification problems
  - **Multilabel Classification**: Multiple labels per sample
- Optimizers: Adam, Gradient Descent
- L2 Regularization
- Batch training with customizable batch size

### 🖼️ Convolutional Neural Networks (CNN)
- 2D convolutional layers with customizable filters
- **ConvolveFFT2D**: Fast Fourier Transform-based convolution for optimized performance
- Max Pooling layers
- Flatten layer for transition to dense layers
- Support for multiclass and multilabel classification modes
- Multi-channel image support (grayscale, RGB, custom channels)
- Network architecture visualization
- Model export and loading

### 🔍 Hyperparameter Optimization
- **Random Search**: Random search over parameter space
- **Bayesian Optimization**: Advanced Bayesian optimization powered by Goptuna
- **Sampling Functions**: Intelligent hyperparameter exploration with log-uniform and uniform distributions
- Testing different network architectures
- Model performance comparison
- JSON results export

### 📊 Mathematical Utilities (NGO)
- Matrix operations with Gonum
- **StandardScaler**: Feature standardization with Fit/Transform/FitTransform/InverseTransform
- **PCA (Principal Component Analysis)**: Dimensionality reduction with explained variance tracking
- **Sampling Functions**: `SuggestInt`, `SuggestFloat`, `SuggestLogFloat` for hyperparameter exploration
- Dense matrix manipulation
- Linear functions (linspace, interpolation)
- Log-uniform and uniform random sampling

### 💬 Natural Language Processing (NLP)
- **Bag of Words (BoW)**: Text vectorization
- **TF-IDF**: Term frequency-inverse document frequency
- Text preprocessing
- Feature extraction for ML models

### 📈 I/O Utilities
- CSV data loading
- Model saving and loading in JSON format
- Large-scale dataset support
- Model summary generation

## 🎯 Usage Examples

### 1. Regression (Sine Function Prediction)

```go
// examples/regression/regression.go
package main

import (
	"math"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data: sine function
	applySin := func(_, _ int, v float64) float64 { 
		return math.Sin(15. * v) 
	}
	xTrain := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	yTrain := ngo.Apply(applySin, xTrain)

	// create model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{1, 40, 20, 10, 1},
		Activation:  network.TanhActivation,
		Mode:        network.ModeRegression,
	})

	// train
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:    network.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       10000},
		network.WithL2Regularization(1.40e-06))
	
	model.Fit(xTrain, yTrain, true)
	model.Save("networkmodel.json")

	// make predictions
	yPred := model.Predict(xTrain)
}
```

**Use case**: Suitable for regression problems such as price prediction, temperature forecasting, time series analysis, etc.

---

### 2. Multiclass Classification (MNIST with Neural Networks)

```go
// examples/multiclass/mnist.go
package main

import (
	"fmt"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// load MNIST data
	xTrain := LoadDataFromFile("../dataset/mnist/train_x_shuffled.csv")
	yTrain := LoadDataFromFile("../dataset/mnist/train_label_shuffled.csv")
	xTest := LoadDataFromFile("../dataset/mnist/test_x.csv")
	yTest := LoadDataFromFile("../dataset/mnist/test_label.csv")

	// normalize
	applyNormalization := func(_, _ int, v float64) float64 { 
		return v / 255.0 
	}
	xTrain = ngo.Apply(applyNormalization, xTrain)
	xTest = ngo.Apply(applyNormalization, xTest)

	// create model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{784, 100, 100, 10},
		Activation:  network.TanhActivation,
		Mode:        network.ModeMultiClass,
	})

	// train
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:    network.AdamOptimizer,
		LearningRate: 0.0075,
		Epochs:       100},
		network.WithBatchSize(32),
		network.WithL2Regularization(1.40e-06))
	
	model.Summary()
	model.Fit(xTrain, yTrain, true)
	model.Save("networkmodel.json")

	// evaluate
	fmt.Printf("training accuracy: %.4f\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing accuracy:  %.4f\n", model.Evaluate(xTest, yTest))
}
```

**Use case**: Handwritten digit classification, image categorization, etc.

---

### 3. CNN for Image Classification (MNIST)

```go
// examples/cnn/mnist/mnist.go
package main

import (
	"fmt"
	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// load and normalize data
	x := LoadDataFromFile("../../dataset/mnist/train_x_shuffled.csv")
	v := LoadDataFromFile("../../dataset/mnist/test_x.csv")
	
	applyNormalization := func(_, _ int, v float64) float64 { 
		return v / 255.0 
	}
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	// convert to image format (28x28)
	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	for i := range xTrain {
		xTrain[i] = make([]*mat.Dense, 1)
		xTrain[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, x))
	}
	
	xTest := make([][]*mat.Dense, v.RawMatrix().Cols)
	for i := range xTest {
		xTest[i] = make([]*mat.Dense, 1)
		xTest[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, v))
	}

	yTrain := LoadDataFromFile("../../dataset/mnist/train_label_shuffled.csv")
	yTest := LoadDataFromFile("../../dataset/mnist/test_label.csv")

	// create CNN
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{1, 28, 28}, // 1 channel, 28x28
		Activation: cnn.ReLUActivation,
		Mode:       cnn.ModeMultiClass,
	})

	// build architecture
	neural.AddConv2DLayer(16, 3, 1)      // 16 filters 3x3
	neural.AddMaxPooling2DLayer(2, 2)    // Max pooling 2x2
	neural.AddConv2DLayer(32, 3, 1)      // 32 filters 3x3
	neural.AddMaxPooling2DLayer(2, 2)    // Max pooling 2x2
	neural.AddDenseLayer([]int{128, 10}) // Dense layers

	// train
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:    cnn.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       20},
		cnn.WithBatchSize(32),
		cnn.WithL2Regularization(1.40e-06))
	
	model.Summary()
	model.Fit(xTrain, yTrain, true)
	model.Save("networkmodel.json")

	// evaluate
	fmt.Printf("training accuracy: %.4f\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing accuracy:  %.4f\n", model.Evaluate(xTest, yTest))
}
```

**Use case**: Complex image classification, object detection, facial recognition, etc.

---

### 4. Multilabel Classification (Multiple Labels)

```go
// examples/multilabel/multilabel.go
package main

import (
	"fmt"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nlp"
)

func main() {
	// data with multiple labels per sample
	data := LoadTextsFromFile("../dataset/multilabel/texts.csv")
	dataLabel := LoadDataFromFile("../dataset/multilabel/texts_label.csv")

	vectorizer := nlp.NewCountVectorizer(3000)
	countMatrix := vectorizer.FitTransform(data...)

	xTrain, xTest := ngo.Split(countMatrix, 0.75)
	yTrain, yTest := ngo.Split(dataLabel, 0.75)

	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// create multilabel model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 64, 32, outputDim},
		Activation:  network.ReLUActivation,
		Mode:        network.ModeMultiLabel,
	})

	// train
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:    network.AdamOptimizer,
		LearningRate: 0.01,
		Epochs:       100},
		network.WithBatchSize(16))
	
	model.Fit(xTrain, yTrain, true)
	fmt.Printf("training accuracy: %.4f\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing accuracy:  %.4f\n", model.Evaluate(xTest, yTest))
}
```

**Use case**: Image tagging, multi-topic text classification, medical diagnosis with multiple conditions.

---

### 5. Hyperparameter Optimization

```go
// examples/hyperopt/hyperopt.go
package main

import (
	"fmt"
	"github.com/adynascimento/deep-learning/hyperopt"
)

func main() {
	// define search space
	space := hyperopt.SearchSpace{
		InputDim:          784,
		OutputDim:         10,
		NLayersRange:      []int{2, 3, 4},
		NHiddenRange:      []int{32, 64, 128},
		LearningRateRange: []float64{0.001, 0.01, 0.1},
		LambdRange:        []float64{0, 1e-6, 1e-4},
		NModels:           10,
	}

	// create optimizer
	hp := hyperopt.NewHyperparameterOptimization(space)

	// define objective function
	objective := func(trial int, params hyperopt.Params) float64 {
		// train model with these parameters
		// return metric (e.g., accuracy)
		return trainModel(params)
	}

    // bayesian optimization
	hp.BayesianOptimization(hyperopt.Maximize, objective)

	// or random search
	hp.RandomSearchOptimization(hyperopt.Maximize, objective)
	
	// get best parameters
	bestParams := hp.GetBestParams()
	fmt.Println("best parameters:", bestParams)
}
```

**Use case**: Automatically find the best network architecture and training parameters.

---

### 6. Data Scaling and Dimensionality Reduction

```go
// feature standardization with StandardScaler
package main

import (
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// create StandardScaler
	scaler := ngo.NewStandardScaler()
	
	// fit on training data
	scaler.Fit(xTrain)
	
	// transform both training and test data
	xTrainScaled := scaler.Transform(xTrain)
	xTestScaled := scaler.Transform(xTest)
	
	// or fit and transform in one step
	xScaled := scaler.FitTransform(x)
	
	// inverse transform to get original scale
	xOriginal := scaler.InverseTransform(xScaled)
	
	// access computed statistics
	means := scaler.GetMean()
	stdDevs := scaler.GetStdDev()
}
```

```go
// principal component analysis for dimensionality reduction
package main

import (
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// create PCA with desired number of components
	pca := ngo.NewPCA(nComponents)
	
	// fit PCA model
	pca.Fit(xTrain)
	
	// transform to reduced dimensions
	xReduced := pca.Transform(xTrain)
	
	// or fit and transform in one step
	xReduced := pca.FitTransform(xTrain)
	
	// reconstruct original space (with information loss)
	xReconstructed := pca.InverseTransform(xReduced)
	
	// get PCA components and explained variance
	components := pca.GetComponents()
	variance := pca.GetExplainedVariance()
}
```

**Use case**: Feature preprocessing, curse of dimensionality reduction, data compression before training.

---

### 7. CNN with Multi-channel Images (RGB)

```go
// examples/cnn/cats-vs-dogs/multilabel.go
package main

import (
	"fmt"
	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// load RGB image data
	x := LoadDataFromFile("../../dataset/cats-vs-dogs/train_x.csv")
	v := LoadDataFromFile("../../dataset/cats-vs-dogs/test_x.csv")
	
	// normalize pixel values
	applyNormalization := func(_, _ int, v float64) float64 { 
		return v / 255.0 
	}
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	// convert flattened data to 3-channel (RGB) image format
	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	for n := range xTrain {
		data := mat.Col(nil, n, x)
		
		// reshape into RGB channels
		rgb := make([][]float64, 3) // 3 channels: R, G, B
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])     // red channel
			rgb[1] = append(rgb[1], data[idx+1])   // green channel
			rgb[2] = append(rgb[2], data[idx+2])   // blue channel
		}
		
		xTrain[n] = make([]*mat.Dense, 3)
		xTrain[n][0] = mat.NewDense(100, 100, rgb[0])
		xTrain[n][1] = mat.NewDense(100, 100, rgb[1])
		xTrain[n][2] = mat.NewDense(100, 100, rgb[2])
	}

	xTest := make([][]*mat.Dense, v.RawMatrix().Cols)
	for n := range xTest {
		data := mat.Col(nil, n, v)

		// reshape into RGB channels
		rgb := make([][]float64, 3) // 3 channels: R, G, B
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])     // red channel
			rgb[1] = append(rgb[1], data[idx+1])   // green channel
			rgb[2] = append(rgb[2], data[idx+2])   // blue channel
		}

		xTest[n] = make([]*mat.Dense, 3)
		xTest[n][0] = mat.NewDense(100, 100, rgb[0])
		xTest[n][1] = mat.NewDense(100, 100, rgb[1])
		xTest[n][2] = mat.NewDense(100, 100, rgb[2])
	}

	yTrain := LoadDataFromFile("../../dataset/cats-vs-dogs/train_label.csv")
	yTest := LoadDataFromFile("../../dataset/cats-vs-dogs/test_label.csv")

	// create CNN for 3-channel (RGB) images
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{3, 100, 100}, // 3 RGB channels, 100x100 images
		Activation: cnn.ReLUActivation,
		Mode:       cnn.ModeMultiLabel,
	})

	// build architecture optimized for color images
	neural.AddConv2DLayer(32, 3, 1)      // 32 filters 3x3
	neural.AddMaxPooling2DLayer(2, 2)    // max pooling 2x2
	neural.AddConv2DLayer(64, 3, 1)      // 64 filters 3x3
	neural.AddMaxPooling2DLayer(2, 2)    // max pooling 2x2
	neural.AddConv2DLayer(128, 3, 1)     // 128 filters 3x3
	neural.AddMaxPooling2DLayer(2, 2)    // max pooling 2x2
	neural.AddDenseLayer([]int{256, yTrain.RawMatrix().Rows})

	// train with color images
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:    cnn.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       50},
		cnn.WithBatchSize(32),
		cnn.WithL2Regularization(1e-5))
	
	model.Summary()
	model.Fit(xTrain, yTrain, true)
	model.Save("networkmodel.json")

	// evaluate
	fmt.Printf("training accuracy: %.4f\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing accuracy:  %.4f\n", model.Evaluate(xTest, yTest))
}
```

**Use case**: Real-world image classification with color images, animals classification, scene recognition.

---

### 8. Natural Language Processing

```go
// example using NLP
package main

import (
	"github.com/adynascimento/deep-learning/nlp"
)

func main() {
	// Bag of Words
	texts := []string{
		"this is an example of text",
		"another example of processing",
		"text for language analysis",
	}
	
	bow := nlp.NewCountVectorizer(100)
	vectors := bow.FitTransform(texts...)

	// TF-IDF
	tfidf := nlp.NewTFIDFVectorizer(100)
	weights := tfidf.FitTransform(texts...)

	_, _ = vectors, weights
}
```

**Use case**: Text vectorization for ML models, sentiment analysis, document classification.

---

## 📁 Project Structure

```
deep-learning/
├── neuralnetwork/       # Artificial Neural Networks
│   ├── neural.go        # Core architecture
│   ├── activation.go    # Activation functions
│   ├── loss.go          # Loss functions
│   ├── optimizer.go     # Optimization algorithms
│   └── dataio.go        # Data I/O
├── cnn/                 # Convolutional Neural Networks
│   ├── cnn.go           # CNN architecture
│   ├── convlayer.go     # Convolutional layers
│   ├── convolve2D.go    # 2D convolution operation
│   ├── poolinglayer.go  # Max Pooling
│   ├── flatten.go       # Flatten layer
│   └── activation.go    # CNN activations
├── hyperopt/            # Hyperparameter Optimization
│   ├── optimization.go  # Interface
│   ├── bayesian.go      # Bayesian Optimization
│   └── randomsearch.go  # Random Search
├── ngo/                 # Mathematical Utilities
│   ├── matrix.go        # Matrix operations
│   ├── floats.go        # Float operations
│   ├── scaler.go        # Data scaling
│   └── pca.go           # PCA
├── nlp/                 # Natural Language Processing
│   ├── bow.go           # Bag of Words
│   ├── tfidf.go         # TF-IDF
├── examples/            # Usage examples
│   ├── regression/      # Regression example
│   ├── multiclass/      # Multiclass classification
│   ├── multilabel/      # Multilabel classification
│   ├── cnn/mnist/       # CNN with MNIST
│   └── dataset/         # Training data
└── go.mod               # Go module
```

---

## 🚀 Installation

### Prerequisites
- Go 1.22 or higher
- Gonum (managed by go.mod)

### Steps

Install the package in your Go project:

```bash
go get github.com/adynascimento/deep-learning
```

Then import it:

```go
import (
	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/hyperopt"
	"github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nlp"
)
```

To run the examples from this repository:

```bash
cd examples/regression
go run regression.go
```

---

## 📦 Main Dependencies

- **Gonum**: Numerical computing in Go
- **Goptuna**: Bayesian hyperparameter optimization
- **Plot** ([github.com/adynascimento/plot](https://github.com/adynascimento/plot)): Graph visualization and plotting
- **ProgressBar**: Training progress bar
- **TableWriter**: Table formatting for output

---

## 🚀 Advanced Features

### ConvolveFFT2D: Fast Fourier Transform Convolution

For large filters or high-dimensional convolutions, the library provides FFT-based convolution for superior performance:

```go
package cnn

// automatically optimized FFT-based convolution
// ideal for large filters (> 7x7) or performance-critical applications
// uses Gonum's FFT implementation for efficient computation
func ConvolveFFT2D(x, filter *mat.Dense, stride int) *mat.Dense {
	// ... FFT implementation
}
```

**When to use**: Large convolutional filters, high-resolution images, performance optimization in production.

---

### Hyperparameter Sampling Functions

For intelligent hyperparameter exploration:

```go
package ngo

// uniform integer sampling [min, max]
learningRateExp := ngo.SuggestInt(1, 5) // e.g., for 0.0001 to 0.00001

// uniform float sampling [min, max]
lr := ngo.SuggestFloat(0.0001, 0.1)

// log-uniform float sampling (better for learning rates and regularization)
lr := ngo.SuggestLogFloat(1e-5, 1e-1) // logarithmically distributed
```

**Use case**: Hyperparameter optimization, random search, Bayesian optimization with Goptuna.

---

## 🎓 Supported Concepts

### Activation Functions
- **Tanh**: Zero-centered activation commonly used in hidden layers
- **ReLU**: Nonlinear activation that keeps positive values and clips negatives to zero
- **Sigmoid**: Maps values to the 0-1 range
- **ELU**: Smooth nonlinear activation with negative outputs for negative inputs
- **Softmax**: For multiclass classification

### Optimizers
- **Adam**: Adapts learning rate per parameter with momentum and squared gradient tracking (recommended)
- **Gradient Descent**: Classic gradient descent with configurable learning rate
- Bias correction for Adam optimizer
- Layer-wise learning rate adaptation

### Regularization
- **L2 (Ridge)**: Penalizes large weights to prevent overfitting

### Training Modes
- **Regression**: MSE loss, linear output
- **Multiclass**: Cross-entropy loss, softmax
- **Multilabel**: Binary cross-entropy, sigmoid

---

## 🔧 Advanced Configuration

### Batch Training
```go
model := neural.NewTrainer(config,
	network.WithBatchSize(32))
```

### L2 Regularization
```go
model := neural.NewTrainer(config,
	network.WithL2Regularization(1e-6))
```

### Saving and Loading Models
```go
// save
model.Save("my_model.json")

// load an ANN model
loadedANN := network.Load("my_model.json")

// load a CNN model
loadedCNN := cnn.Load("my_cnn_model.json")
```

---

## 💡 Recommended Use Cases

| Task | Component | Example |
|------|-----------|----------|
| Continuous value prediction | ANN + Regression | Real estate pricing |
| Image classification | CNN | MNIST, CIFAR-10 |
| Text classification | ANN + NLP | Sentiment analysis |
| Multiple labels | ANN + Multilabel | Product tagging |
| Find optimal parameters | Hyperopt | Architecture tuning |



## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details. All computational code follows standard open-source practices and is provided as-is.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements.

---

## 📚 Additional Resources

- [Gonum Documentation](https://www.gonum.org/)
- [Deep Learning Fundamentals](https://en.wikipedia.org/wiki/Deep_learning)
- [CNN Architecture](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Hyperparameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

---

**Designed to explore deep learning concepts in Go with a simple and intuitive code.**
