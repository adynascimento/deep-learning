# Deep Feedforward Neural Network (DNN) Code from Scratch Using Golang

A comprehensive deep learning library written in Go from scratch, featuring support for Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Hyperparameter Optimization, Mathematical Utilities and Natural Language Processing (NLP).

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
- Optimized convolution using Im2Col + GEMM
- FFT-based convolution available for large kernels
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

## ✨ Highlights

- ✔ Written entirely in Go
- ✔ BLAS-accelerated matrix operations (Gonum)
- ✔ Optimized convolution using Im2Col + GEMM
- ✔ Hyperparameter optimization
- ✔ Multi-channel image support
- ✔ Model serialization
- ✔ Natural Language Processing utilities
- ✔ Mathematical utilities (PCA, StandardScaler, Sampling Functions)
- ✔ Batch training

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
	
	model.Fit(xTrain, yTrain)
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
	model.Fit(xTrain, yTrain)
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
	model.Fit(xTrain, yTrain)
	model.Save("networkmodel.json")

	// evaluate
	fmt.Printf("training accuracy: %.4f\n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("testing accuracy:  %.4f\n", model.Evaluate(xTest, yTest))
}
```

**Use case**: Complex image classification, object detection, facial recognition, etc.

---

### 4. Hyperparameter Optimization

```go
// examples/hyperopt/hyperopt.go
package main

import (
	"fmt"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// define search space
	space := hyperopt.SearchSpace{
			InputDim:          xTrain.RawMatrix().Rows,
			OutputDim:         yTrain.RawMatrix().Rows,
			NLayersRange:      []int{3, 5},           // minimum and maximum number of layers
			NHiddenRange:      []int{50, 100},        // minimum and maximum number of hidden units per layer
			LearningRateRange: []float64{1e-4, 1e-2}, // minimum and maximum of learning rate
			LambdRange:        []float64{1e-6, 1e-2}, // minimum and maximum of regularization parameter
			NModels:           3,                     // number of models
		})

	// create optimizer
	hp := hyperopt.NewHyperparameterOptimization(space)

	// define objective function
	objective := func(trialID int, params hyperopt.Params) float64 {
		// neural network model
		neural := network.NewNeuralNetwork(network.NeuralConfig{
			NNStructure: params.NNStructure,     // neural network structure
			Activation:  network.TanhActivation, // activation function
			Mode:        network.ModeMultiClass, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(network.TrainerConfig{
			Optimizer:    network.AdamOptimizer, // optimizer
			LearningRate: params.LearningRate,   // learning rate
			Epochs:       400},                  // number of iterations
			network.WithBatchSize(32),
			network.WithL2Regularization(params.L2Regularization),
		)
		model.Fit(xTrain, yTrain, network.WithVerbose(false))
		model.Save("./trials/networkmodel" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTrain, yTrain)
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

### 5. Data Scaling and Dimensionality Reduction

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

### 6. Natural Language Processing

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
// save an ANN model
model.Save("ann_model.json")

// load an ANN model
loadedANN := network.Load("ann_model.json")

// load a CNN model
loadedCNN := cnn.Load("cnn_model.json")
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

This project is licensed under the MIT License. See the [MIT LICENSE](LICENSE) file for details.

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
