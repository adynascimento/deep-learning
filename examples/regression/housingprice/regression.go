package main

import (
	/*
		#cgo CFLAGS: -g -O2
		#cgo LDFLAGS: -framework Accelerate
	*/
	"C"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"

	"github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)
import "fmt"

func init() {
	// force the Gonum ecosystem to use the optimized C engine
	blas64.Use(netlib.Implementation{})
}

func main() {
	data := LoadDataFromFile("california_housing.csv")

	// split the data into features and target variable
	x := data.Slice(0, data.RawMatrix().Rows-1, 0, data.RawMatrix().Cols).(*mat.Dense)
	y := mat.NewDense(1, data.RawMatrix().Cols, data.RawRowView(data.RawMatrix().Rows-1))

	// split the dataset into training and testing sets
	xTrain, xTest := ngo.Split(x, 0.80)
	yTrain, yTest := ngo.Split(y, 0.80)

	// feature scaling
	scaler := ngo.NewStandardScaler()
	xTrain = scaler.FitTransform(xTrain) // fit the scaler to the training data and transform it
	xTest = scaler.Transform(xTest)      // transform the testing data using the same scaler fitted to the training data

	// input and output features
	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := neuralnetwork.NewNeuralNetwork(neuralnetwork.NeuralConfig{
		NNStructure: []int{inputDim, 128, 64, 32, 16, outputDim},
		Activation:  neuralnetwork.ReLUActivation,
		Mode:        neuralnetwork.ModeRegression,
	})

	// optimizer to train the model
	model := neural.NewTrainer(neuralnetwork.TrainerConfig{
		Optimizer:    neuralnetwork.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       100})
	model.Summary()
	model.Fit(xTrain, yTrain, true)

	// accuracy of the model making predictions
	fmt.Println()
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))
}
