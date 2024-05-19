package main

import (
	"math"

	network "github.com/adynascimento/deep-learning/neuralnetwork"
	ngo "github.com/adynascimento/deep-learning/numeric"

	"github.com/adynascimento/plot/plot"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	applySin := func(_, _ int, v float64) float64 { return math.Sin(15. * v) }
	xTrain := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	yTrain := ngo.Apply(applySin, xTrain)

	// input and output features
	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 40, 20, 10, outputDim}, // neural network structure
		Activation:  network.TanhActivation,                 // activation function
		Mode:        network.ModeRegression,                 // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:        network.AdamOptimizer, // optimizer
		LearningRate:     0.001,                 // learning rate
		L2Regularization: 1.40e-06,              // l2 regularization
		NIterations:      10000,                 // number of iterations
	})
	model.Fit(xTrain, yTrain, true)

	// saves neural network model to file
	model.Save("networkmodel.json")

	// make predictions
	yPred := model.Predict(xTrain)

	// plotting
	plt := plot.NewPlot()
	plt.FigSize(12, 9)

	plt.Plot(xTrain.RawMatrix().Data, yTrain.RawMatrix().Data)
	plt.Plot(xTrain.RawMatrix().Data, yPred.RawMatrix().Data)
	plt.Title("neural network predictions")
	plt.XLabel("x values")
	plt.YLabel("y values")
	plt.Legend("true model", "prediction")
	plt.XLim(0.0, 1.0)
	plt.YLim(-1.0, 1.0)

	plt.Save("prediction.png")
}
