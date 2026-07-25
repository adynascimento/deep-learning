package main

import (
	"math"

	"github.com/adynascimento/deep-learning/mlp"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"

	"github.com/adynascimento/plot/plotter"
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
	neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
		NNStructure: []int{inputDim, 40, 20, 10, outputDim}, // neural network structure
		Activation:  nncore.TanhActivation,                  // activation function
		Mode:        nncore.ModeRegression,                  // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(mlp.TrainerConfig{
		Optimizer:    nncore.AdamOptimizer, // optimizer
		LearningRate: 0.001,                // learning rate
		Epochs:       10000},               // number of iterations
		mlp.WithL2Regularization(1.40e-06))
	model.Fit(xTrain, yTrain)

	// saves neural network model to file
	model.Save("networkmodel.json")

	// make predictions
	yPred := model.Predict(xTrain)

	// plotting
	plt := plotter.NewPlot()
	plt.FigSize(12, 9)

	plt.Plot(xTrain.RawMatrix().Data, yTrain.RawMatrix().Data)
	plt.Plot(xTrain.RawMatrix().Data, yPred.RawMatrix().Data,
		plotter.WithLineColor(plotter.Blue),
		plotter.WithMarker(plotter.Circle),
		plotter.WithMarkerSpacing(8),
	)
	plt.Title("neural network predictions")
	plt.XLabel("x values")
	plt.YLabel("y values")
	plt.Legend("function", "prediction")
	plt.XLim(0.0, 1.0)
	plt.YLim(-1.0, 1.0)
	plt.Grid()

	// plt.Save("prediction.png")
	plt.Show()
}
