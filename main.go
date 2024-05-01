package main

import (
	"fmt"
	"math"

	network "deep_learning/neuralNetwork"
	ngo "deep_learning/numeric"

	"github.com/adyllyson-gomes/plot/plot"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	applySin := func(_, _ int, v float64) float64 { return math.Sin(15. * v) }
	x_train := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	y_train := ngo.Apply(applySin, x_train)

	// input and output features
	input_dim := x_train.RawMatrix().Rows
	output_dim := y_train.RawMatrix().Rows

	// hyperparameters
	nn_structure := []int{input_dim, 40, 20, 10, output_dim} // neural network structure
	activation_function := network.ActivationTanh            // activation function
	l2_regularization := 1.40e-06                            // regularization parameter
	num_iterations := 10000                                  // number of iterations

	// neural network model
	model := network.NewNeuralNetwork(
		nn_structure,
		activation_function,
		l2_regularization,
		num_iterations,
	)

	// optimizer to train the model
	learning_rate := 0.001
	model.NewTrainer(network.AdamOptimizer, learning_rate)
	model.Fit(x_train, y_train, true)

	// saves neural network model to file
	model.Save("model.json")

	// make predictions
	predictions := model.Predict(x_train)
	fmt.Println(predictions.Dims())

	// plotting
	plt := plot.NewPlot()
	plt.FigSize(12, 9)

	plt.Plot(x_train.RawMatrix().Data, y_train.RawMatrix().Data)
	plt.Plot(x_train.RawMatrix().Data, predictions.RawMatrix().Data)
	plt.Title("neural network predictions")
	plt.XLabel("x values")
	plt.YLabel("y values")
	plt.Legend("true model", "prediction")
	plt.XLim(0.0, 1.0)
	plt.YLim(-1.0, 1.0)

	plt.Save("plot.png")
}
