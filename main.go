package main

import (
	"fmt"
	"math"

	network "deep_learning/neuralNetwork"
	ngo "deep_learning/numeric"

	"gonum.org/v1/gonum/mat"
)

func main() {

	// training data
	applySin := func(_, _ int, v float64) float64 { return math.Sin(15. * v) }
	x_train := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	y_train := ngo.Apply(applySin, x_train)

	// hyperparameters
	input_dim := x_train.RawMatrix().Rows
	output_dim := y_train.RawMatrix().Rows

	nn_structure := []int{input_dim, 40, 20, 10, output_dim}
	activation_function := network.ActivationTanh
	optimizer := network.AdamOptimizer
	learning_rate := 0.001
	num_iterations := 10000

	// neural network model
	model := network.NewNeuralNetwork(
		nn_structure,
		activation_function,
		num_iterations,
	)

	// optimizer to train the model
	model.NewTrainer(optimizer, learning_rate)
	model.Fit(x_train, y_train, true)

	// make predictions
	predictions := model.Predict(x_train)
	fmt.Println(predictions.Dims())

}
