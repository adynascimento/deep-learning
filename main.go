package main

import (
	"fmt"
	"math"

	model "deep_learning/neuralNetwork"
	ngo "deep_learning/numeric"

	"gonum.org/v1/gonum/mat"
)

func main() {

	// training data
	x_train := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	values := []float64{}
	for _, v := range x_train.RawMatrix().Data {
		values = append(values, math.Sin(15.*v))
	}
	y_train := mat.NewDense(1, 301, values)

	// hyperparameters
	nn_structure := []int{1, 40, 20, 10, 1}
	num_iterations := 40001
	learning_rate := 0.08

	// neural network model
	parameters, _ := model.Fit(x_train, y_train, nn_structure, num_iterations, learning_rate, true)

	// make predictions
	predictions := model.Predict(parameters, x_train)
	fmt.Println(predictions.Dims())

}
