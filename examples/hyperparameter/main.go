package main

import (
	"math"
	"strconv"

	hyperopt "github.com/adynascimento/deep-learning/hyperparameter"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	ngo "github.com/adynascimento/deep-learning/numeric"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	applySin := func(_, _ int, v float64) float64 { return math.Sin(15. * v) }
	xTrain := mat.NewDense(1, 301, ngo.Linspace(0., 1., 301))
	yTrain := ngo.Apply(applySin, xTrain)

	neuralNetworkModel := func(trialID int, nnStructure []int, lambd float64) float64 {
		// neural network model
		neural := network.NewNeuralNetwork(network.NeuralConfig{
			NNStructure: nnStructure,            // neural network structure
			Activation:  network.ActivationTanh, // activation function
			Mode:        network.ModeRegression, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := network.NewTrainer(neural, network.TrainerConfig{
			Optimizer:        network.AdamOptimizer, // optimizer
			LearningRate:     0.001,                 // learning rate
			L2Regularization: lambd,                 // l2 regularization
			NIterations:      10000,                 // number of iterations
		})
		model.Fit(xTrain, yTrain, false)

		// make predictions and evaluate mean square error
		yPred := model.Predict(xTrain)
		err := mat.Sum(ngo.Square(ngo.Sub(yTrain, yPred))) / float64(yTrain.RawMatrix().Cols)

		// saves neural network model to file
		model.Save("./trials/model_" + strconv.Itoa(trialID) + ".json")

		return err
	}

	model := hyperopt.NewHyperparameterOptimization(hyperopt.Params{
		InputDim:     xTrain.RawMatrix().Rows,
		OutputDim:    yTrain.RawMatrix().Rows,
		NLayersRange: []int{3, 8},                                   // minimum and maximum number of layers
		NHiddenRange: []int{8, 36},                                  // minimum and maximum number of hidden units per layers
		LambdRange:   []float64{math.Pow(10, -6), math.Pow(10, -2)}, // minimum and maximum of regularization parameter
		NModels:      3,                                             // number of models
	})

	model.RandomSearchOptimization(neuralNetworkModel)
}
