package main

import (
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/dataset"
	hyperopt "github.com/adynascimento/deep-learning/hyperparameter"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
)

func main() {
	// training data
	xTrain := dataset.LoadFromFile("../../dataset/mnist/train_x.csv")
	yTrain := dataset.LoadFromFile("../../dataset/mnist/train_label.csv")

	neuralNetworkModel := func(trialID int, nnStructure []int, lambd float64) float64 {
		// neural network model
		neural := network.NewNeuralNetwork(network.NeuralConfig{
			NNStructure: nnStructure,            // neural network structure
			Activation:  network.TanhActivation, // activation function
			Mode:        network.ModeMultiClass, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(network.TrainerConfig{
			Optimizer:        network.AdamOptimizer, // optimizer
			LearningRate:     0.0075,                // learning rate
			L2Regularization: lambd,                 // l2 regularization
			NIterations:      1000,                  // number of iterations
		})
		model.Fit(xTrain, yTrain, false)

		// saves neural network model to file
		model.Save("./trials/networkmodel" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTrain, yTrain)
	}

	model := hyperopt.NewHyperparameterOptimization(hyperopt.Params{
		InputDim:     xTrain.RawMatrix().Rows,
		OutputDim:    yTrain.RawMatrix().Rows,
		NLayersRange: []int{3, 5},                                   // minimum and maximum number of layers
		NHiddenRange: []int{50, 100},                                // minimum and maximum number of hidden units per layers
		LambdRange:   []float64{math.Pow(10, -6), math.Pow(10, -2)}, // minimum and maximum of regularization parameter
		NModels:      3,                                             // number of models
	})

	model.RandomSearchOptimization(hyperopt.Maximize, neuralNetworkModel)
}
