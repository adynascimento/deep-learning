package main

import (
	"fmt"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// training data
	xTrain := LoadDataFromFile("../dataset/mnist/train_x_shuffled.csv")
	yTrain := LoadDataFromFile("../dataset/mnist/train_label_shuffled.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	xTrain = ngo.Apply(applyNormalization, xTrain)

	neuralNetworkModel := func(trialID int, params hyperopt.Params) float64 {
		// neural network model
		neural := network.NewNeuralNetwork(network.NeuralConfig{
			NNStructure: params.NNStructure,     // neural network structure
			Activation:  network.TanhActivation, // activation function
			Mode:        network.ModeMultiClass, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(network.TrainerConfig{
			Optimizer:        network.AdamOptimizer,   // optimizer
			LearningRate:     params.LearningRate,     // learning rate
			L2Regularization: params.L2Regularization, // l2 regularization
			NIterations:      200,                     // number of iterations
			BatchSize:        32,                      // batch size
		})
		model.Fit(xTrain, yTrain, false)
		model.Save("./trials/networkmodel" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTrain, yTrain)
	}

	study := hyperopt.NewHyperparameterOptimization(
		hyperopt.SearchSpace{
			InputDim:          xTrain.RawMatrix().Rows,
			OutputDim:         yTrain.RawMatrix().Rows,
			NLayersRange:      []int{3, 5},           // minimum and maximum number of layers
			NHiddenRange:      []int{50, 100},        // minimum and maximum number of hidden units per layers
			LearningRateRange: []float64{1e-4, 1e-2}, // minimum and maximum of learning rate
			LambdRange:        []float64{1e-6, 1e-2}, // minimum and maximum of regularization parameter
			NModels:           3,                     // number of models
		})

	study.BayesianOptimization(hyperopt.Maximize, neuralNetworkModel)
	fmt.Println("best params:", study.GetBestParams())
}
