package main

import (
	"fmt"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	"github.com/adynascimento/deep-learning/mlp"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
)

func main() {
	// training data
	xTrain := LoadDataFromFile("../../dataset/mnist/train_x_shuffled.csv")
	yTrain := LoadDataFromFile("../../dataset/mnist/train_label_shuffled.csv")

	// testing data
	xTest := LoadDataFromFile("../../dataset/mnist/test_x.csv")
	yTest := LoadDataFromFile("../../dataset/mnist/test_label.csv")

	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	xTrain = ngo.Apply(applyNormalization, xTrain)
	xTest = ngo.Apply(applyNormalization, xTest)

	model := func(trialID int, params mlp.Params) float64 {
		nnStructure := []int{xTrain.RawMatrix().Cols}              // input dimension
		nnStructure = append(nnStructure, params.HiddenLayers...)  // hidden layers
		nnStructure = append(nnStructure, yTrain.RawMatrix().Cols) // output dimension

		// neural network model
		neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
			NNStructure: nnStructure,           // neural network structure
			Activation:  nncore.ReLUActivation, // activation function
			Mode:        nncore.ModeMultiClass, // mode determines output layer activation and loss function
		})

		// optimizer to train the model
		model := neural.NewTrainer(mlp.TrainerConfig{
			Optimizer:    nncore.AdamOptimizer, // optimizer
			LearningRate: params.LearningRate,  // learning rate
			Epochs:       20},                  // number of epochs
			mlp.WithBatchSize(32),
			mlp.WithL2Regularization(params.L2Regularization),
		)
		model.Fit(xTrain, yTrain, mlp.WithVerbose(false))
		model.Save("./trials/model" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTest, yTest)
	}

	hp := mlp.NewHyperopt(mlp.SearchSpace{
		NHiddenLayersRange: mlp.IntRange{Min: 1, Max: 3},         // minimum and maximum number of hidden layers
		NHiddenRange:       mlp.IntRange{Min: 50, Max: 100},      // minimum and maximum number of hidden units per layers
		LearningRateRange:  mlp.FloatRange{Min: 1e-4, Max: 1e-2}, // minimum and maximum of learning rate
		L2Range:            mlp.FloatRange{Min: 1e-6, Max: 1e-2}, // minimum and maximum of regularization parameter
		NTrials:            3,                                    // number of trials
	})

	hp.Optimize(hyperopt.Bayesian, hyperopt.Maximize, model)
	fmt.Println("best params:", hp.GetBestParams())
}
