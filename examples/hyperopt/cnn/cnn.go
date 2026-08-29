package main

import (
	"fmt"
	"strconv"

	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/hyperopt"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	x := LoadDataFromFile("../../dataset/mnist/train_x_shuffled.csv")
	v := LoadDataFromFile("../../dataset/mnist/test_x.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	for i := range xTrain {
		xTrain[i] = make([]*mat.Dense, 1)
		xTrain[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, x))
	}
	xTest := make([][]*mat.Dense, v.RawMatrix().Cols)
	for i := range xTest {
		xTest[i] = make([]*mat.Dense, 1)
		xTest[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, v))
	}
	yTrain := LoadDataFromFile("../../dataset/mnist/train_label_shuffled.csv")
	yTest := LoadDataFromFile("../../dataset/mnist/test_label.csv")

	model := func(trialID int, params cnn.Params) float64 {
		// neural network model
		neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
			InputShape: [3]int{1, 28, 28},
			Activation: nncore.ReLUActivation,
			Mode:       nncore.ModeMultiClass,
		})

		// conv layers
		nFilters := 16
		for i := 0; i < params.NConvLayers; i++ {
			neural.AddConv2DLayer(nFilters, 3, 1)
			neural.AddMaxPooling2DLayer(2, 2)
			nFilters *= 2
		}

		// dense layer
		denseStructure := append([]int{}, params.HiddenLayers...)        // hidden layers
		denseStructure = append(denseStructure, yTrain.RawMatrix().Rows) // output dimension
		neural.AddDenseLayer(denseStructure)

		// optimizer to train the model
		model := neural.NewTrainer(cnn.TrainerConfig{
			Optimizer:    nncore.AdamOptimizer, // optimizer
			LearningRate: params.LearningRate,  // learning rate
			Epochs:       20},                  // number of iterations
			cnn.WithBatchSize(32),
			cnn.WithL2Regularization(params.L2Regularization),
		)
		model.Fit(xTrain, yTrain, cnn.WithVerbose(false))
		model.Save("./trials/model" + strconv.Itoa(trialID) + ".json")

		// make predictions and evaluate model
		return model.Evaluate(xTest, yTest)
	}

	hp := cnn.NewHyperopt(cnn.SearchSpace{
		NConvLayersRange:   cnn.IntRange{Min: 1, Max: 3},
		NHiddenLayersRange: cnn.IntRange{Min: 1, Max: 3},
		NHiddenRange:       cnn.IntRange{Min: 100, Max: 400},
		LearningRateRange:  cnn.FloatRange{Min: 1e-4, Max: 1e-2}, // minimum and maximum of learning rate
		L2Range:            cnn.FloatRange{Min: 1e-6, Max: 1e-2}, // minimum and maximum of regularization parameter
		NTrials:            3,                                    // number of trials
	})

	hp.Optimize(hyperopt.Bayesian, hyperopt.Maximize, model)
	fmt.Println("best params:", hp.GetBestParams())
}
