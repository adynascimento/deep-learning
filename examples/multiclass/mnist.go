package main

import (
	"fmt"

	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
)

func main() {
	// training data
	xTrain := LoadDataFromFile("../dataset/mnist/train_x_shuffled.csv")
	yTrain := LoadDataFromFile("../dataset/mnist/train_label_shuffled.csv")

	// testing data
	xTest := LoadDataFromFile("../dataset/mnist/test_x.csv")
	yTest := LoadDataFromFile("../dataset/mnist/test_label.csv")

	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	xTrain = ngo.Apply(applyNormalization, xTrain)
	xTest = ngo.Apply(applyNormalization, xTest)

	// input and output features
	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 100, 100, outputDim}, // neural network structure
		Activation:  network.TanhActivation,               // activation function
		Mode:        network.ModeMultiClass,               // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:    network.AdamOptimizer, // optimizer
		LearningRate: 0.0075,                // learning rate
		NIterations:  400},                  // number of iterations
		network.WithBatchSize(32),
		network.WithL2Regularization(1.40e-06))
	model.Summary()
	model.Fit(xTrain, yTrain, true)

	// saves neural network model to file
	model.Save("networkmodel.json")

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))

	number, probability := PredictFromImage(model, "../dataset/mnist/numbers/4.png")
	fmt.Printf("prediction of the model: number %d (%.1f %% probability)\n", number, probability)
}
