package main

import (
	"fmt"

	"github.com/adynascimento/deep-learning/examples/dataset"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
)

func main() {
	// training data
	xTrain := dataset.LoadFromFile("../../dataset/mnist/train_x.csv")
	yTrain := dataset.LoadFromFile("../../dataset/mnist/train_label.csv")

	// testing data
	xTest := dataset.LoadFromFile("../../dataset/mnist/test_x.csv")
	yTest := dataset.LoadFromFile("../../dataset/mnist/test_label.csv")

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
		Optimizer:        network.AdamOptimizer, // optimizer
		LearningRate:     0.0075,                // learning rate
		L2Regularization: 1.40e-06,              // l2 regularization
		NIterations:      1000,                  // number of iterations
	})
	model.Fit(xTrain, yTrain, true)

	// saves neural network model to file
	model.Save("networkmodel.json")

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))

	number, probability := dataset.PredictFromImage(model, "dataset/numbers/4.png")
	fmt.Printf("prediction of the model: number %d (%.1f %% probability)\n", number, probability)
}
