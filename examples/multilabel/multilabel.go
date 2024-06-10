package main

import (
	"fmt"

	"github.com/adynascimento/deep-learning/examples/dataset"
	network "github.com/adynascimento/deep-learning/neuralnetwork"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nlp"
)

func main() {
	// loading data
	data := dataset.LoadTextsFromFile("../dataset/multilabel/texts.csv")
	dataLabel := dataset.LoadDataFromFile("../dataset/multilabel/texts_label.csv")

	// preprocessing dataset by doing vectorizaton
	vectorizer := nlp.NewCountVectorizer(3000)
	countMatrix := vectorizer.FitTransform(data...)

	//split data into training and testing dataset
	xTrain, xTest := ngo.Split(countMatrix, 0.75)
	yTrain, yTest := ngo.Split(dataLabel, 0.75)

	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 20, 20, outputDim}, // neural network structure
		Activation:  network.TanhActivation,             // activation function
		Mode:        network.ModeMultiLabel,             // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(network.TrainerConfig{
		Optimizer:        network.AdamOptimizer, // optimizer
		LearningRate:     1e-03,                 // learning rate
		L2Regularization: 1e-5,                  // l2 regularization
		NIterations:      5000,                  // number of iterations
	})
	model.Fit(xTrain, yTrain, true)

	// saves neural network model to file
	model.Save("networkmodel.json")

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))
}
