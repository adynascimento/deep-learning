package main

import (
	"fmt"

	"github.com/adynascimento/deep-learning/mlp"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nlp"
	"github.com/adynascimento/deep-learning/nncore"
)

func main() {
	// loading data
	data := LoadTextsFromFile("../dataset/multilabel/texts.csv")
	dataLabel := LoadDataFromFile("../dataset/multilabel/texts_label.csv")

	// preprocessing dataset by doing vectorizaton
	vectorizer := nlp.NewCountVectorizer(3000)
	countMatrix := vectorizer.FitTransform(data...)

	//split data into training and testing dataset
	xTrain, xTest := ngo.Split(countMatrix, 0.75)
	yTrain, yTest := ngo.Split(dataLabel, 0.75)

	inputDim := xTrain.RawMatrix().Cols
	outputDim := yTrain.RawMatrix().Cols

	// neural network model
	neural := mlp.NewNeuralNetwork(mlp.NeuralConfig{
		NNStructure: []int{inputDim, 20, 20, outputDim}, // neural network structure
		Activation:  nncore.TanhActivation,              // activation function
		Mode:        nncore.ModeMultiLabel,              // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := neural.NewTrainer(mlp.TrainerConfig{
		Optimizer:    nncore.AdamOptimizer, // optimizer
		LearningRate: 1e-03,                // learning rate
		Epochs:       20},                  // number of iterations
		mlp.WithBatchSize(32),
		mlp.WithL2Regularization(1.0e-06),
		mlp.WithDropout(0.4),
	)
	model.Fit(xTrain, yTrain)

	// saves neural network model to file
	model.Save("model.json")

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))
}
