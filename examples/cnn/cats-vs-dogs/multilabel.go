package main

import (
	"fmt"

	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	x := LoadDataFromFile("../../dataset/cats-vs-dogs/train_x.csv")
	v := LoadDataFromFile("../../dataset/cats-vs-dogs/test_x.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	for n := range xTrain {
		data := mat.Col(nil, n, x)

		rgb := make([][]float64, 3)
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])
			rgb[1] = append(rgb[1], data[idx+1])
			rgb[2] = append(rgb[2], data[idx+2])
		}

		xTrain[n] = make([]*mat.Dense, 3)
		xTrain[n][0] = mat.NewDense(100, 100, rgb[0])
		xTrain[n][1] = mat.NewDense(100, 100, rgb[1])
		xTrain[n][2] = mat.NewDense(100, 100, rgb[2])
	}
	xTest := make([][]*mat.Dense, v.RawMatrix().Cols)
	for n := range xTest {
		data := mat.Col(nil, n, v)

		rgb := make([][]float64, 3)
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])
			rgb[1] = append(rgb[1], data[idx+1])
			rgb[2] = append(rgb[2], data[idx+2])
		}

		xTest[n] = make([]*mat.Dense, 3)
		xTest[n][0] = mat.NewDense(100, 100, rgb[0])
		xTest[n][1] = mat.NewDense(100, 100, rgb[1])
		xTest[n][2] = mat.NewDense(100, 100, rgb[2])
	}
	yTrain := LoadDataFromFile("../../dataset/cats-vs-dogs/train_label.csv")
	yTest := LoadDataFromFile("../../dataset/cats-vs-dogs/test_label.csv")

	// neural network model
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{3, 100, 100},
		Activation: cnn.ReLUActivation,
		Mode:       cnn.ModeMultiLabel,
	})
	neural.AddConv2DLayer(8, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddDenseLayer([]int{64, yTrain.RawMatrix().Rows})

	// optimizer to train the model
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:    cnn.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       30},
		cnn.WithBatchSize(32),
	)
	model.Summary()
	model.Fit(xTrain, yTrain, true)

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))
}
