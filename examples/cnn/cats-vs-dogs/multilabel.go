package main

import (
	"fmt"

	/*
		#cgo CFLAGS: -g -O2
		#cgo LDFLAGS: -framework Accelerate
	*/
	"C"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"

	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/mat"
)

func main() {
	blas64.Use(netlib.Implementation{})

	// training data
	x := LoadDataFromFile("../../dataset/cats-vs-dogs/train_x.csv")
	v := LoadDataFromFile("../../dataset/cats-vs-dogs/test_x.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	xTrain := make([][]*mat.Dense, x.RawMatrix().Rows)
	for i := range xTrain {
		data := x.RawRowView(i)

		rgb := make([][]float64, 3)
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])
			rgb[1] = append(rgb[1], data[idx+1])
			rgb[2] = append(rgb[2], data[idx+2])
		}

		xTrain[i] = make([]*mat.Dense, 3)
		xTrain[i][0] = mat.NewDense(100, 100, rgb[0])
		xTrain[i][1] = mat.NewDense(100, 100, rgb[1])
		xTrain[i][2] = mat.NewDense(100, 100, rgb[2])
	}
	xTest := make([][]*mat.Dense, v.RawMatrix().Rows)
	for i := range xTest {
		data := v.RawRowView(i)

		rgb := make([][]float64, 3)
		for idx := 0; idx < len(data); idx += 3 {
			rgb[0] = append(rgb[0], data[idx])
			rgb[1] = append(rgb[1], data[idx+1])
			rgb[2] = append(rgb[2], data[idx+2])
		}

		xTest[i] = make([]*mat.Dense, 3)
		xTest[i][0] = mat.NewDense(100, 100, rgb[0])
		xTest[i][1] = mat.NewDense(100, 100, rgb[1])
		xTest[i][2] = mat.NewDense(100, 100, rgb[2])
	}
	yTrain := LoadDataFromFile("../../dataset/cats-vs-dogs/train_label.csv")
	yTest := LoadDataFromFile("../../dataset/cats-vs-dogs/test_label.csv")

	// neural network model
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{3, 100, 100},
		Activation: nncore.ReLUActivation,
		Mode:       nncore.ModeMultiLabel,
	})
	neural.AddConv2DLayer(16, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddConv2DLayer(32, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddDenseLayer([]int{32, yTrain.RawMatrix().Cols})

	// optimizer to train the model
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:    nncore.AdamOptimizer,
		LearningRate: 0.001,
		Epochs:       20},
		cnn.WithBatchSize(32),
	)
	model.Summary()
	model.Fit(xTrain, yTrain)

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))
}
