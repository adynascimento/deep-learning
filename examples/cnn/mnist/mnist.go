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

	// neural network model
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{1, 28, 28},
		Activation: nncore.ReLUActivation,
		Mode:       nncore.ModeMultiClass,
	})
	neural.AddConv2DLayer(16, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddConv2DLayer(32, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddDenseLayer([]int{128, yTrain.RawMatrix().Rows})

	// optimizer to train the model
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:    nncore.AdamOptimizer, // optimizer
		LearningRate: 0.001,                // learning rate
		Epochs:       20},                  // number of iterations
		cnn.WithBatchSize(32),
		cnn.WithL2Regularization(1.40e-06))
	model.Summary()
	model.Fit(xTrain, yTrain)

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))

	number, probability := PredictFromImage(model, "../../dataset/mnist/numbers/4.png")
	fmt.Printf("prediction of the model: number %d (%.1f%% probability)\n", number, probability)
}
