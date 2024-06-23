package main

import (
	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/examples/dataset"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// x := make([][]*mat.Dense, 2)
	// for i := range x {
	// 	x[i] = make([]*mat.Dense, 1)
	// }
	// x[0][0] = mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	// x[1][0] = mat.NewDense(3, 3, []float64{10, 11, 12, 13, 14, 15, 16, 17, 18})

	// training data
	x := dataset.LoadDataFromFile("examples/dataset/mnist/train_x_shuffled.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	x = ngo.Apply(applyNormalization, x)

	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	yTrain := dataset.LoadDataFromFile("examples/dataset/mnist/train_label_shuffled.csv")
	for i := range xTrain {
		xTrain[i] = make([]*mat.Dense, 1)
		xTrain[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, x))
	}

	// neural network model
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{1, 28, 28},
		Activation: cnn.ReLUActivation,
		Mode:       cnn.ModeMultiClass,
	})
	neural.AddConvLayer(8, 3, 1)
	neural.AddPoolLayer(2, 2)
	neural.AddConvLayer(16, 3, 1)
	neural.AddPoolLayer(2, 2)
	neural.AddDenseLayer([]int{128, yTrain.RawMatrix().Rows})

	// optimizer to train the model
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:        cnn.AdamOptimizer, // optimizer
		LearningRate:     0.001,             // learning rate
		L2Regularization: 1.40e-06,          // l2 regularization
		NIterations:      20,                // number of iterations
		BatchSize:        32,                // batch size
	})
	// model.Summary()
	model.Fit(xTrain, yTrain, true)
}
