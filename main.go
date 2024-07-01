package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/adynascimento/deep-learning/cnn"
	"github.com/adynascimento/deep-learning/examples/dataset"
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	x := dataset.LoadDataFromFile("examples/dataset/mnist/train_x_shuffled.csv")
	v := dataset.LoadDataFromFile("examples/dataset/mnist/test_x.csv")
	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	x = ngo.Apply(applyNormalization, x)
	v = ngo.Apply(applyNormalization, v)

	xTrain := make([][]*mat.Dense, x.RawMatrix().Cols)
	xTest := make([][]*mat.Dense, v.RawMatrix().Cols)
	for i := range xTrain {
		xTrain[i] = make([]*mat.Dense, 1)
		xTrain[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, x))
	}
	for i := range xTest {
		xTest[i] = make([]*mat.Dense, 1)
		xTest[i][0] = mat.NewDense(28, 28, mat.Col(nil, i, v))
	}
	yTrain := dataset.LoadDataFromFile("examples/dataset/mnist/train_label_shuffled.csv")
	yTest := dataset.LoadDataFromFile("examples/dataset/mnist/test_label.csv")

	// neural network model
	neural := cnn.NewConvNeuralNetwork(cnn.CNNConfig{
		InputShape: [3]int{1, 28, 28},
		Activation: cnn.ReLUActivation,
		Mode:       cnn.ModeMultiClass,
	})
	neural.AddConv2DLayer(32, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddConv2DLayer(64, 3, 1)
	neural.AddMaxPooling2DLayer(2, 2)
	neural.AddDenseLayer([]int{128, yTrain.RawMatrix().Rows})

	// optimizer to train the model
	model := neural.NewTrainer(cnn.TrainerConfig{
		Optimizer:        cnn.AdamOptimizer, // optimizer
		LearningRate:     0.001,             // learning rate
		L2Regularization: 1.40e-06,          // l2 regularization
		NIterations:      20,                // number of iterations
		BatchSize:        32,                // batch size
	})
	model.Summary()
	model.Fit(xTrain, yTrain, true)

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.4f \n", model.Evaluate(xTrain, yTrain))
	fmt.Printf("accuracy of testing data:  %.4f \n", model.Evaluate(xTest, yTest))

	number, probability := predictFromImage(model, "examples/dataset/mnist/numbers/4.png")
	fmt.Printf("prediction of the model: number %d (%.1f%% probability)\n", number, probability)
}

func predictFromImage(model cnn.CNNModel, path string) (int, float64) {
	// make predictions
	yPred := model.Predict(loadFromImage(path))

	fmt.Println("prediction from image:")
	fmt.Println(mat.Formatted(yPred))
	idx := floats.MaxIdx(mat.Col(nil, 0, yPred))

	return idx, math.Floor(yPred.At(idx, 0)*1000.0) / 10.0
}

func loadFromImage(path string) [][]*mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading image from file:", err.Error())
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Println("error decoding image:", err.Error())
	}

	var values []float64
	for y := 0; y < img.Bounds().Max.Y; y++ {
		for x := 0; x < img.Bounds().Max.X; x++ {
			v := float64(color.GrayModel.Convert(img.At(x, y)).(color.Gray).Y)
			values = append(values, float64(v)/255.0)
		}
	}

	m := make([][]*mat.Dense, 1)
	m[0] = make([]*mat.Dense, 1)
	m[0][0] = mat.NewDense(img.Bounds().Max.X, img.Bounds().Max.Y, values)

	return m
}
