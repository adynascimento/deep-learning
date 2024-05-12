package main

import (
	"encoding/csv"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"math"
	"os"
	"strconv"

	network "github.com/adynascimento/deep-learning/neuralnetwork"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	xTrain := loadFromFile("dataset/train_x.csv")
	yTrain := loadFromFile("dataset/train_label.csv")

	// testing data
	xTest := loadFromFile("dataset/test_x.csv")
	yTest := loadFromFile("dataset/test_label.csv")

	// input and output features
	inputDim := xTrain.RawMatrix().Rows
	outputDim := yTrain.RawMatrix().Rows

	// neural network model
	neural := network.NewNeuralNetwork(network.NeuralConfig{
		NNStructure: []int{inputDim, 100, 50, outputDim}, // neural network structure
		Activation:  network.ActivationTanh,              // activation function
		Mode:        network.ModeMultiClass,              // mode determines output layer activation and loss function
	})

	// optimizer to train the model
	model := network.NewTrainer(neural, network.TrainerConfig{
		Optimizer:        network.AdamOptimizer, // optimizer
		LearningRate:     0.0075,                // learning rate
		L2Regularization: 1.40e-06,              // l2 regularization
		NIterations:      1000,                  // number of iterations
	})
	model.Fit(xTrain, yTrain, true)

	// saves neural network model to file
	model.Save("model.json")

	// accuracy of the model making predictions
	fmt.Printf("accuracy of training data: %.1f %%\n", accuracy(model, xTrain, yTrain))
	fmt.Printf("accuracy of testing data:   %.1f %%\n", accuracy(model, xTest, yTest))

	number, probability := predictFromImage(model, "dataset/numbers/4.png")
	fmt.Printf("prediction of the model: number %d (%.1f %% probability)\n", number, probability)
}

func loadFromFile(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading features from file: ", err.Error())
	}
	defer file.Close()

	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Println("error reading features from file: ", err.Error())
	}

	m := mat.NewDense(len(lines[0]), len(lines), nil)
	for j, line := range lines {
		for i, col := range line {
			value, _ := strconv.ParseFloat(col, 64)
			m.Set(i, j, value)
		}
	}

	return m
}

func accuracy(model network.NeuralModel, x, y *mat.Dense) float64 {
	// make predictions
	predictions := model.Predict(x)

	var count float64
	for j := 0; j < y.RawMatrix().Cols; j++ {
		y := floats.MaxIdx(mat.Col(nil, j, y))
		yPredict := floats.MaxIdx(mat.Col(nil, j, predictions))

		if y == yPredict {
			count++
		}
	}

	return (count / float64(y.RawMatrix().Cols)) * 100.0
}

func predictFromImage(model network.NeuralModel, path string) (int, float64) {
	x := loadFromImage(path)

	// make predictions
	predictions := model.Predict(x)
	fmt.Println(mat.Formatted(predictions))
	idx := floats.MaxIdx(mat.Col(nil, 0, predictions))

	return idx, math.Floor(predictions.At(idx, 0)*1000.0) / 10.0
}

func loadFromImage(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading image from file: ", err.Error())
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Println("error decoding image: ", err.Error())
	}

	grayImg := image.NewGray(img.Bounds())
	for y := 0; y < img.Bounds().Max.Y; y++ {
		for x := 0; x < img.Bounds().Max.X; x++ {
			grayImg.Set(x, y, img.At(x, y))
		}
	}

	m := mat.NewDense(len(grayImg.Pix), 1, nil)
	for i, v := range grayImg.Pix {
		m.Set(i, 0, float64(v))
	}

	return m
}
