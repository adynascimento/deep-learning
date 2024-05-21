package dataset

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

func LoadFromFile(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading features from file:", err.Error())
	}
	defer file.Close()

	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Println("error reading features from file:", err.Error())
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

func PredictFromImage(model network.NeuralModel, path string) (int, float64) {
	x := loadFromImage(path)

	// make predictions
	yPred := model.Predict(x)

	fmt.Println("prediction from image:")
	fmt.Println(mat.Formatted(yPred))
	idx := floats.MaxIdx(mat.Col(nil, 0, yPred))

	return idx, math.Floor(yPred.At(idx, 0)*1000.0) / 10.0
}

func loadFromImage(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading image from file:", err.Error())
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Println("error decoding image:", err.Error())
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
