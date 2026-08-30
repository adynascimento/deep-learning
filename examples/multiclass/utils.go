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

	"github.com/adynascimento/deep-learning/mlp"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func LoadDataFromFile(path string) *mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading features from file:", err.Error())
	}
	defer file.Close()

	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Println("error reading features from file:", err.Error())
	}

	m := mat.NewDense(len(lines), len(lines[0]), nil)
	for i, line := range lines {
		for j, col := range line {
			value, _ := strconv.ParseFloat(col, 64)
			m.Set(i, j, value)
		}
	}

	return m
}

func PredictFromImage(model mlp.NeuralModel, path string) (int, float64) {
	x := LoadFromImage(path)

	// make predictions
	yPred := model.Predict(x)

	fmt.Println("prediction from image:")
	fmt.Println(mat.Formatted(yPred.T()))
	idx := floats.MaxIdx(yPred.RawRowView(0))

	return idx, math.Floor(yPred.At(0, idx)*1000.0) / 10.0
}

func LoadFromImage(path string) *mat.Dense {
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

	m := mat.NewDense(1, len(grayImg.Pix), nil)
	for i, v := range grayImg.Pix {
		m.Set(0, i, float64(v)/255.0)
	}

	return m
}
