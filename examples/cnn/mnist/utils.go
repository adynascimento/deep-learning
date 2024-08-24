package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	_ "image/png"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/adynascimento/deep-learning/cnn"
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

	m := mat.NewDense(len(lines[0]), len(lines), nil)
	for j, line := range lines {
		for i, col := range line {
			value, _ := strconv.ParseFloat(col, 64)
			m.Set(i, j, value)
		}
	}

	return m
}

func PredictFromImage(model cnn.CNNModel, path string) (int, float64) {
	// make predictions
	yPred := model.Predict(LoadFromImage(path))

	fmt.Println("prediction from image:")
	fmt.Println(mat.Formatted(yPred))
	idx := floats.MaxIdx(mat.Col(nil, 0, yPred))

	return idx, math.Floor(yPred.At(idx, 0)*1000.0) / 10.0
}

func LoadFromImage(path string) [][]*mat.Dense {
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
