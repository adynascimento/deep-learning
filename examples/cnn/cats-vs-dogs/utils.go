package main

import (
	"encoding/csv"
	"image"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"strconv"

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

// load data and image labels
func LoadData(basepath string) ([][]*mat.Dense, *mat.Dense) {
	xData := [][]*mat.Dense{}
	labels := [][]float64{}

	cats, _ := os.ReadDir(filepath.Join(basepath, "cats"))
	dogs, _ := os.ReadDir(filepath.Join(basepath, "dogs"))
	for i := 0; i < len(cats); i++ {
		xData = append(xData,
			LoadFromImage(filepath.Join(basepath, "cats", cats[i].Name())),
			LoadFromImage(filepath.Join(basepath, "dogs", dogs[i].Name())),
		)
		labels = append(labels, []float64{1}, []float64{0})
	}

	// image labels
	y := mat.NewDense(1, len(xData), nil)
	for j, v := range labels {
		y.SetCol(j, v)
	}

	return xData, y
}

// function to convert an image to []*mat.Dense
func LoadFromImage(path string) []*mat.Dense {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading image from file:", err.Error())
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Println("error decoding image:", err.Error())
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// initialize slices for each channel (R, G, B)
	rValues := make([]float64, width*height)
	gValues := make([]float64, width*height)
	bValues := make([]float64, width*height)

	// populate the slices with pixel values, normalized by dividing by 255
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			index := y*width + x
			rValues[index] = float64(r>>8) / 255.0
			gValues[index] = float64(g>>8) / 255.0
			bValues[index] = float64(b>>8) / 255.0
		}
	}

	// create dense matrices from the slices
	rMatrix := mat.NewDense(height, width, rValues)
	gMatrix := mat.NewDense(height, width, gValues)
	bMatrix := mat.NewDense(height, width, bValues)

	return []*mat.Dense{rMatrix, gMatrix, bMatrix}
}
