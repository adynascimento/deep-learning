package main

import (
	"encoding/csv"
	"log"
	"os"
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

func LoadTextsFromFile(path string) []string {
	file, err := os.Open(path)
	if err != nil {
		log.Println("error loading features from file:", err.Error())
	}
	defer file.Close()

	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Println("error reading features from file:", err.Error())
	}

	var texts []string
	for _, line := range lines {
		if len(line) > 0 {
			texts = append(texts, line[0])
		}
	}

	return texts
}
