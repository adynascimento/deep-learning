package nncore

import (
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type LossFunction func(*mat.Dense, *mat.Dense, map[string]*mat.Dense, float64) float64

// computing the mean squared error loss function
func MeanSquaredError(y, yHat *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := yHat.RawMatrix().Rows
	loss := mat.Sum(ngo.Square(ngo.Sub(yHat, y)))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum += mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss += lambd * sum

	return (1.0 / (2.0 * float64(m)) * loss)
}

// computing the cross entropy loss function
func CrossEntropy(y, yHat *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := yHat.RawMatrix().Rows

	epsilon := 1e-07
	applyLog := func(_, _ int, v float64) float64 { return math.Log(v + epsilon) }
	loss := (-1.0 / (float64(m))) * mat.Sum(ngo.Multiply(y, ngo.Apply(applyLog, yHat)))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum += mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss += (0.5 / (float64(m))) * lambd * sum

	return loss
}

// computing the binary cross entropy loss function
func BinaryCrossEntropy(y, yHat *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := yHat.RawMatrix().Rows

	epsilon := 1e-07
	applyLog := func(_, _ int, v float64) float64 { return math.Log(v + epsilon) }
	applyOneMinusLog := func(_, _ int, v float64) float64 { return math.Log(1 - v + epsilon) }
	applyOneMinus := func(_, _ int, v float64) float64 { return 1 - v }

	term1 := ngo.Multiply(y, ngo.Apply(applyLog, yHat))
	term2 := ngo.Multiply(ngo.Apply(applyOneMinus, y), ngo.Apply(applyOneMinusLog, yHat))
	loss := (-1.0 / (float64(m))) * mat.Sum(ngo.Add(term1, term2))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum += mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss += (0.5 / (float64(m))) * lambd * sum

	return loss
}
