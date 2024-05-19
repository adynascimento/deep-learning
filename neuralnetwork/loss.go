package neuralnetwork

import (
	"math"
	"strconv"

	ngo "github.com/adynascimento/deep-learning/numeric"

	"gonum.org/v1/gonum/mat"
)

type lossFunction func(*mat.Dense, *mat.Dense, map[string]*mat.Dense, float64) float64

// computing the mean squared error loss function
func meanSquaredError(yHat, y *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := yHat.RawMatrix().Cols
	loss := mat.Sum(ngo.Square(ngo.Sub(yHat, y)))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum = sum + mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss = loss + lambd*sum

	return (1.0 / (2.0 * float64(m)) * loss)
}

// computing the cross entropy loss function
func crossEntropy(y_hat, y *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := y_hat.RawMatrix().Cols

	epsilon := 1e-07
	applyLog := func(_, _ int, v float64) float64 { return math.Log(v + epsilon) }
	loss := mat.Sum(ngo.Multiply(y, ngo.Apply(applyLog, y_hat)))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum = sum + mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss = loss + 0.5*lambd*sum

	return -(1.0 / (float64(m)) * loss)
}

// computing the binary cross entropy loss function
func binaryCrossEntropy(y_hat, y *mat.Dense, parameters map[string]*mat.Dense, lambd float64) float64 {
	m := y_hat.RawMatrix().Cols

	epsilon := 1e-07
	applyLog := func(_, _ int, v float64) float64 { return math.Log(v + epsilon) }
	applyOneMinusLog := func(_, _ int, v float64) float64 { return math.Log(1 - v + epsilon) }
	applyOneMinus := func(_, _ int, v float64) float64 { return 1 - v }

	term1 := ngo.Multiply(y, ngo.Apply(applyLog, y_hat))
	term2 := ngo.Multiply(ngo.Apply(applyOneMinus, y), ngo.Apply(applyOneMinusLog, y_hat))
	loss := mat.Sum(ngo.Add(term1, term2))

	// l2 regularization loss
	L := len(parameters) / 2 // number of layers
	var sum float64
	for l := 0; l < L; l++ {
		sum = sum + mat.Sum(ngo.Square(parameters["W"+strconv.Itoa(l+1)]))
	}
	loss = loss + 0.5*lambd*sum

	return -(1.0 / (float64(m)) * loss)
}
