package cnn

import (
	"math"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

var convOptimizerSettings = map[optimizerType]convOptimizer{
	GradientDescentOptimizer: {
		Name:     GradientDescentOptimizer,
		Function: convGradientDescentOptimizer,
	},
	AdamOptimizer: {
		Name:     AdamOptimizer,
		Function: convAdamOptimizer,
	},
}

type convOptimizerFunction func(*convOptimizer, parameters, [][]*mat.Dense, *mat.Dense, float64, float64) parameters

type convOptimizer struct {
	Name     optimizerType
	Function convOptimizerFunction
	Adam     convAdamParameters
}

type convAdamParameters struct {
	vdW [][]*mat.Dense
	sdW [][]*mat.Dense
	vdb *mat.Dense
	sdb *mat.Dense
}

// update the parameters (gradient descent)
// weights with shape (nFilters, nChannels, filterSize, filterSize)
// biases with shape (nFilters, 1)
func convGradientDescentOptimizer(opt *convOptimizer, parameters parameters, dW [][]*mat.Dense, db *mat.Dense,
	learningRate, t float64) parameters {
	nFilters := len(parameters.W)
	nChannels := len(parameters.W[0])
	for f := 0; f < nFilters; f++ {
		for c := 0; c < nChannels; c++ {
			parameters.W[f][c] = ngo.Sub(parameters.W[f][c], ngo.Scale(learningRate, dW[f][c]))
		}
	}
	parameters.B = ngo.Sub(parameters.B, ngo.Scale(learningRate, db))

	return parameters
}

// update the parameters (adam optimizer)
// weights with shape (nFilters, nChannels, filterSize, filterSize)
// biases with shape (nFilters, 1)
func convAdamOptimizer(opt *convOptimizer, parameters parameters, dW [][]*mat.Dense, db *mat.Dense,
	learningRate, t float64) parameters {
	// default parameters
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-08

	// initializing the adam model parameters
	nFilters := len(parameters.W)
	nChannels := len(parameters.W[0])

	applySqrt := func(_, _ int, v float64) float64 { return math.Sqrt(v) }
	for f := 0; f < nFilters; f++ {
		for c := 0; c < nChannels; c++ {
			// moving average of the gradients
			// compute bias-corrected first moment estimate
			opt.Adam.vdW[f][c] = ngo.Add(ngo.Scale(beta1, opt.Adam.vdW[f][c]), ngo.Scale((1-beta1), dW[f][c]))
			vdWCorr := ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.Adam.vdW[f][c])

			// moving average of the squared gradients
			// compute bias-corrected second raw moment estimate
			opt.Adam.sdW[f][c] = ngo.Add(ngo.Scale(beta2, opt.Adam.sdW[f][c]), ngo.Scale((1.0-beta2), ngo.Square(dW[f][c])))
			sdWCorr := ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.Adam.sdW[f][c])

			sqrtW := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sdWCorr))
			parameters.W[f][c] = ngo.Sub(parameters.W[f][c], ngo.Scale(learningRate, ngo.DivElem(vdWCorr, sqrtW)))
		}
	}

	// moving average of the gradients
	// compute bias-corrected first moment estimate
	opt.Adam.vdb = ngo.Add(ngo.Scale(beta1, opt.Adam.vdb), ngo.Scale((1-beta1), db))
	vdbCorr := ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.Adam.vdb)

	// moving average of the squared gradients
	// compute bias-corrected second raw moment estimate
	opt.Adam.sdb = ngo.Add(ngo.Scale(beta2, opt.Adam.sdb), ngo.Scale((1.0-beta2), ngo.Square(db)))
	sdbCorr := ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.Adam.sdb)

	sqrtb := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sdbCorr))
	parameters.B = ngo.Sub(parameters.B, ngo.Scale(learningRate, ngo.DivElem(vdbCorr, sqrtb)))

	return parameters
}

// initializing the adam model parameters
func convInitializeAdam(W [][]*mat.Dense) convAdamParameters {
	// initializing the adam model parameters
	nFilters := len(W)
	nChannels := len(W[0])
	filterSize, _ := W[0][0].Dims()
	vdW := make([][]*mat.Dense, nFilters)
	sdW := make([][]*mat.Dense, nFilters)
	for i := range vdW {
		vdW[i] = make([]*mat.Dense, nChannels)
		sdW[i] = make([]*mat.Dense, nChannels)
		for j := range vdW[i] {
			vdW[i][j] = mat.NewDense(filterSize, filterSize, nil)
			sdW[i][j] = mat.NewDense(filterSize, filterSize, nil)
		}
	}
	vdb := mat.NewDense(nFilters, 1, nil)
	sdb := mat.NewDense(nFilters, 1, nil)

	return convAdamParameters{
		vdW: vdW,
		sdW: sdW,
		vdb: vdb,
		sdb: sdb,
	}
}
