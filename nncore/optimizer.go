package nncore

import (
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type OptimizerType string
type OptimizerFunction func(*Optimizer, map[string]*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, float64, float64)

const (
	AdamOptimizer            OptimizerType = "adam"
	GradientDescentOptimizer OptimizerType = "gradientdescent"
)

type Optimizer struct {
	Name     OptimizerType
	Function OptimizerFunction

	adam AdamState
}

type AdamState struct {
	v map[string]*mat.Dense
	s map[string]*mat.Dense
}

func NewOptimizer(optType OptimizerType, parameters map[string]*mat.Dense) Optimizer {
	switch optType {
	case GradientDescentOptimizer:
		return Optimizer{
			Name:     optType,
			Function: gradientDescentOptimizer,
		}

	case AdamOptimizer:
		return Optimizer{
			Name:     optType,
			Function: adamOptimizer,
			adam:     initializeAdam(parameters),
		}

	default:
		panic("optimizer not implemented")
	}
}

// update the parameters (gradient descent)
func gradientDescentOptimizer(opt *Optimizer, parameters, dW, db map[string]*mat.Dense, learningRate, t float64) {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, dW[strconv.Itoa(l+1)]))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, db[strconv.Itoa(l+1)]))
	}
}

// update the parameters (adam optimizer)
func adamOptimizer(opt *Optimizer, parameters, dW, db map[string]*mat.Dense, learningRate, t float64) {
	// default parameters
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-08

	vCorr := make(map[string]*mat.Dense) // map containing the parameters
	sCorr := make(map[string]*mat.Dense) // map containing the parameters

	L := len(parameters) / 2 // number of layers

	applySqrt := func(_, _ int, v float64) float64 { return math.Sqrt(v) }
	for l := 0; l < L; l++ {
		// moving average of the gradients
		opt.adam.v["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, opt.adam.v["dW"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), dW[strconv.Itoa(l+1)]))
		opt.adam.v["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, opt.adam.v["db"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), db[strconv.Itoa(l+1)]))

		// compute bias-corrected first moment estimate
		vCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.adam.v["dW"+strconv.Itoa(l+1)])
		vCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.adam.v["db"+strconv.Itoa(l+1)])

		// moving average of the squared gradients
		opt.adam.s["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, opt.adam.s["dW"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(dW[strconv.Itoa(l+1)])))
		opt.adam.s["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, opt.adam.s["db"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(db[strconv.Itoa(l+1)])))

		// compute bias-corrected second raw moment estimate
		sCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.adam.s["dW"+strconv.Itoa(l+1)])
		sCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.adam.s["db"+strconv.Itoa(l+1)])

		sqrtW := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["dW"+strconv.Itoa(l+1)]))
		sqrtb := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["db"+strconv.Itoa(l+1)]))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["dW"+strconv.Itoa(l+1)], sqrtW)))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["db"+strconv.Itoa(l+1)], sqrtb)))
	}
}

// initializing the adam model parameters
func initializeAdam(parameters map[string]*mat.Dense) AdamState {
	L := len(parameters) / 2         // number of layers
	v := make(map[string]*mat.Dense) // map containing the parameters
	s := make(map[string]*mat.Dense) // map containing the parameters

	for l := 0; l < L; l++ {
		nw, mw := parameters["W"+strconv.Itoa(l+1)].Dims()
		nb, mb := parameters["b"+strconv.Itoa(l+1)].Dims()

		v["dW"+strconv.Itoa(l+1)] = mat.NewDense(nw, mw, nil)
		v["db"+strconv.Itoa(l+1)] = mat.NewDense(nb, mb, nil)
		s["dW"+strconv.Itoa(l+1)] = mat.NewDense(nw, mw, nil)
		s["db"+strconv.Itoa(l+1)] = mat.NewDense(nb, mb, nil)
	}

	return AdamState{
		v: v,
		s: s,
	}
}
