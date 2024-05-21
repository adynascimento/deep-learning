package neuralnetwork

import (
	"math"
	"strconv"

	ngo "github.com/adynascimento/deep-learning/gonum"

	"gonum.org/v1/gonum/mat"
)

var optimizerSettings = map[optimizerType]optimizer{
	GradientDescentOptimizer: {
		Name:     GradientDescentOptimizer,
		Function: gradientDescentOptimizer,
	},
	AdamOptimizer: {
		Name:     AdamOptimizer,
		Function: adamOptimizer,
	},
}

type optimizerType string
type optimizerFunction func(*optimizer, map[string]*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, float64, float64) map[string]*mat.Dense

const (
	AdamOptimizer            optimizerType = "adam"
	GradientDescentOptimizer optimizerType = "gradientdescent"
)

type optimizer struct {
	Name     optimizerType
	Function optimizerFunction
	Adam     adamParameters
}

type adamParameters struct {
	v map[string]*mat.Dense
	s map[string]*mat.Dense
}

// update the parameters (gradient descent)
func gradientDescentOptimizer(opt *optimizer, parameters, dW, db map[string]*mat.Dense, learningRate, t float64) map[string]*mat.Dense {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, dW[strconv.Itoa(l+1)]))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, db[strconv.Itoa(l+1)]))
	}

	return parameters
}

// update the parameters (adam optimizer)
func adamOptimizer(opt *optimizer, parameters, dW, db map[string]*mat.Dense, learningRate, t float64) map[string]*mat.Dense {
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
		opt.Adam.v["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, opt.Adam.v["dW"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), dW[strconv.Itoa(l+1)]))
		opt.Adam.v["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, opt.Adam.v["db"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), db[strconv.Itoa(l+1)]))

		// compute bias-corrected first moment estimate
		vCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.Adam.v["dW"+strconv.Itoa(l+1)])
		vCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.Adam.v["db"+strconv.Itoa(l+1)])

		// moving average of the squared gradients
		opt.Adam.s["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, opt.Adam.s["dW"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(dW[strconv.Itoa(l+1)])))
		opt.Adam.s["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, opt.Adam.s["db"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(db[strconv.Itoa(l+1)])))

		// compute bias-corrected second raw moment estimate
		sCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.Adam.s["dW"+strconv.Itoa(l+1)])
		sCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.Adam.s["db"+strconv.Itoa(l+1)])

		sqrtW := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["dW"+strconv.Itoa(l+1)]))
		sqrtb := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["db"+strconv.Itoa(l+1)]))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["dW"+strconv.Itoa(l+1)], sqrtW)))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["db"+strconv.Itoa(l+1)], sqrtb)))
	}

	return parameters
}

// initializing the adam model parameters
func initializeAdam(parameters map[string]*mat.Dense) adamParameters {
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

	return adamParameters{
		v: v,
		s: s,
	}
}
