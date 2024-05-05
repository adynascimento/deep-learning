package neuralnetwork

import (
	ngo "deep_learning/numeric"
	"math"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type optimizerType string
type optimizerFunction func(map[string]*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, float64, float64) map[string]*mat.Dense

const (
	AdamOptimizer            optimizerType = "adam"
	GradientDescentOptimizer optimizerType = "gradientDescent"
)

type adamParameters struct {
	v map[string]*mat.Dense
	s map[string]*mat.Dense
}

type optimizer struct {
	Name     optimizerType
	Function optimizerFunction
	Adam     adamParameters
}

// update the parameters (gradient descent)
func (model *optimizer) GradientDescentOptimizer(parameters, dW, db map[string]*mat.Dense, learningRate, t float64) map[string]*mat.Dense {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, dW[strconv.Itoa(l+1)]))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, db[strconv.Itoa(l+1)]))
	}

	return parameters
}

// initializing the adam model parameters
func initializeAdam(parameters map[string]*mat.Dense) (map[string]*mat.Dense, map[string]*mat.Dense) {
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

	return v, s
}

// update the parameters (adam optimizer)
func (model *optimizer) AdamOptimizer(parameters, dW, db map[string]*mat.Dense, learningRate, t float64) map[string]*mat.Dense {
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
		model.Adam.v["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, model.Adam.v["dW"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), dW[strconv.Itoa(l+1)]))
		model.Adam.v["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta1, model.Adam.v["db"+strconv.Itoa(l+1)]), ngo.Scale((1-beta1), db[strconv.Itoa(l+1)]))

		// compute bias-corrected first moment estimate
		vCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), model.Adam.v["dW"+strconv.Itoa(l+1)])
		vCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), model.Adam.v["db"+strconv.Itoa(l+1)])

		// moving average of the squared gradients
		model.Adam.s["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, model.Adam.s["dW"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(dW[strconv.Itoa(l+1)])))
		model.Adam.s["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta2, model.Adam.s["db"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta2), ngo.Square(db[strconv.Itoa(l+1)])))

		// compute bias-corrected second raw moment estimate
		sCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), model.Adam.s["dW"+strconv.Itoa(l+1)])
		sCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), model.Adam.s["db"+strconv.Itoa(l+1)])

		sqrtW := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["dW"+strconv.Itoa(l+1)]))
		sqrtb := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["db"+strconv.Itoa(l+1)]))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["dW"+strconv.Itoa(l+1)], sqrtW)))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learningRate, ngo.DivElem(vCorr["db"+strconv.Itoa(l+1)], sqrtb)))
	}

	return parameters
}
