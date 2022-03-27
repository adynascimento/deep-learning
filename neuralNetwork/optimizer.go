package neuralNetwork

import (
	ngo "deep_learning/numeric"
	"math"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type optimizerType string

const (
	AdamOptimizer            optimizerType = "adam"
	GradientDescentOptimizer optimizerType = "gradientDescent"
)

type optimizerFunction func(map[string]*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, float64, float64) map[string]*mat.Dense

type adam struct {
	v map[string]*mat.Dense
	s map[string]*mat.Dense
}

type optimizer struct {
	name              optimizerType
	optimizerFunction optimizerFunction
	adam              adam
}

// update the parameters (gradient descent)
func (model *optimizer) gradientDescentOptimizer(parameters, dW, db map[string]*mat.Dense, learning_rate, t float64) map[string]*mat.Dense {
	L := len(parameters) / 2 // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, dW[strconv.Itoa(l+1)]))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, db[strconv.Itoa(l+1)]))
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
func (model *optimizer) adamOptimizer(parameters, dW, db map[string]*mat.Dense, learning_rate, t float64) map[string]*mat.Dense {
	// default parameters
	beta_1 := 0.9
	beta_2 := 0.999
	epsilon := 1e-08

	vCorr := make(map[string]*mat.Dense) // map containing the parameters
	sCorr := make(map[string]*mat.Dense) // map containing the parameters

	L := len(parameters) / 2 // number of layers

	applySqrt := func(_, _ int, v float64) float64 { return math.Sqrt(v) }

	for l := 0; l < L; l++ {
		// moving average of the gradients
		model.adam.v["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta_1, model.adam.v["dW"+strconv.Itoa(l+1)]), ngo.Scale((1-beta_1), dW[strconv.Itoa(l+1)]))
		model.adam.v["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta_1, model.adam.v["db"+strconv.Itoa(l+1)]), ngo.Scale((1-beta_1), db[strconv.Itoa(l+1)]))

		// compute bias-corrected first moment estimate
		vCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta_1, t)), model.adam.v["dW"+strconv.Itoa(l+1)])
		vCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta_1, t)), model.adam.v["db"+strconv.Itoa(l+1)])

		// moving average of the squared gradients
		model.adam.s["dW"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta_2, model.adam.s["dW"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta_2), ngo.Square(dW[strconv.Itoa(l+1)])))
		model.adam.s["db"+strconv.Itoa(l+1)] = ngo.Add(ngo.Scale(beta_2, model.adam.s["db"+strconv.Itoa(l+1)]), ngo.Scale((1.0-beta_2), ngo.Square(db[strconv.Itoa(l+1)])))

		// compute bias-corrected second raw moment estimate
		sCorr["dW"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta_2, t)), model.adam.s["dW"+strconv.Itoa(l+1)])
		sCorr["db"+strconv.Itoa(l+1)] = ngo.Scale(1.0/(1.0-math.Pow(beta_2, t)), model.adam.s["db"+strconv.Itoa(l+1)])

		sqrtW := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["dW"+strconv.Itoa(l+1)]))
		sqrtb := ngo.Apply(func(_, _ int, v float64) float64 { return v + epsilon }, ngo.Apply(applySqrt, sCorr["db"+strconv.Itoa(l+1)]))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Sub(parameters["W"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, ngo.DivElem(vCorr["dW"+strconv.Itoa(l+1)], sqrtW)))
		parameters["b"+strconv.Itoa(l+1)] = ngo.Sub(parameters["b"+strconv.Itoa(l+1)], ngo.Scale(learning_rate, ngo.DivElem(vCorr["db"+strconv.Itoa(l+1)], sqrtb)))
	}

	return parameters
}
