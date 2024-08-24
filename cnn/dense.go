package cnn

import (
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type denseLayer struct {
	NNStructure      []int
	Activation       activation
	OutputActivation outputActivation
	Optimizer        denseOptimizer
	Parameters       map[string]*mat.Dense
	Iter             float64
}

func newDenseLayer(nnStructure []int, activation activation, outputActivation outputActivation, optType optimizerType) *denseLayer {
	// initializing the model parameters
	parameters := initializeParameters(nnStructure)

	// choice of optimization algorithm
	optimizer := denseOptimizerSettings[optType]
	if optType == AdamOptimizer {
		optimizer.Adam = denseInitializeAdam(parameters)
	}

	return &denseLayer{
		NNStructure:      nnStructure,
		Activation:       activation,
		OutputActivation: outputActivation,
		Optimizer:        optimizer,
		Parameters:       parameters,
		Iter:             1,
	}
}

// forward propagation step
// matrix shape (nFeatures, nSamples)
func (dl *denseLayer) ForwardPropagation(x *mat.Dense) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	L := len(dl.Parameters) / 2      // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	A[strconv.Itoa(0)] = x

	applyActivationFunction := func(_, _ int, v float64) float64 { return dl.Activation.Function(v) }
	for l := 0; l < L-1; l++ {
		W := dl.Parameters["W"+strconv.Itoa(l+1)] // weights W
		b := dl.Parameters["b"+strconv.Itoa(l+1)] // biases b

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(W, A[strconv.Itoa(l)]), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = ngo.Apply(applyActivationFunction, Z[strconv.Itoa(l+1)])  // compute the non linear operation
	}
	// for output layer
	Z[strconv.Itoa(L)] = ngo.AddMatrixVector(ngo.MatMul(dl.Parameters["W"+strconv.Itoa(L)], A[strconv.Itoa(L-1)]), dl.Parameters["b"+strconv.Itoa(L)])
	A[strconv.Itoa(L)] = dl.OutputActivation.Function(Z[strconv.Itoa(L)])

	// prediction
	yHat := A[strconv.Itoa(L)]

	return yHat, Z, A
}

// backward propagation step
func (dl *denseLayer) BackwardPropagation(Z, A map[string]*mat.Dense, y *mat.Dense, learningRate, lambd float64) *mat.Dense {
	m := y.RawMatrix().Cols     // number of training examples
	L := len(dl.Parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	dZ[strconv.Itoa(L)] = ngo.Scale(1./float64(m), ngo.Sub(A[strconv.Itoa(L)], y))
	dW[strconv.Itoa(L)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(L)], A[strconv.Itoa(L-1)].T()), ngo.Scale(lambd/float64(m), dl.Parameters["W"+strconv.Itoa(L)]))
	db[strconv.Itoa(L)] = ngo.Sum(dZ[strconv.Itoa(L)], ngo.OverColumns)

	applyActivationFunctionDerivative := func(_, _ int, v float64) float64 { return dl.Activation.Derivative(v) }
	for l := L - 1; l > 0; l-- {
		dA[strconv.Itoa(l)] = ngo.MatMul(dl.Parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])
		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], ngo.Apply(applyActivationFunctionDerivative, Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T()), ngo.Scale(lambd/float64(m), dl.Parameters["W"+strconv.Itoa(l)]))
		db[strconv.Itoa(l)] = ngo.Sum(dZ[strconv.Itoa(l)], ngo.OverColumns)
	}

	// update parameters (optimization algorithm)
	dl.Parameters = dl.Optimizer.Function(&dl.Optimizer, dl.Parameters, dW, db, learningRate, dl.Iter)
	dl.Iter++

	dA[strconv.Itoa(0)] = ngo.MatMul(dl.Parameters["W"+strconv.Itoa(1)].T(), dZ[strconv.Itoa(1)])
	
	return dA[strconv.Itoa(0)]
}

// initializing the model parameters
func initializeParameters(nnStructure []int) map[string]*mat.Dense {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(nnStructure) - 1                 // number of layers

	for l := 0; l < L; l++ {
		scalar := math.Sqrt((6.0 / float64(nnStructure[l]+nnStructure[l+1])))*0.1

		parameters["W"+strconv.Itoa(l+1)] = ngo.Scale(scalar, ngo.Randn(nnStructure[l+1], nnStructure[l]))
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(nnStructure[l+1], 1, nil)
	}

	return parameters
}
