package nncore

import (
	"bytes"
	"encoding/gob"
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type DenseConfig struct {
	NNStructure      []int
	Activation       Activation
	OutputActivation OutputActivation
	Optimizer        OptimizerType
	L2Regularization float64
	Dropout          float64
	RNG              *RNG
}

type Dense struct {
	Activation       Activation
	OutputActivation OutputActivation
	Optimizer        Optimizer
	Parameters       map[string]*mat.Dense
	Iter             float64
	L2Regularization float64
	Dropout          float64
	RNG              *RNG
}

func NewDense(config DenseConfig) *Dense {
	// initializing the model parameters
	parameters := initializeDenseParameters(config.NNStructure, config.Activation, config.RNG)

	// choice of optimization algorithm
	optimizer := NewOptimizer(config.Optimizer)

	return &Dense{
		Activation:       config.Activation,
		OutputActivation: config.OutputActivation,
		Optimizer:        optimizer,
		Parameters:       parameters,
		Iter:             1,
		L2Regularization: config.L2Regularization,
		Dropout:          config.Dropout,
		RNG:              config.RNG.Spawn(),
	}
}

// initializing the model parameters
func initializeDenseParameters(nnStructure []int, activation Activation, rng *RNG) map[string]*mat.Dense {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(nnStructure) - 1                 // number of layers

	for l := 0; l < L; l++ {
		fanIn := nnStructure[l]
		fanOut := nnStructure[l+1]

		scalar := 1.0
		if l == L-1 {
			scalar = math.Sqrt(2.0 / float64(fanIn+fanOut)) // Xavier (Glorot)
		} else {
			switch activation.Name {
			case ReLUActivation, EluActivation:
				scalar = math.Sqrt(2.0 / float64(fanIn)) // He (Kaiming)
			default: // tanh, sigmoid, softmax, linear...
				scalar = math.Sqrt(2.0 / float64(fanIn+fanOut)) // Xavier (Glorot)
			}
		}

		parameters["W"+strconv.Itoa(l+1)] = ngo.Scale(scalar, rng.Randn(fanIn, fanOut))
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(1, fanOut, nil)
	}

	return parameters
}

// forward propagation step
func (dn *Dense) ForwardPropagation(x *mat.Dense, training bool) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, map[string][]bool) {
	L := len(dn.Parameters) / 2      // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	D := make(map[string][]bool)     // dropout masks (training only)
	A[strconv.Itoa(0)] = x

	for l := 0; l < L-1; l++ {
		W := dn.Parameters["W"+strconv.Itoa(l+1)] // weights W
		b := dn.Parameters["b"+strconv.Itoa(l+1)] // biases b

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(A[strconv.Itoa(l)], W), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = dn.Activation.Function(Z[strconv.Itoa(l+1)])              // compute the non linear operation

		// apply dropout
		if training && dn.Dropout > 0 {
			D[strconv.Itoa(l+1)] = DropoutMask(A[strconv.Itoa(l+1)], dn.Dropout, dn.RNG)
			ApplyDropoutMask(A[strconv.Itoa(l+1)], D[strconv.Itoa(l+1)], dn.Dropout)
		}
	}

	// for output layer
	Z[strconv.Itoa(L)] = ngo.AddMatrixVector(ngo.MatMul(A[strconv.Itoa(L-1)], dn.Parameters["W"+strconv.Itoa(L)]), dn.Parameters["b"+strconv.Itoa(L)])
	A[strconv.Itoa(L)] = dn.OutputActivation.Function(Z[strconv.Itoa(L)])

	// prediction
	yHat := A[strconv.Itoa(L)]

	return yHat, Z, A, D
}

// backward propagation step
func (dn *Dense) BackwardPropagation(Z, A map[string]*mat.Dense, D map[string][]bool, y *mat.Dense) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	m := y.RawMatrix().Rows     // number of training examples
	L := len(dn.Parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	dZ[strconv.Itoa(L)] = ngo.Scale(1./float64(m), ngo.Sub(A[strconv.Itoa(L)], y))
	dW[strconv.Itoa(L)] = ngo.Add(ngo.MatMul(A[strconv.Itoa(L-1)].T(), dZ[strconv.Itoa(L)]), ngo.Scale(dn.L2Regularization/float64(m), dn.Parameters["W"+strconv.Itoa(L)]))
	db[strconv.Itoa(L)] = ngo.Sum(dZ[strconv.Itoa(L)], ngo.OverRows)

	for l := L - 1; l > 0; l-- {
		dA[strconv.Itoa(l)] = ngo.MatMul(dZ[strconv.Itoa(l+1)], dn.Parameters["W"+strconv.Itoa(l+1)].T())

		// apply dropout
		if dn.Dropout > 0 {
			ApplyDropoutMask(dA[strconv.Itoa(l)], D[strconv.Itoa(l)], dn.Dropout)
		}

		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], dn.Activation.Derivative(Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.Add(ngo.MatMul(A[strconv.Itoa(l-1)].T(), dZ[strconv.Itoa(l)]), ngo.Scale(dn.L2Regularization/float64(m), dn.Parameters["W"+strconv.Itoa(l)]))
		db[strconv.Itoa(l)] = ngo.Sum(dZ[strconv.Itoa(l)], ngo.OverRows)
	}

	dA[strconv.Itoa(0)] = ngo.MatMul(dZ[strconv.Itoa(1)], dn.Parameters["W"+strconv.Itoa(1)].T())

	return dA[strconv.Itoa(0)], dW, db
}

// update parameters (optimization algorithm)
func (dn *Dense) UpdateParameters(dW, db map[string]*mat.Dense, learningRate float64) {
	dn.Optimizer.Step(dn.TrainableParameters(dW, db), learningRate, dn.Iter)
	dn.Iter++
}

func (dn *Dense) TrainableParameters(dW, db map[string]*mat.Dense) []*Parameter {
	L := len(dn.Parameters) / 2 // number of layers

	params := []*Parameter{}
	for l := 0; l < L; l++ {
		params = append(params, &Parameter{
			Value:    dn.Parameters["W"+strconv.Itoa(l+1)],
			Gradient: dW[strconv.Itoa(l+1)],
			Update: func(m *mat.Dense) {
				dn.Parameters["W"+strconv.Itoa(l+1)] = m
			},
		})

		params = append(params, &Parameter{
			Value:    dn.Parameters["b"+strconv.Itoa(l+1)],
			Gradient: db[strconv.Itoa(l+1)],
			Update: func(m *mat.Dense) {
				dn.Parameters["b"+strconv.Itoa(l+1)] = m
			},
		})
	}

	return params
}

func (dn *Dense) MarshalParameters() ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(dn.Parameters); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (dn *Dense) UnmarshalParameters(data []byte) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(&dn.Parameters)
}
