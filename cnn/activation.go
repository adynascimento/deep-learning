package cnn

import (
	"math"

	"github.com/adynascimento/deep-learning/ngo"

	"gonum.org/v1/gonum/mat"
)


var activationSettings = map[activationType]activation{
	TanhActivation: {
		Name:       TanhActivation,
		Function:   tanh,
		Derivative: tanhDerivative,
	},
	SigmoidActivation: {
		Name:       SigmoidActivation,
		Function:   sigmoid,
		Derivative: sigmoidDerivative,
	},
	EluActivation: {
		Name:       EluActivation,
		Function:   elu,
		Derivative: eluDerivative,
	},
	ReLUActivation: {
		Name:       ReLUActivation,
		Function:   relu,
		Derivative: reluDerivative,
	},
}

type configMode struct {
	outputActivation outputActivation
	lossFunction     lossFunction
}

var modeSettings = map[modeType]configMode{
	ModeMultiClass: {
		outputActivation: outputActivation{
			Mode:     ModeMultiClass,
			Function: applySoftmax,
		},
		lossFunction: crossEntropy,
	},
	ModeMultiLabel: {
		outputActivation: outputActivation{
			Mode:     ModeMultiLabel,
			Function: applySigmoid,
		},
		lossFunction: binaryCrossEntropy,
	},
}

type activationType string
type activationFunction func(*mat.Dense) *mat.Dense

type modeType string
type outputActivationFunction func(*mat.Dense) *mat.Dense

const (
	TanhActivation    activationType = "tanh"
	SigmoidActivation activationType = "sigmoid"
	EluActivation     activationType = "elu"
	ReLUActivation    activationType = "relu"

	ModeMultiClass modeType = "multiclass" // softmax output with cross entropy loss
	ModeMultiLabel modeType = "multilabel" // sigmoid output with binary cross entropy loss
)

type activation struct {
	Name       activationType
	Function   activationFunction
	Derivative activationFunction
}

// implements the Tanh function for use in activation functions.
func tanh(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			data[i] = math.Tanh(data[i])
		}
	})
}

// implements the derivative of the Tanh function for backpropagation.
func tanhDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			t := math.Tanh(data[i])
			data[i] = 1.0 - t*t
		}
	})
}

// implements the sigmoid function for use in activation functions.
func sigmoid(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			data[i] = 1.0 / (1.0 + math.Exp(-data[i]))
		}
	})
}

// implements the derivative of the sigmoid function for backpropagation.
func sigmoidDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			s := 1.0 / (1.0 + math.Exp(-data[i]))
			data[i] = s * (1.0 - s)
		}
	})
}

// implements the elu function for use in activation functions.
func elu(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] <= 0 {
				data[i] = math.Exp(data[i]) - 1.0
			}
		}
	})
}

// implements the derivative of the elu function for backpropagation.
func eluDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] < 0 {
				data[i] = math.Exp(data[i])
			} else {
				data[i] = 1.0
			}
		}
	})
}

// implements the relu function for use in activation functions.
func relu(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			}
		}
	})
}

// implements the derivative of the relu function for backpropagation.
func reluDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] > 0 {
				data[i] = 1
			} else {
				data[i] = 0
			}
		}
	})
}

type outputActivation struct {
	Mode     modeType
	Function outputActivationFunction
}

// applies softmax function for output layer
func applySoftmax(a *mat.Dense) *mat.Dense {
	applyExp := func(_, _ int, v float64) float64 { return math.Exp(v) }
	exp := ngo.Apply(applyExp, a)
	sum := ngo.Sum(exp, ngo.OverRows)

	return ngo.DivMatrixVector(exp, sum)
}

// applies sigmoid function for output layer
func applySigmoid(a *mat.Dense) *mat.Dense {
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoidActivation(v) }
	sigmoid := ngo.Apply(applySigmoid, a)

	return sigmoid
}

// implements the sigmoid function for use in activation functions.
func sigmoidActivation(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
