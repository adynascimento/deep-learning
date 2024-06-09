package neuralnetwork

import (
	"math"

	ngo "github.com/adynascimento/deep-learning/gonum"

	"gonum.org/v1/gonum/mat"
)

var activationSettings = map[activationType]activation{
	TanhActivation: {
		Name:       TanhActivation,
		Function:   tanhActivation,
		Derivative: tanhActivationDerivative,
	},
	SigmoidActivation: {
		Name:       SigmoidActivation,
		Function:   sigmoidActivation,
		Derivative: sigmoidActivationDerivative,
	},
	EluActivation: {
		Name:       EluActivation,
		Function:   eluActivation,
		Derivative: eluActivationDerivative,
	},
	ReLUActivation: {
		Name:       ReLUActivation,
		Function:   reluActivation,
		Derivative: reluActivationDerivative,
	},
}

type configMode struct {
	outputActivation outputActivation
	lossFunction     lossFunction
}

var modeSettings = map[modeType]configMode{
	ModeRegression: {
		outputActivation: outputActivation{
			Mode:     ModeRegression,
			Function: applyLinear,
		},
		lossFunction: meanSquaredError,
	},
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
type activationFunction func(float64) float64

type modeType string
type outputActivationFunction func(*mat.Dense) *mat.Dense

const (
	TanhActivation    activationType = "tanh"
	SigmoidActivation activationType = "sigmoid"
	EluActivation     activationType = "elu"
	ReLUActivation    activationType = "relu"

	ModeRegression modeType = "regression" // linear output with mse loss
	ModeMultiClass modeType = "multiclass" // softmax output with cross entropy loss
	ModeMultiLabel modeType = "multilabel" // sigmoid output with binary cross entropy loss
)

type activation struct {
	Name       activationType
	Function   activationFunction
	Derivative activationFunction
}

// implements the Tanh function for use in activation functions.
func tanhActivation(x float64) float64 {
	return math.Tanh(x)
}

// implements the derivative of the Tanh function for backpropagation.
func tanhActivationDerivative(x float64) float64 {
	return 1.0 - tanhActivation(x)*tanhActivation(x)
}

// implements the sigmoid function for use in activation functions.
func sigmoidActivation(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// implements the derivative of the sigmoid function for backpropagation.
func sigmoidActivationDerivative(x float64) float64 {
	return sigmoidActivation(x) * (1.0 - sigmoidActivation(x))
}

// implements the elu function for use in activation functions.
func eluActivation(x float64) float64 {
	var out float64
	if x <= 0 {
		out = math.Exp(x) - 1.0
	} else {
		out = x
	}
	return out
}

// implements the derivative of the elu function for backpropagation.
func eluActivationDerivative(x float64) float64 {
	var out float64
	if x < 0 {
		out = math.Exp(x)
	} else {
		out = 1.0
	}
	return out
}

// implements the relu function for use in activation functions.
func reluActivation(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// implements the derivative of the relu function for backpropagation.
func reluActivationDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type outputActivation struct {
	Mode     modeType
	Function outputActivationFunction
}

// applies linear function for output layer
func applyLinear(a *mat.Dense) *mat.Dense {
	applyLinear := func(_, _ int, v float64) float64 { return v }
	linear := ngo.Apply(applyLinear, a)

	return linear
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
