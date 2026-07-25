package nncore

import (
	"math"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

var ActivationSettings = map[ActivationType]Activation{
	TanhActivation: {
		Name:       TanhActivation,
		Function:   Tanh,
		Derivative: TanhDerivative,
	},
	SigmoidActivation: {
		Name:       SigmoidActivation,
		Function:   Sigmoid,
		Derivative: SigmoidDerivative,
	},
	EluActivation: {
		Name:       EluActivation,
		Function:   Elu,
		Derivative: EluDerivative,
	},
	ReLUActivation: {
		Name:       ReLUActivation,
		Function:   ReLU,
		Derivative: ReLUDerivative,
	},
}

var ModeSettings = map[ModeType]ConfigMode{
	ModeRegression: {
		OutputActivation: OutputActivation{
			Mode:     ModeRegression,
			Function: ApplyLinear,
		},
		LossFunction: MeanSquaredError,
	},
	ModeMultiClass: {
		OutputActivation: OutputActivation{
			Mode:     ModeMultiClass,
			Function: ApplySoftmax,
		},
		LossFunction: CrossEntropy,
	},
	ModeMultiLabel: {
		OutputActivation: OutputActivation{
			Mode:     ModeMultiLabel,
			Function: ApplySigmoid,
		},
		LossFunction: BinaryCrossEntropy,
	},
}

type ConfigMode struct {
	OutputActivation OutputActivation
	LossFunction     LossFunction
}

type ActivationType string
type ActivationFunction func(*mat.Dense) *mat.Dense

type ModeType string
type OutputActivationFunction func(*mat.Dense) *mat.Dense

const (
	TanhActivation    ActivationType = "tanh"
	SigmoidActivation ActivationType = "sigmoid"
	EluActivation     ActivationType = "elu"
	ReLUActivation    ActivationType = "relu"

	ModeRegression ModeType = "regression" // linear output with mse loss
	ModeMultiClass ModeType = "multiclass" // softmax output with cross entropy loss
	ModeMultiLabel ModeType = "multilabel" // sigmoid output with binary cross entropy loss
)

type Activation struct {
	Name       ActivationType
	Function   ActivationFunction
	Derivative ActivationFunction
}

// implements the Tanh function for use in activation functions.
func Tanh(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			data[i] = math.Tanh(data[i])
		}
	})
}

// implements the derivative of the Tanh function for backpropagation.
func TanhDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			t := math.Tanh(data[i])
			data[i] = 1.0 - t*t
		}
	})
}

// implements the sigmoid function for use in activation functions.
func Sigmoid(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			data[i] = SigmoidScalar(data[i])
		}
	})
}

// implements the derivative of the sigmoid function for backpropagation.
func SigmoidDerivative(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			s := SigmoidScalar(data[i])
			data[i] = s * (1.0 - s)
		}
	})
}

// implements the elu function for use in activation functions.
func Elu(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] <= 0 {
				data[i] = math.Exp(data[i]) - 1.0
			}
		}
	})
}

// implements the derivative of the elu function for backpropagation.
func EluDerivative(a *mat.Dense) *mat.Dense {
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
func ReLU(a *mat.Dense) *mat.Dense {
	return ngo.ApplyRaw(a, func(data []float64) {
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			}
		}
	})
}

// implements the derivative of the relu function for backpropagation.
func ReLUDerivative(a *mat.Dense) *mat.Dense {
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

type OutputActivation struct {
	Mode     ModeType
	Function OutputActivationFunction
}

// applies linear function for output layer
func ApplyLinear(a *mat.Dense) *mat.Dense {
	return ngo.Apply(func(_, _ int, v float64) float64 { return v }, a)
}

// applies softmax function for output layer
func ApplySoftmax(a *mat.Dense) *mat.Dense {
	exp := ngo.Apply(func(_, _ int, v float64) float64 { return math.Exp(v) }, a)
	sum := ngo.Sum(exp, ngo.OverRows)

	return ngo.DivMatrixVector(exp, sum)
}

// applies sigmoid function for output layer
func ApplySigmoid(a *mat.Dense) *mat.Dense {
	return ngo.Apply(func(_, _ int, v float64) float64 { return SigmoidScalar(v) }, a)
}

// implements the sigmoid function for use in activation functions.
func SigmoidScalar(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
