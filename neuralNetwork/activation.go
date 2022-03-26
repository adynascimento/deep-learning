package neuralNetwork

import "math"

type activationType string

const (
	ActivationTanh    activationType = "tanh"
	ActivationSigmoid activationType = "sigmoid"
	ActivationElu     activationType = "elu"
)

type activationFunction func(float64) float64

type activation struct {
	name       activationType
	function   activationFunction
	derivative activationFunction
}

// implements the Tanh function for use in activation functions.
func tanhActivation(x float64) float64 {
	return math.Tanh(x)
}

// implements the derivative of the Tanh function for backpropagation.
func tanhPrimeActivation(x float64) float64 {
	return 1.0 - tanhActivation(x)*tanhActivation(x)
}

// implements the Tanh function for use in activation functions.
func sigmoidActivation(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// implements the derivative of the Tanh function for backpropagation.
func sigmoidPrimeActivation(x float64) float64 {
	return sigmoidActivation(x) * (1.0 - sigmoidActivation(x))
}

// implements the Tanh function for use in activation functions.
func eluActivation(x float64) float64 {
	var out float64
	if x <= 0 {
		out = math.Exp(x) - 1.0
	} else {
		out = x
	}
	return out
}

// implements the derivative of the Tanh function for backpropagation.
func eluPrimeActivation(x float64) float64 {
	var out float64
	if x < 0 {
		out = math.Exp(x)
	} else {
		out = 1.0
	}
	return out
}
