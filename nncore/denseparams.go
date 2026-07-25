package nncore

import (
	"math"
	"strconv"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

// initializing the model parameters
func InitializeDenseParameters(nnStructure []int, activation Activation) map[string]*mat.Dense {
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

		parameters["W"+strconv.Itoa(l+1)] = ngo.Scale(scalar, ngo.Randn(nnStructure[l+1], nnStructure[l]))
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(nnStructure[l+1], 1, nil)
	}

	return parameters
}
