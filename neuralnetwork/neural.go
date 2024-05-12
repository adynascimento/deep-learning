package neuralnetwork

import (
	"fmt"
	"math"
	"strconv"
	"time"

	ngo "github.com/adynascimento/deep-learning/numeric"

	"gonum.org/v1/gonum/mat"
)

type NeuralConfig struct {
	NNStructure []int
	Activation  activationType
	Mode        modeType
}

type NeuralNetwork struct {
	NNStructure      []int
	Activation       activation
	OutputActivation outputActivation
	LossFunction     lossFunction
	Parameters       map[string]*mat.Dense
}

func NewNeuralNetwork(config NeuralConfig) NeuralNetwork {
	// choice of activation function
	activationFunction := activation{}
	switch config.Activation {
	case ActivationTanh:
		activationFunction = activation{
			Name:       config.Activation,
			Function:   tanhActivation,
			Derivative: tanhActivationDerivative,
		}
	case ActivationSigmoid:
		activationFunction = activation{
			Name:       config.Activation,
			Function:   sigmoidActivation,
			Derivative: sigmoidActivationDerivative,
		}
	case ActivationElu:
		activationFunction = activation{
			Name:       config.Activation,
			Function:   eluActivation,
			Derivative: eluActivationDerivative,
		}
	}

	// choice of output layer activation function and loss
	var lossFunction lossFunction
	outputActivationFunction := outputActivation{}
	switch config.Mode {
	case ModeRegression:
		outputActivationFunction = outputActivation{
			Mode:     config.Mode,
			Function: applyLinear,
		}
		lossFunction = meanSquareError
	case ModeMultiClass:
		outputActivationFunction = outputActivation{
			Mode:     config.Mode,
			Function: applySoftmax,
		}
		lossFunction = crossEntropy
	case ModeMultiLabel:
		outputActivationFunction = outputActivation{
			Mode:     config.Mode,
			Function: applySigmoid,
		}
		lossFunction = crossEntropy
	case ModeBinary:
		outputActivationFunction = outputActivation{
			Mode:     config.Mode,
			Function: applySigmoid,
		}
		lossFunction = binaryCrossEntropy
	}

	// initializing the model parameters
	parameters := initializeParameters(config.NNStructure)

	return NeuralNetwork{
		NNStructure:      config.NNStructure,
		Activation:       activationFunction,
		OutputActivation: outputActivationFunction,
		LossFunction:     lossFunction,
		Parameters:       parameters,
	}
}

type TrainerConfig struct {
	Optimizer        optimizerType
	LearningRate     float64
	L2Regularization float64
	NIterations      int
}

type NeuralModel struct {
	NeuralNetwork
	Optimizer        optimizer
	LearningRate     float64
	L2Regularization float64
	NIterations      int
}

func NewTrainer(neural NeuralNetwork, config TrainerConfig) NeuralModel {
	// choice of optimization algorithm
	optimizer := optimizer{}
	switch config.Optimizer {
	case GradientDescentOptimizer:
		optimizer.Name = config.Optimizer
		optimizer.Function = optimizer.GradientDescentOptimizer
	case AdamOptimizer:
		optimizer.Name = config.Optimizer
		optimizer.Function = optimizer.AdamOptimizer

		v, s := initializeAdam(neural.Parameters)
		optimizer.Adam = adamParameters{v: v, s: s}
	}

	return NeuralModel{
		NeuralNetwork:    neural,
		Optimizer:        optimizer,
		LearningRate:     config.LearningRate,
		L2Regularization: config.L2Regularization,
		NIterations:      config.NIterations,
	}
}

// initializing the model parameters
func initializeParameters(nnStructure []int) map[string]*mat.Dense {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(nnStructure) - 1                 // number of layers

	for l := 0; l < L; l++ {
		scalar := math.Sqrt((6.0 / float64(nnStructure[l]+nnStructure[l+1])))

		parameters["W"+strconv.Itoa(l+1)] = ngo.Scale(scalar, ngo.Randn(nnStructure[l+1], nnStructure[l]))
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(nnStructure[l+1], 1, nil)
	}

	return parameters
}

// forward propagation step
func forwardPropagation(parameters map[string]*mat.Dense, x *mat.Dense,
	activation activationFunction, outputActivation outputActivationFunction) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	L := len(parameters) / 2         // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	A[strconv.Itoa(0)] = x

	applyActivationFunction := func(_, _ int, v float64) float64 { return activation(v) }
	for l := 0; l < L-1; l++ {
		W := parameters["W"+strconv.Itoa(l+1)] // weights W
		b := parameters["b"+strconv.Itoa(l+1)] // biases b

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(W, A[strconv.Itoa(l)]), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = ngo.Apply(applyActivationFunction, Z[strconv.Itoa(l+1)])  // compute the non linear operation
	}
	// for output layer
	Z[strconv.Itoa(L)] = ngo.AddMatrixVector(ngo.MatMul(parameters["W"+strconv.Itoa(L)], A[strconv.Itoa(L-1)]), parameters["b"+strconv.Itoa(L)])
	A[strconv.Itoa(L)] = outputActivation(Z[strconv.Itoa(L)])

	// prediction
	yHat := A[strconv.Itoa(L)]

	return yHat, Z, A
}

// backward propagation step
func backwardPropagation(parameters, Z, A map[string]*mat.Dense, y *mat.Dense, derivative activationFunction, lambd float64) (map[string]*mat.Dense, map[string]*mat.Dense) {
	m := y.RawMatrix().Cols  // number of training examples
	L := len(parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	dZ[strconv.Itoa(L)] = ngo.Scale(1./float64(m), ngo.Sub(A[strconv.Itoa(L)], y))
	dW[strconv.Itoa(L)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(L)], A[strconv.Itoa(L-1)].T()), ngo.Scale(lambd/float64(m), parameters["W"+strconv.Itoa(L)]))
	db[strconv.Itoa(L)] = ngo.Sum(dZ[strconv.Itoa(L)], ngo.OverColumns)

	applyActivationFunctionDerivative := func(_, _ int, v float64) float64 { return derivative(v) }
	for l := L - 1; l > 0; l-- {
		dA[strconv.Itoa(l)] = ngo.MatMul(parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])
		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], ngo.Apply(applyActivationFunctionDerivative, Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T()), ngo.Scale(lambd/float64(m), parameters["W"+strconv.Itoa(l)]))
		db[strconv.Itoa(l)] = ngo.Sum(dZ[strconv.Itoa(l)], ngo.OverColumns)
	}

	return dW, db
}

// train model
func (n *NeuralModel) Fit(xTrain, yTrain *mat.Dense, printLoss bool) []float64 {
	// keep track of the loss
	losses := []float64{}
	start := time.Now()

	// loop
	for i := 1; i <= n.NIterations; i++ {
		// forward propagation
		yHat, Z, A := forwardPropagation(n.Parameters, xTrain, n.Activation.Function, n.OutputActivation.Function)

		// loss function
		loss := n.LossFunction(yHat, yTrain, n.Parameters, n.L2Regularization)

		// backward propagation
		dW, db := backwardPropagation(n.Parameters, Z, A, yTrain, n.Activation.Derivative, n.L2Regularization)

		// update parameters (optimization algorithm)
		n.Parameters = n.Optimizer.Function(n.Parameters, dW, db, n.LearningRate, float64(i))

		// print the loss every 1000 iterations
		losses = append(losses, loss)
		if printLoss && i%100 == 0 || printLoss && i == 1 {
			fmt.Printf("it %d: | t: %.2fs | loss: %e \n", i, time.Since(start).Seconds(), loss)
		}
	}

	return losses
}

// predictions
func (n *NeuralModel) Predict(x *mat.Dense) *mat.Dense {
	// forward propagation
	predictions, _, _ := forwardPropagation(n.Parameters, x, n.Activation.Function, n.OutputActivation.Function)

	return predictions
}
