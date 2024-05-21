package neuralnetwork

import (
	"fmt"
	"math"
	"strconv"
	"time"

	ngo "github.com/adynascimento/deep-learning/gonum"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork interface {
	NewTrainer(config TrainerConfig) NeuralModel
}

type NeuralModel interface {
	Fit(xTrain *mat.Dense, yTrain *mat.Dense, printLoss bool) []float64
	Predict(x *mat.Dense) *mat.Dense
	Evaluate(x *mat.Dense, y *mat.Dense) float64
	Save(path string)
}

type NeuralConfig struct {
	NNStructure []int
	Activation  activationType
	Mode        modeType
}

type TrainerConfig struct {
	Optimizer        optimizerType
	LearningRate     float64
	L2Regularization float64
	NIterations      int
}

type neuralNetwork struct {
	NNStructure      []int
	Activation       activation
	Mode             modeType
	OutputActivation outputActivation
	LossFunction     lossFunction
	Parameters       map[string]*mat.Dense
}

type neuralModel struct {
	*neuralNetwork
	Optimizer        optimizer
	LearningRate     float64
	L2Regularization float64
	NIterations      int
}

func NewNeuralNetwork(config NeuralConfig) NeuralNetwork {
	// choice of activation function
	activationFunction := activationSettings[config.Activation]

	// choice of output layer activation function and loss function
	lossFunction := modeSettings[config.Mode].lossFunction
	outputActivationFunction := modeSettings[config.Mode].outputActivation

	// initializing the model parameters
	parameters := initializeParameters(config.NNStructure)

	return &neuralNetwork{
		NNStructure:      config.NNStructure,
		Activation:       activationFunction,
		Mode:             config.Mode,
		OutputActivation: outputActivationFunction,
		LossFunction:     lossFunction,
		Parameters:       parameters,
	}
}

func (nn *neuralNetwork) NewTrainer(config TrainerConfig) NeuralModel {
	// choice of optimization algorithm
	optimizer := optimizerSettings[config.Optimizer]
	if config.Optimizer == AdamOptimizer {
		optimizer.Adam = initializeAdam(nn.Parameters)
	}

	return &neuralModel{
		neuralNetwork:    nn,
		Optimizer:        optimizer,
		LearningRate:     config.LearningRate,
		L2Regularization: config.L2Regularization,
		NIterations:      config.NIterations,
	}
}

// train model
func (nm *neuralModel) Fit(xTrain, yTrain *mat.Dense, printLoss bool) []float64 {
	// keep track of the loss
	start := time.Now()
	losses := []float64{}

	// loop
	for i := 1; i <= nm.NIterations; i++ {
		// forward propagation
		yHat, Z, A := forwardPropagation(nm.Parameters, xTrain, nm.Activation.Function,
			nm.OutputActivation.Function)

		// loss function
		loss := nm.LossFunction(yHat, yTrain, nm.Parameters, nm.L2Regularization)

		// backward propagation
		dW, db := backwardPropagation(nm.Parameters, Z, A, yTrain, nm.Activation.Derivative, nm.L2Regularization)

		// update parameters (optimization algorithm)
		nm.Parameters = nm.Optimizer.Function(&nm.Optimizer, nm.Parameters, dW, db,
			nm.LearningRate, float64(i))

		// print the loss every x iterations
		losses = append(losses, loss)
		if printLoss && i%(nm.NIterations/10) == 0 || printLoss && i == 1 {
			if nm.Mode == ModeRegression {
				fmt.Printf("iter %6d/%d: | t: %5.2fs | loss: %.6e \n", i, nm.NIterations, time.Since(start).Seconds(), loss)
			} else {
				fmt.Printf("iter %6d/%d: | t: %5.2fs | loss: %.6e | acc: %.4f \n", i, nm.NIterations,
					time.Since(start).Seconds(), loss, nm.Evaluate(xTrain, yTrain))
			}
		}
	}

	return losses
}

// predictions with forward propagation
func (nm *neuralModel) Predict(x *mat.Dense) *mat.Dense {
	predictions, _, _ := forwardPropagation(nm.Parameters, x, nm.Activation.Function,
		nm.OutputActivation.Function)

	return predictions
}

// evaluate model
func (nm *neuralModel) Evaluate(x, y *mat.Dense) float64 {
	yPred := nm.Predict(x)

	metric := 0.0
	switch nm.Mode {
	case ModeRegression:
		// mean squared error
		metric = mat.Sum(ngo.Square(ngo.Sub(y, yPred))) / float64(y.RawMatrix().Cols)
	case ModeMultiClass:
		// accuracy
		for j := 0; j < y.RawMatrix().Cols; j++ {
			trueClass := floats.MaxIdx(mat.Col(nil, j, y))
			predClass := floats.MaxIdx(mat.Col(nil, j, yPred))

			if trueClass == predClass {
				metric++
			}
		}
		metric = (metric / float64(y.RawMatrix().Cols))
	}

	return metric
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
