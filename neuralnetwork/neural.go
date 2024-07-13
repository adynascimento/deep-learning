package neuralnetwork

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/olekukonko/tablewriter"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type NeuralNetwork interface {
	NewTrainer(config TrainerConfig, options ...func(*neuralModel)) NeuralModel
}

type NeuralModel interface {
	Fit(xTrain *mat.Dense, yTrain *mat.Dense, printLoss bool) []float64
	Predict(x *mat.Dense) *mat.Dense
	Evaluate(x *mat.Dense, y *mat.Dense) float64
	Save(path string)
	Summary()
}

type NeuralConfig struct {
	NNStructure []int
	Activation  activationType
	Mode        modeType
}

type TrainerConfig struct {
	Optimizer    optimizerType
	LearningRate float64
	NIterations  int
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
	BatchSize        int
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

func (nn *neuralNetwork) NewTrainer(config TrainerConfig, options ...func(*neuralModel)) NeuralModel {
	// choice of optimization algorithm
	optimizer := optimizerSettings[config.Optimizer]
	if config.Optimizer == AdamOptimizer {
		optimizer.Adam = initializeAdam(nn.Parameters)
	}

	model := neuralModel{
		neuralNetwork: nn,
		Optimizer:     optimizer,
		LearningRate:  config.LearningRate,
		NIterations:   config.NIterations,
	}

	// apply additional options
	for _, option := range options {
		option(&model)
	}

	return &model
}

// forward propagation step
func (nm *neuralModel) ForwardPropagation(x *mat.Dense) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	L := len(nm.Parameters) / 2      // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	A[strconv.Itoa(0)] = x

	applyActivationFunction := func(_, _ int, v float64) float64 { return nm.Activation.Function(v) }
	for l := 0; l < L-1; l++ {
		W := nm.Parameters["W"+strconv.Itoa(l+1)] // weights W
		b := nm.Parameters["b"+strconv.Itoa(l+1)] // biases b

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(W, A[strconv.Itoa(l)]), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = ngo.Apply(applyActivationFunction, Z[strconv.Itoa(l+1)])  // compute the non linear operation
	}
	// for output layer
	Z[strconv.Itoa(L)] = ngo.AddMatrixVector(ngo.MatMul(nm.Parameters["W"+strconv.Itoa(L)],
		A[strconv.Itoa(L-1)]), nm.Parameters["b"+strconv.Itoa(L)])
	A[strconv.Itoa(L)] = nm.OutputActivation.Function(Z[strconv.Itoa(L)])

	// prediction
	yHat := A[strconv.Itoa(L)]

	return yHat, Z, A
}

// backward propagation step
func (nm *neuralModel) BackwardPropagation(Z, A map[string]*mat.Dense, y *mat.Dense) (map[string]*mat.Dense, map[string]*mat.Dense) {
	m := y.RawMatrix().Cols     // number of training examples
	L := len(nm.Parameters) / 2 // number of layers

	dZ := make(map[string]*mat.Dense) // derivatives of the linear function Z
	dW := make(map[string]*mat.Dense) // derivatives of the weigths W
	db := make(map[string]*mat.Dense) // derivatives of the biases b
	dA := make(map[string]*mat.Dense) // derivatives of the activation function A

	dZ[strconv.Itoa(L)] = ngo.Scale(1./float64(m), ngo.Sub(A[strconv.Itoa(L)], y))
	dW[strconv.Itoa(L)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(L)], A[strconv.Itoa(L-1)].T()),
		ngo.Scale(nm.L2Regularization/float64(m), nm.Parameters["W"+strconv.Itoa(L)]))
	db[strconv.Itoa(L)] = ngo.Sum(dZ[strconv.Itoa(L)], ngo.OverColumns)

	applyActivationFunctionDerivative := func(_, _ int, v float64) float64 { return nm.Activation.Derivative(v) }
	for l := L - 1; l > 0; l-- {
		dA[strconv.Itoa(l)] = ngo.MatMul(nm.Parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])
		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], ngo.Apply(applyActivationFunctionDerivative, Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T()),
			ngo.Scale(nm.L2Regularization/float64(m), nm.Parameters["W"+strconv.Itoa(l)]))
		db[strconv.Itoa(l)] = ngo.Sum(dZ[strconv.Itoa(l)], ngo.OverColumns)
	}

	return dW, db
}

// performs model training with the xTrain and yTrain matrices,
// which is represented as an rows X cols matrix a where each
// row is a variable and each column is an observation.
// matrix shape (nFeatures, nSamples)
func (nm *neuralModel) Fit(xTrain, yTrain *mat.Dense, printLoss bool) []float64 {
	nSamples := xTrain.RawMatrix().Cols
	if nm.BatchSize == 0 {
		nm.BatchSize = nSamples
	}

	// keep track of the loss
	losses := []float64{}

	// loop
	start := time.Now()
	for i := 1; i <= nm.NIterations; i++ {
		lossBatches := []float64{}

		for startIdx := 0; startIdx < nSamples; startIdx += nm.BatchSize {
			endIdx := startIdx + nm.BatchSize
			if endIdx > nSamples {
				endIdx = nSamples
			}

			xBatch := xTrain.Slice(0, xTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)
			yBatch := yTrain.Slice(0, yTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)

			// forward propagation
			yHat, Z, A := nm.ForwardPropagation(xBatch)

			// loss function
			loss := nm.LossFunction(yHat, yBatch, nm.Parameters, nm.L2Regularization)
			lossBatches = append(lossBatches, loss)

			// backward propagation
			dW, db := nm.BackwardPropagation(Z, A, yBatch)

			// update parameters (optimization algorithm)
			nm.Parameters = nm.Optimizer.Function(&nm.Optimizer, nm.Parameters, dW, db,
				nm.LearningRate, float64(i))
		}

		// print the loss every x iterations
		meanLoss := stat.Mean(lossBatches, nil)
		if printLoss && i%(nm.NIterations/10) == 0 || printLoss && i == 1 {
			if nm.Mode == ModeRegression {
				fmt.Printf("iter %6d/%d: | t: %5.2fs | loss: %.6e \n", i, nm.NIterations, time.Since(start).Seconds(), meanLoss)
			} else {
				fmt.Printf("iter %6d/%d: | t: %5.2fs | loss: %.6e | acc: %.4f \n", i, nm.NIterations,
					time.Since(start).Seconds(), meanLoss, nm.Evaluate(xTrain, yTrain))
			}
		}
		losses = append(losses, meanLoss)
	}

	return losses
}

// predictions with forward propagation
func (nm *neuralModel) Predict(x *mat.Dense) *mat.Dense {
	predictions, _, _ := nm.ForwardPropagation(x)
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

	case ModeMultiLabel:
		// hamming accuracy
		for j := 0; j < y.RawMatrix().Cols; j++ {
			correctLabels := 0.0
			for i, pred := range mat.Col(nil, j, yPred) {
				// round considers the threshold 0.5
				if y.At(i, j) == math.Round(pred) {
					correctLabels++
				}
			}
			metric += correctLabels / float64(len(mat.Col(nil, j, yPred)))
		}
		metric = (metric / float64(y.RawMatrix().Cols))
	}

	return metric
}

// model summary
func (nm *neuralModel) Summary() {
	data := [][]string{}
	for i, v := range nm.NNStructure[1:] {
		data = append(data, []string{
			fmt.Sprintf("Dense Layer %d", i+1), fmt.Sprintf("(None, %d)", v), fmt.Sprintf("%d",
				nm.NNStructure[i]*nm.NNStructure[i+1]+nm.NNStructure[i+1]),
		})
	}

	// table configuration
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Layer (type)", "Output Shape", "Param #"})
	table.SetCenterSeparator("|")
	table.AppendBulk(data)
	table.Render()
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

func WithBatchSize(batchSize int) func(*neuralModel) {
	return func(nm *neuralModel) {
		nm.BatchSize = batchSize
	}
}

func WithL2Regularization(lambd float64) func(*neuralModel) {
	return func(nm *neuralModel) {
		nm.L2Regularization = lambd
	}
}
