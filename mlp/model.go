package mlp

import (
	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork interface {
	NewTrainer(config TrainerConfig, options ...func(*neuralModel)) NeuralModel
}

type NeuralModel interface {
	// performs model training using the xTrain and yTrain matrices.
	// both matrices have shape (nFeatures, nSamples), where each row
	// corresponds to a feature and each column corresponds to a training sample.
	Fit(xTrain *mat.Dense, yTrain *mat.Dense, options ...func(*fitConfig)) []float64
	Predict(x *mat.Dense) *mat.Dense
	Evaluate(x *mat.Dense, y *mat.Dense) float64
	Save(path string)
	Summary()
}

type NeuralConfig struct {
	NNStructure []int
	Activation  nncore.ActivationType
	Mode        nncore.ModeType
}

type TrainerConfig struct {
	Optimizer    nncore.OptimizerType
	LearningRate float64
	Epochs       int
}

type neuralNetwork struct {
	NNStructure      []int
	Activation       nncore.Activation
	Mode             nncore.ModeType
	OutputActivation nncore.OutputActivation
	LossFunction     nncore.LossFunction
	Parameters       map[string]*mat.Dense
}

type neuralModel struct {
	*neuralNetwork
	Optimizer        nncore.Optimizer
	LearningRate     float64
	L2Regularization float64
	Dropout          float64
	Epochs           int
	BatchSize        int
}

type fitConfig struct {
	Verbose     bool
	LogInterval int
}

func NewNeuralNetwork(config NeuralConfig) NeuralNetwork {
	// choice of activation function
	activation := nncore.ActivationSettings[config.Activation]

	// choice of output layer activation function and loss function
	configMode := nncore.ModeSettings[config.Mode]

	// initializing the model parameters
	parameters := nncore.InitializeDenseParameters(config.NNStructure, activation)

	return &neuralNetwork{
		NNStructure:      config.NNStructure,
		Activation:       activation,
		Mode:             config.Mode,
		OutputActivation: configMode.OutputActivation,
		LossFunction:     configMode.LossFunction,
		Parameters:       parameters,
	}
}

func (nn *neuralNetwork) NewTrainer(config TrainerConfig, options ...func(*neuralModel)) NeuralModel {
	// choice of optimization algorithm
	optimizer := nncore.OptimizerSettings[config.Optimizer]
	if config.Optimizer == nncore.AdamOptimizer {
		optimizer.Adam = nncore.InitializeAdam(nn.Parameters)
	}

	model := neuralModel{
		neuralNetwork: nn,
		Optimizer:     optimizer,
		LearningRate:  config.LearningRate,
		Epochs:        config.Epochs,
		BatchSize:     32,
	}

	// apply additional options
	for _, option := range options {
		option(&model)
	}

	return &model
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

func WithDropout(dropout float64) func(*neuralModel) {
	return func(nm *neuralModel) {
		nm.Dropout = dropout
	}
}

func WithVerbose(verbose bool) func(*fitConfig) {
	return func(c *fitConfig) {
		c.Verbose = verbose
	}
}

func WithLogInterval(interval int) func(*fitConfig) {
	return func(c *fitConfig) {
		if interval > 0 {
			c.LogInterval = interval
		}
	}
}
