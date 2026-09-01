package cnn

import (
	"runtime"

	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/mat"
)

type CNN interface {
	AddConv2DLayer(nFilters, filterSize, stride int)
	AddMaxPooling2DLayer(size, stride int)
	AddDenseLayer(nnStructure []int)
	NewTrainer(config TrainerConfig, options ...func(*cnnModel)) CNNModel
}

type CNNModel interface {
	// performs model training using the xTrain and yTrain datasets.
	// xTrain is a 4D tensor with shape (nTraining, nChannels, hIn, wIn).
	// yTrain is a matrix with shape (nSamples, nFeatures), where each row
	// corresponds to a training sample and each column corresponds to a feature.
	Fit(xTrain [][]*mat.Dense, yTrain *mat.Dense, options ...func(*fitConfig)) []float64
	Predict(x [][]*mat.Dense) *mat.Dense
	Evaluate(x [][]*mat.Dense, y *mat.Dense) float64
	Save(path string)
	Summary()
}

// input shape (nChannels, height, width)
type CNNConfig struct {
	InputShape [3]int
	Activation nncore.ActivationType
	Mode       nncore.ModeType
}

type TrainerConfig struct {
	Optimizer    nncore.OptimizerType
	LearningRate float64
	Epochs       int
}

// input shape (nChannels, height, width)
type cnn struct {
	InputShape          [3]int
	Activation          nncore.Activation
	Mode                nncore.ModeType
	OutputActivation    nncore.OutputActivation
	LossFunction        nncore.LossFunction
	ConvLayers          []*convLayer
	ConvConfigs         []convConfig
	PoolLayers          []*poolLayer
	FlattenLayer        *flatten
	DenseLayer          *nncore.Dense
	DenseLayerStructure []int
}

type cnnModel struct {
	*cnn
	*cnnForwardOutputs
	NWorkers         int
	WorkerGradients  [][]gradients
	LearningRate     float64
	Epochs           int
	BatchSize        int
	L2Regularization float64
	Dropout          float64
	Seed             *uint64
}

type cnnForwardOutputs struct {
	ConvOutputs map[string][][]*mat.Dense
	PoolOutputs map[string][]*poolCache
}

type fitConfig struct {
	Shuffle     bool
	Verbose     bool
	LogInterval int
}

func NewConvNeuralNetwork(config CNNConfig) CNN {
	// choice of activation function
	activation := nncore.NewActivation(config.Activation)

	// choice of output layer activation function and loss function
	configMode := nncore.NewMode(config.Mode)

	return &cnn{
		InputShape:       config.InputShape,
		Activation:       activation,
		Mode:             config.Mode,
		OutputActivation: configMode.OutputActivation,
		LossFunction:     configMode.LossFunction,
	}
}

// add convolutional layer
func (c *cnn) AddConv2DLayer(nFilters, filterSize, stride int) {
	inputShape := c.LastOutputShape()
	hOut := (inputShape[1]-filterSize)/stride + 1
	wOut := (inputShape[2]-filterSize)/stride + 1

	c.ConvConfigs = append(c.ConvConfigs, convConfig{
		InputShape:  inputShape,
		OutputShape: [3]int{nFilters, hOut, wOut},
		NFilters:    nFilters,
		FilterSize:  filterSize,
		Stride:      stride,
	})
}

// add pooling layer
func (c *cnn) AddMaxPooling2DLayer(size, stride int) {
	inputShape := c.LastOutputShape()
	c.PoolLayers = append(c.PoolLayers, newPoolLayer(size, stride, inputShape))
}

// add fully connected layer
func (c *cnn) AddDenseLayer(nnStructure []int) {
	// input dimension features (previous layer output)
	inputShape := c.LastOutputShape()
	inputDim := inputShape[0] * inputShape[1] * inputShape[2]

	nnStructure = append([]int{inputDim}, nnStructure...)
	c.DenseLayerStructure = nnStructure
}

// returns the output shape of the last layer in the network
func (c *cnn) LastOutputShape() [3]int {
	// no convolutional layers have been added yet,
	// so the network output is still the input shape
	if len(c.ConvConfigs) == 0 {
		return c.InputShape
	}
	// the last added layer is a pooling layer
	// (one pooling layer for each convolutional layer)
	if len(c.PoolLayers) == len(c.ConvConfigs) {
		return c.PoolLayers[len(c.PoolLayers)-1].OutputShape
	}
	// the last added layer is a convolutional layer
	// (a pooling layer has not been added after it yet)
	return c.ConvConfigs[len(c.ConvConfigs)-1].OutputShape
}

func (c *cnn) NewTrainer(config TrainerConfig, options ...func(*cnnModel)) CNNModel {
	model := cnnModel{
		cnn: c,
		cnnForwardOutputs: &cnnForwardOutputs{
			ConvOutputs: make(map[string][][]*mat.Dense),
			PoolOutputs: make(map[string][]*poolCache),
		},
		LearningRate: config.LearningRate,
		Epochs:       config.Epochs,
		BatchSize:    32,
	}

	// apply additional options
	for _, option := range options {
		option(&model)
	}

	// initialize the random number generator
	rng := nncore.NewRand(model.Seed)

	// add convolutional layer
	for _, convConfig := range c.ConvConfigs {
		convLayer := newConvLayer(convLayerConfig{
			convConfig:       convConfig,
			Activation:       c.Activation,
			Optimizer:        config.Optimizer,
			L2Regularization: model.L2Regularization,
			RNG:              rng.Spawn(),
		})
		c.ConvLayers = append(c.ConvLayers, convLayer)
	}

	// add flatten layer
	c.FlattenLayer = newFlatten()

	// add fully connected layer
	c.DenseLayer = nncore.NewDense(nncore.DenseConfig{
		NNStructure:      c.DenseLayerStructure,
		Activation:       c.Activation,
		OutputActivation: c.OutputActivation,
		Optimizer:        config.Optimizer,
		L2Regularization: model.L2Regularization,
		Dropout:          model.Dropout,
		RNG:              rng.Spawn(),
	})

	// each worker accumulates its own local gradients in the backward propagation
	// for a subset of the training samples before the final gradient reduction
	model.NWorkers = runtime.GOMAXPROCS(0)
	model.WorkerGradients = newWorkerGradients(c.ConvLayers, model.NWorkers)

	return &model
}

// initialize worker gradients
func newWorkerGradients(convLayers []*convLayer, nWorkers int) [][]gradients {
	workerGradients := make([][]gradients, nWorkers)
	for w := 0; w < nWorkers; w++ {
		workerGradients[w] = make([]gradients, len(convLayers))
		for i := range convLayers {
			layer := convLayers[i]
			workerGradients[w][i] = newGradients(layer.NFilters, layer.NChannels, layer.FilterSize)
		}
	}

	return workerGradients
}

func WithBatchSize(batchSize int) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.BatchSize = batchSize
	}
}

func WithL2Regularization(lambd float64) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.L2Regularization = lambd
	}
}

func WithDropout(dropout float64) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.Dropout = dropout
	}
}

func WithSeed(seed uint64) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.Seed = &seed
	}
}

func WithShuffle(shuffle bool) func(*fitConfig) {
	return func(c *fitConfig) {
		c.Shuffle = shuffle
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
