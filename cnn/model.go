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
	// yTrain is a matrix with shape (nFeatures, nSamples), where each row
	// corresponds to a feature and each column corresponds to a training sample.
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
	Optimizer        nncore.OptimizerType
	LearningRate     float64
	L2Regularization float64
	Epochs           int
	BatchSize        int
}

type cnnForwardOutputs struct {
	ConvOutputs map[string][][]*mat.Dense
	PoolOutputs map[string][]*poolCache
}

type fitConfig struct {
	Verbose     bool
	LogInterval int
}

func NewConvNeuralNetwork(config CNNConfig) CNN {
	// choice of activation function
	activationFunction := nncore.ActivationSettings[config.Activation]

	// choice of output layer activation function and loss function
	configMode := nncore.ModeSettings[config.Mode]

	return &cnn{
		InputShape:       config.InputShape,
		Activation:       activationFunction,
		Mode:             config.Mode,
		OutputActivation: configMode.OutputActivation,
		LossFunction:     configMode.LossFunction,
	}
}

// add convolutional layer
func (c *cnn) AddConv2DLayer(nFilters, filterSize, stride int) {
	inputShape := c.InputShape
	if len(c.ConvConfigs) > 0 {
		inputShape = c.PoolLayers[len(c.PoolLayers)-1].OutputShape
	}
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
	inputShape := c.ConvConfigs[len(c.ConvConfigs)-1].OutputShape
	c.PoolLayers = append(c.PoolLayers, newPoolLayer(size, stride, inputShape))
}

// add fully connected layer
func (c *cnn) AddDenseLayer(nnStructure []int) {
	// input dimension features (previous layer output)
	inputShape := c.ConvConfigs[len(c.ConvConfigs)-1].OutputShape
	if len(c.ConvConfigs) == len(c.PoolLayers) {
		inputShape = c.PoolLayers[len(c.PoolLayers)-1].OutputShape
	}
	inputDim := inputShape[0] * inputShape[1] * inputShape[2]

	nnStructure = append([]int{inputDim}, nnStructure...)
	c.DenseLayerStructure = nnStructure
}

func (c *cnn) NewTrainer(config TrainerConfig, options ...func(*cnnModel)) CNNModel {
	// add convolutional layer
	for _, v := range c.ConvConfigs {
		convLayer := newConvLayer(v.NFilters, v.FilterSize, v.Stride, c.Activation, config.Optimizer,
			v.InputShape, v.OutputShape)
		c.ConvLayers = append(c.ConvLayers, convLayer)
	}

	// add flatten layer
	c.FlattenLayer = newFlatten()

	// add fully connected layer
	c.DenseLayer = nncore.NewDense(nncore.DenseConfig{
		NNStructure:      c.DenseLayerStructure,
		Activation:       c.Activation,
		OutputActivation: c.OutputActivation,
	})
	c.DenseLayer.Optimizer = nncore.NewOptimizer(config.Optimizer, c.DenseLayer.Parameters)

	// each worker accumulates its own local gradients in the backward propagation
	// for a subset of the training samples before the final gradient reduction
	nWorkers := runtime.GOMAXPROCS(0)
	workerGradients := make([][]gradients, nWorkers)
	for w := 0; w < nWorkers; w++ {
		workerGradients[w] = make([]gradients, len(c.ConvLayers))
		for i := range c.ConvLayers {
			layer := c.ConvLayers[i]

			// initialize worker gradients
			workerGradients[w][i] = newGradients(layer.NFilters, layer.NChannels, layer.FilterSize)
		}
	}

	model := cnnModel{
		cnn: c,
		cnnForwardOutputs: &cnnForwardOutputs{
			ConvOutputs: make(map[string][][]*mat.Dense),
			PoolOutputs: make(map[string][]*poolCache),
		},
		NWorkers:        nWorkers,
		WorkerGradients: workerGradients,
		Optimizer:       config.Optimizer,
		LearningRate:    config.LearningRate,
		Epochs:          config.Epochs,
		BatchSize:       32,
	}

	// apply additional options
	for _, option := range options {
		option(&model)
	}

	return &model
}

func WithBatchSize(batchSize int) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.BatchSize = batchSize
	}
}

func WithL2Regularization(lambd float64) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.L2Regularization = lambd
		nm.DenseLayer.L2Regularization = lambd
	}
}

func WithDropout(dropout float64) func(*cnnModel) {
	return func(nm *cnnModel) {
		nm.DenseLayer.Dropout = dropout
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
