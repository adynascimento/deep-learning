package cnn

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/schollz/progressbar/v3"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type CNN interface {
	AddConv2DLayer(nFilters, filterSize, stride int)
	AddMaxPooling2DLayer(size, stride int)
	AddDenseLayer(nnStructure []int)
	NewTrainer(config TrainerConfig, options ...func(*cnnModel)) CNNModel
}

type CNNModel interface {
	Fit(xTrain [][]*mat.Dense, yTrain *mat.Dense, options ...func(*fitConfig)) []float64
	Predict(x [][]*mat.Dense) *mat.Dense
	Evaluate(x [][]*mat.Dense, y *mat.Dense) float64
	Save(path string)
	Summary()
}

// input shape (nChannels, height, width)
type CNNConfig struct {
	InputShape [3]int
	Activation activationType
	Mode       modeType
}

type TrainerConfig struct {
	Optimizer    optimizerType
	LearningRate float64
	Epochs       int
}

// input shape (nChannels, height, width)
type cnn struct {
	InputShape          [3]int
	Activation          activation
	Mode                modeType
	OutputActivation    outputActivation
	LossFunction        lossFunction
	ConvLayers          []*convLayer
	ConvConfigs         []convConfig
	PoolLayers          []*poolLayer
	FlattenLayer        *flatten
	DenseLayer          *denseLayer
	DenseLayerStructure []int
}

type cnnModel struct {
	*cnn
	*cnnForwardOutputs
	NWorkers         int
	WorkerGradients  [][]gradients
	Optimizer        optimizerType
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
	activationFunction := activationSettings[config.Activation]

	// choice of output layer activation function and loss function
	lossFunction := modeSettings[config.Mode].lossFunction
	outputActivationFunction := modeSettings[config.Mode].outputActivation

	return &cnn{
		InputShape:       config.InputShape,
		Activation:       activationFunction,
		Mode:             config.Mode,
		OutputActivation: outputActivationFunction,
		LossFunction:     lossFunction,
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
	c.DenseLayer = newDenseLayer(c.DenseLayerStructure, c.Activation, c.OutputActivation, config.Optimizer)

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

// cnn forward propagation step
func (cm *cnnModel) ForwardPropagation(x [][]*mat.Dense) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense) {
	nTraining := len(x)

	for i := range cm.ConvLayers {
		cm.ConvOutputs["convI"+strconv.Itoa(i+1)] = make([][]*mat.Dense, nTraining)
		cm.ConvOutputs["convZ"+strconv.Itoa(i+1)] = make([][]*mat.Dense, nTraining)
	}

	for i := range cm.PoolLayers {
		cm.PoolOutputs["pool"+strconv.Itoa(i+1)] = make([]*poolCache, nTraining)
	}

	var wg sync.WaitGroup
	workers := make(chan int, cm.NWorkers)

	// convolutional and pooling steps
	poolOut := make([][]*mat.Dense, nTraining)
	for w := 0; w < cm.NWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range workers {
				out := x[t]
				for i := range cm.ConvLayers {
					cm.ConvOutputs["convI"+strconv.Itoa(i+1)][t] = out
					cm.ConvOutputs["convZ"+strconv.Itoa(i+1)][t], out = cm.ConvLayers[i].ForwardPropagation(out)
					if i < len(cm.PoolLayers) {
						out, cm.PoolOutputs["pool"+strconv.Itoa(i+1)][t] = cm.PoolLayers[i].ForwardPropagation(out)
					}
				}
				poolOut[t] = out
			}
		}()
	}

	// send workers
	for t := 0; t < nTraining; t++ {
		workers <- t
	}
	close(workers)

	// wait for all workers to finish
	wg.Wait()

	// flatten step
	// input dimension features (pool layer output)
	flattened := cm.FlattenLayer.ForwardPropagation(poolOut)

	// fully connected layer step
	// input dimension features (flatten layer output)
	yPred, Z, A := cm.DenseLayer.ForwardPropagation(flattened)

	return yPred, Z, A
}

// cnn backward propagation step
func (cm *cnnModel) BackwardPropagation(Z, A map[string]*mat.Dense, yTrue *mat.Dense) {
	// fully connected layer step
	dOutDense := cm.DenseLayer.BackwardPropagation(Z, A, yTrue, cm.LearningRate, cm.L2Regularization)

	// flatten step
	dOutFlatten := cm.FlattenLayer.BackwardPropagation(dOutDense)

	var wg sync.WaitGroup
	workers := make(chan int, cm.NWorkers)

	// pooling and convolutional steps
	for w := 0; w < cm.NWorkers; w++ {
		grad := cm.WorkerGradients[w]
		wg.Add(1)
		go func(workerGradient []gradients) {
			defer wg.Done()
			for t := range workers {
				dOut := dOutFlatten[t]
				for i := len(cm.ConvLayers) - 1; i >= 0; i-- {
					if i < len(cm.PoolLayers) {
						dOut = cm.PoolLayers[i].BackwardPropagation(dOut, cm.PoolOutputs["pool"+strconv.Itoa(i+1)][t])
					}
					dOut = cm.ConvLayers[i].BackwardPropagation(cm.ConvOutputs["convI"+strconv.Itoa(i+1)][t],
						cm.ConvOutputs["convZ"+strconv.Itoa(i+1)][t], dOut, &workerGradient[i])
				}
			}
		}(grad)
	}

	// send workers
	nTraining := len(cm.ConvOutputs["convI1"])
	for t := 0; t < nTraining; t++ {
		workers <- t
	}
	close(workers)

	// wait for all workers to finish
	wg.Wait()

	// update convlayers parameters (optimization algorithm)
	for i := len(cm.ConvLayers) - 1; i >= 0; i-- {
		workerGradients := make([]gradients, cm.NWorkers)
		for w := 0; w < cm.NWorkers; w++ {
			workerGradients[w] = cm.WorkerGradients[w][i]
		}

		cm.ConvLayers[i].ReduceWorkerGradients(workerGradients)
		cm.ConvLayers[i].UpdateParameters(cm.LearningRate)
	}
}

// performs model training with the xTrain and yTrain matrices,
// xTrain is represented with shape (nTraining, nChannels, hIn, wIn)
// yTrain is represented as an rows X cols matrix a where each
// row is a variable and each column is an observation.
// yTrain matrix shape (nFeatures, nSamples)
func (cm *cnnModel) Fit(xTrain [][]*mat.Dense, yTrain *mat.Dense, options ...func(*fitConfig)) []float64 {
	nSamples := len(xTrain)

	// default values
	config := fitConfig{
		Verbose:     true,
		LogInterval: 1,
	}

	// apply additional options
	for _, opt := range options {
		opt(&config)
	}

	// keep track of the loss
	losses := []float64{}

	// loop
	iterPerEpoch := int(math.Ceil(float64(nSamples) / float64(cm.BatchSize)))
	for i := 1; i <= cm.Epochs; i++ {
		start := time.Now()
		weights := []float64{}
		lossBatches := []float64{}

		bar := progressBar(iterPerEpoch, fmt.Sprintf("epoch %5d/%d: ", i, cm.Epochs), config.Verbose)
		for startIdx := 0; startIdx < nSamples; startIdx += cm.BatchSize {
			bar.Add(1)
			endIdx := startIdx + cm.BatchSize
			if endIdx > nSamples {
				endIdx = nSamples
			}

			xBatch := xTrain[startIdx:endIdx]
			yBatch := yTrain.Slice(0, yTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)

			// forward propagation
			yPred, Z, A := cm.ForwardPropagation(xBatch)

			// loss function
			loss := cm.LossFunction(yPred, yBatch, cm.DenseLayer.Parameters, cm.L2Regularization)
			lossBatches = append(lossBatches, loss)
			weights = append(weights, float64(len(xBatch)))

			// backward propagation with update parameters (optimization algorithm)
			cm.BackwardPropagation(Z, A, yBatch)
		}

		// print the loss every x iterations
		meanLoss := stat.Mean(lossBatches, weights)
		if config.Verbose && (i%config.LogInterval == 0 || i == 1 || i == cm.Epochs) {
			fmt.Printf(" | t: %7.2fms | loss: %.6e | acc: %.4f \n",
				float64(time.Since(start))/float64(time.Millisecond), meanLoss, cm.Evaluate(xTrain, yTrain))
		}
		losses = append(losses, meanLoss)
	}

	return losses
}

// predictions with forward propagation
func (cm *cnnModel) Predict(x [][]*mat.Dense) *mat.Dense {
	predictions, _, _ := cm.ForwardPropagation(x)
	return predictions
}

// evaluate model
func (cm *cnnModel) Evaluate(x [][]*mat.Dense, y *mat.Dense) float64 {
	yPred := cm.Predict(x)

	metric := 0.0
	switch cm.Mode {
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
func (cm *cnnModel) Summary() {
	data := [][]string{}
	for i := 0; i < len(cm.ConvLayers); i++ {
		convOutputShape := cm.ConvLayers[i].OutputShape
		data = append(data, []string{
			fmt.Sprintf("Conv2D Layer %d", i+1), fmt.Sprintf("(None, %d, %d, %d)", convOutputShape[0],
				convOutputShape[1], convOutputShape[2]), fmt.Sprintf("%d", cm.ConvLayers[i].TrainableParams),
		})
		if i < len(cm.PoolLayers) {
			poolOutputShape := cm.PoolLayers[i].OutputShape
			data = append(data, []string{
				fmt.Sprintf("MaxPooling2D Layer %d", i+1), fmt.Sprintf("(None, %d, %d, %d)", poolOutputShape[0],
					poolOutputShape[1], poolOutputShape[2]), "0",
			})
		}
	}
	data = append(data, []string{
		"Flatten Layer", fmt.Sprintf("(None, %d)", cm.DenseLayer.NNStructure[0]), "0",
	})
	for i, v := range cm.DenseLayer.NNStructure[1:] {
		data = append(data, []string{
			fmt.Sprintf("Dense Layer %d", i+1), fmt.Sprintf("(None, %d)", v), fmt.Sprintf("%d",
				cm.DenseLayer.NNStructure[i]*cm.DenseLayer.NNStructure[i+1]+cm.DenseLayer.NNStructure[i+1]),
		})
	}

	// table configuration
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Layer (type)", "Output Shape", "Param #"})
	table.SetCenterSeparator("|")
	table.AppendBulk(data)
	table.Render()
}

// progress bar
func progressBar(iter int, description string, verbose bool) *progressbar.ProgressBar {
	return progressbar.NewOptions(iter,
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionShowElapsedTimeOnFinish(),
		progressbar.OptionSetDescription(description),
		progressbar.OptionSetVisibility(verbose),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]━[reset]",
			SaucerPadding: " ",
		}),
	)
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
