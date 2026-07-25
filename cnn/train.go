package cnn

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"github.com/olekukonko/tablewriter"
	"github.com/schollz/progressbar/v3"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

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
	nTraining := len(cm.ConvOutputs["convI1"])

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
						cm.ConvOutputs["convZ"+strconv.Itoa(i+1)][t], dOut, &workerGradient[i], cm.L2Regularization/float64(nTraining))
				}
			}
		}(grad)
	}

	// send workers
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

// performs model training using the xTrain and yTrain datasets.
// xTrain is a 4D tensor with shape (nTraining, nChannels, hIn, wIn).
// yTrain is a matrix with shape (nFeatures, nSamples), where each row
// corresponds to a feature and each column corresponds to a training sample.
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
	case nncore.ModeRegression:
		// mean squared error
		metric = mat.Sum(ngo.Square(ngo.Sub(y, yPred))) / float64(y.RawMatrix().Cols)

	case nncore.ModeMultiClass:
		// accuracy
		for j := 0; j < y.RawMatrix().Cols; j++ {
			trueClass := floats.MaxIdx(mat.Col(nil, j, y))
			predClass := floats.MaxIdx(mat.Col(nil, j, yPred))
			if trueClass == predClass {
				metric++
			}
		}
		metric = (metric / float64(y.RawMatrix().Cols))

	case nncore.ModeMultiLabel:
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
