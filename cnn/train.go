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
)

// cnn forward propagation step
func (cm *cnnModel) ForwardPropagation(x [][]*mat.Dense, training bool) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, map[string][]bool) {
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
	yPred, Z, A, D := cm.DenseLayer.ForwardPropagation(flattened, training)

	return yPred, Z, A, D
}

// cnn backward propagation step
func (cm *cnnModel) BackwardPropagation(Z, A map[string]*mat.Dense, D map[string][]bool, yTrue *mat.Dense) {
	nTraining := len(cm.ConvOutputs["convI1"])

	// fully connected layer step
	// update dense parameters (optimization algorithm)
	dOutDense, dW, db := cm.DenseLayer.BackwardPropagation(Z, A, D, yTrue)
	cm.DenseLayer.UpdateParameters(dW, db, cm.LearningRate)

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
						cm.ConvOutputs["convZ"+strconv.Itoa(i+1)][t], dOut, &workerGradient[i], cm.ConvLayers[i].L2Regularization/float64(nTraining))
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
// yTrain is a matrix with shape (nSamples, nFeatures), where each row
// corresponds to a training sample and each column corresponds to a feature.
func (cm *cnnModel) Fit(xTrain [][]*mat.Dense, yTrain *mat.Dense, options ...func(*fitConfig)) []float64 {
	nSamples := len(xTrain)

	// default values
	config := fitConfig{
		Shuffle:     true,
		Verbose:     true,
		LogInterval: 1,
	}

	// apply additional options
	for _, opt := range options {
		opt(&config)
	}

	// keep track of the loss
	losses := []float64{}

	// create a fixed index slice to map the samples
	indices := make([]int, nSamples)

	// loop
	iterPerEpoch := int(math.Ceil(float64(nSamples) / float64(cm.BatchSize)))
	for i := 1; i <= cm.Epochs; i++ {
		start := time.Now()

		var totalLoss float64
		var totalWeight float64

		// generate a shuffled sample order for the current epoch
		if config.Shuffle {
			for idx := range indices {
				indices[idx] = idx
			}
			rng := nncore.NewRand(cm.Seed)
			rng.Shuffle(nSamples, func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		bar := progressBar(iterPerEpoch, fmt.Sprintf("epoch %5d/%d: ", i, cm.Epochs), config.Verbose)
		for startIdx := 0; startIdx < nSamples; startIdx += cm.BatchSize {
			bar.Add(1)
			endIdx := min(startIdx+cm.BatchSize, nSamples)
			batchSize := float64(endIdx - startIdx)

			// gather the current batch according to the sample order
			var xBatch [][]*mat.Dense
			var yBatch *mat.Dense
			if config.Shuffle {
				batchIndices := indices[startIdx:endIdx]
				xBatch = ngo.GatherSamples(xTrain, batchIndices).([][]*mat.Dense)
				yBatch = ngo.GatherSamples(yTrain, batchIndices).(*mat.Dense)
			} else {
				xBatch = xTrain[startIdx:endIdx]
				yBatch = yTrain.Slice(startIdx, endIdx, 0, yTrain.RawMatrix().Cols).(*mat.Dense)
			}

			// forward propagation
			yPred, Z, A, D := cm.ForwardPropagation(xBatch, true)

			// loss function
			loss := cm.LossFunction(yPred, yBatch, cm.DenseLayer.Parameters, cm.DenseLayer.L2Regularization)
			totalLoss += loss * batchSize
			totalWeight += batchSize

			// backward propagation with update parameters (optimization algorithm)
			cm.BackwardPropagation(Z, A, D, yBatch)
		}

		// print the loss every x iterations
		meanLoss := totalLoss / totalWeight
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
	predictions, _, _, _ := cm.ForwardPropagation(x, false)
	return predictions
}

// evaluate model
func (cm *cnnModel) Evaluate(x [][]*mat.Dense, y *mat.Dense) float64 {
	yPred := cm.Predict(x)

	metric := 0.0
	switch cm.Mode {
	case nncore.ModeRegression:
		// mean squared error
		metric = mat.Sum(ngo.Square(ngo.Sub(y, yPred))) / float64(y.RawMatrix().Rows)

	case nncore.ModeMultiClass:
		// accuracy
		for i := 0; i < y.RawMatrix().Rows; i++ {
			trueClass := floats.MaxIdx(y.RawRowView(i))
			predClass := floats.MaxIdx(yPred.RawRowView(i))
			if trueClass == predClass {
				metric++
			}
		}
		metric = (metric / float64(y.RawMatrix().Rows))

	case nncore.ModeMultiLabel:
		// hamming accuracy
		for i := 0; i < y.RawMatrix().Rows; i++ {
			predRow := yPred.RawRowView(i)

			correctLabels := 0.0
			for j, pred := range predRow {
				// round considers the threshold 0.5
				if y.At(i, j) == math.Round(pred) {
					correctLabels++
				}
			}
			metric += correctLabels / float64(len(predRow))
		}
		metric = (metric / float64(y.RawMatrix().Rows))
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
		"Flatten Layer", fmt.Sprintf("(None, %d)", cm.DenseLayerStructure[0]), "0",
	})
	for i, v := range cm.DenseLayerStructure[1:] {
		data = append(data, []string{
			fmt.Sprintf("Dense Layer %d", i+1), fmt.Sprintf("(None, %d)", v), fmt.Sprintf("%d",
				cm.DenseLayerStructure[i]*cm.DenseLayerStructure[i+1]+cm.DenseLayerStructure[i+1]),
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
