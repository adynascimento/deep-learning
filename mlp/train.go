package mlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"time"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"github.com/olekukonko/tablewriter"
	"github.com/schollz/progressbar/v3"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// performs model training using the xTrain and yTrain matrices.
// both matrices have shape (nFeatures, nSamples), where each row
// corresponds to a feature and each column corresponds to a training sample.
func (nm *neuralModel) Fit(xTrain, yTrain *mat.Dense, options ...func(*fitConfig)) []float64 {
	nSamples := xTrain.RawMatrix().Cols

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

	// create a fixed index slice to map the columns
	indices := make([]int, nSamples)

	// loop
	iterPerEpoch := int(math.Ceil(float64(nSamples) / float64(nm.BatchSize)))
	for i := 1; i <= nm.Epochs; i++ {
		start := time.Now()

		var totalLoss float64
		var totalWeight float64

		if config.Shuffle {
			for idx := range indices {
				indices[idx] = idx
			}

			rng := newRand(nm.Seed)
			rng.Shuffle(nSamples, func(i, j int) {
				indices[i], indices[j] = indices[j], indices[i]
			})
		}

		bar := progressBar(iterPerEpoch, fmt.Sprintf("epoch %5d/%d: ", i, nm.Epochs), config.Verbose)
		for startIdx := 0; startIdx < nSamples; startIdx += nm.BatchSize {
			bar.Add(1)
			endIdx := min(startIdx+nm.BatchSize, nSamples)
			batchSize := float64(endIdx - startIdx)

			var xBatch, yBatch *mat.Dense
			if config.Shuffle {
				batchIndices := indices[startIdx:endIdx]
				xBatch = ngo.GatherColumns(xTrain, batchIndices)
				yBatch = ngo.GatherColumns(yTrain, batchIndices)
			} else {
				xBatch = xTrain.Slice(0, xTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)
				yBatch = yTrain.Slice(0, yTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)
			}

			// forward propagation
			yHat, Z, A, D := nm.Dense.ForwardPropagation(xBatch, true)

			// loss function
			loss := nm.LossFunction(yHat, yBatch, nm.Dense.Parameters, nm.Dense.L2Regularization)
			totalLoss += loss * batchSize
			totalWeight += batchSize

			// backward propagation
			_, dW, db := nm.Dense.BackwardPropagation(Z, A, D, yBatch)

			// update parameters (optimization algorithm)
			nm.Dense.UpdateParameters(dW, db, nm.LearningRate)
		}

		// print the loss every x iterations
		meanLoss := totalLoss / totalWeight
		if config.Verbose && (i%config.LogInterval == 0 || i == 1 || i == nm.Epochs) {
			if nm.Mode == nncore.ModeRegression {
				fmt.Printf(" | t: %7.2fms | loss: %.6e \n", float64(time.Since(start))/float64(time.Millisecond), meanLoss)
			} else {
				fmt.Printf(" | t: %7.2fms | loss: %.6e | acc: %.4f \n",
					float64(time.Since(start))/float64(time.Millisecond), meanLoss, nm.Evaluate(xTrain, yTrain))
			}
		}
		losses = append(losses, meanLoss)
	}

	return losses
}

// predictions with forward propagation
func (nm *neuralModel) Predict(x *mat.Dense) *mat.Dense {
	predictions, _, _, _ := nm.Dense.ForwardPropagation(x, false)
	return predictions
}

// evaluate model
func (nm *neuralModel) Evaluate(x, y *mat.Dense) float64 {
	yPred := nm.Predict(x)

	metric := 0.0
	switch nm.Mode {
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

// initializing permutation
func newRand(seed *uint64) *rand.Rand {
	if seed != nil {
		return rand.New(rand.NewPCG(*seed, *seed))
	}

	return rand.New(rand.NewPCG(
		uint64(time.Now().UnixNano()),
		uint64(time.Now().UnixNano()),
	))
}
