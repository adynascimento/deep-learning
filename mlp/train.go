package mlp

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"github.com/olekukonko/tablewriter"
	"github.com/schollz/progressbar/v3"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// forward propagation step
func (nm *neuralModel) ForwardPropagation(x *mat.Dense, training bool) (*mat.Dense, map[string]*mat.Dense, map[string]*mat.Dense, map[string][]bool) {
	L := len(nm.Parameters) / 2      // number of layers
	Z := make(map[string]*mat.Dense) // linear function
	A := make(map[string]*mat.Dense) // activation function
	D := make(map[string][]bool)     // dropout masks (training only)
	A[strconv.Itoa(0)] = x

	for l := 0; l < L-1; l++ {
		W := nm.Parameters["W"+strconv.Itoa(l+1)] // weights W
		b := nm.Parameters["b"+strconv.Itoa(l+1)] // biases b

		Z[strconv.Itoa(l+1)] = ngo.AddMatrixVector(ngo.MatMul(W, A[strconv.Itoa(l)]), b) // compute the linear operation
		A[strconv.Itoa(l+1)] = nm.Activation.Function(Z[strconv.Itoa(l+1)])              // compute the non linear operation

		// apply dropout
		if training && nm.Dropout > 0 {
			D[strconv.Itoa(l+1)] = nncore.DropoutMask(A[strconv.Itoa(l+1)], nm.Dropout)
			nncore.ApplyDropoutMask(A[strconv.Itoa(l+1)], D[strconv.Itoa(l+1)], nm.Dropout)
		}
	}
	// for output layer
	Z[strconv.Itoa(L)] = ngo.AddMatrixVector(ngo.MatMul(nm.Parameters["W"+strconv.Itoa(L)],
		A[strconv.Itoa(L-1)]), nm.Parameters["b"+strconv.Itoa(L)])
	A[strconv.Itoa(L)] = nm.OutputActivation.Function(Z[strconv.Itoa(L)])

	// prediction
	yHat := A[strconv.Itoa(L)]

	return yHat, Z, A, D
}

// backward propagation step
func (nm *neuralModel) BackwardPropagation(Z, A map[string]*mat.Dense, D map[string][]bool, y *mat.Dense) (map[string]*mat.Dense, map[string]*mat.Dense) {
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

	for l := L - 1; l > 0; l-- {
		dA[strconv.Itoa(l)] = ngo.MatMul(nm.Parameters["W"+strconv.Itoa(l+1)].T(), dZ[strconv.Itoa(l+1)])

		// apply dropout
		if nm.Dropout > 0 {
			nncore.ApplyDropoutMask(dA[strconv.Itoa(l)], D[strconv.Itoa(l)], nm.Dropout)
		}

		dZ[strconv.Itoa(l)] = ngo.Multiply(dA[strconv.Itoa(l)], nm.Activation.Derivative(Z[strconv.Itoa(l)]))
		dW[strconv.Itoa(l)] = ngo.Add(ngo.MatMul(dZ[strconv.Itoa(l)], A[strconv.Itoa(l-1)].T()),
			ngo.Scale(nm.L2Regularization/float64(m), nm.Parameters["W"+strconv.Itoa(l)]))
		db[strconv.Itoa(l)] = ngo.Sum(dZ[strconv.Itoa(l)], ngo.OverColumns)
	}

	return dW, db
}

// performs model training using the xTrain and yTrain matrices.
// both matrices have shape (nFeatures, nSamples), where each row
// corresponds to a feature and each column corresponds to a training sample.
func (nm *neuralModel) Fit(xTrain, yTrain *mat.Dense, options ...func(*fitConfig)) []float64 {
	nSamples := xTrain.RawMatrix().Cols

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
	iterPerEpoch := int(math.Ceil(float64(nSamples) / float64(nm.BatchSize)))
	for i := 1; i <= nm.Epochs; i++ {
		start := time.Now()
		lossBatches := []float64{}

		bar := progressBar(iterPerEpoch, fmt.Sprintf("epoch %5d/%d: ", i, nm.Epochs), config.Verbose)
		for startIdx := 0; startIdx < nSamples; startIdx += nm.BatchSize {
			bar.Add(1)
			endIdx := startIdx + nm.BatchSize
			if endIdx > nSamples {
				endIdx = nSamples
			}

			xBatch := xTrain.Slice(0, xTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)
			yBatch := yTrain.Slice(0, yTrain.RawMatrix().Rows, startIdx, endIdx).(*mat.Dense)

			// forward propagation
			yHat, Z, A, D := nm.ForwardPropagation(xBatch, true)

			// loss function
			loss := nm.LossFunction(yHat, yBatch, nm.Parameters, nm.L2Regularization)
			lossBatches = append(lossBatches, loss)

			// backward propagation
			dW, db := nm.BackwardPropagation(Z, A, D, yBatch)

			// update parameters (optimization algorithm)
			nm.Parameters = nm.Optimizer.Function(&nm.Optimizer, nm.Parameters, dW, db,
				nm.LearningRate, float64(i))
		}

		// print the loss every x iterations
		meanLoss := stat.Mean(lossBatches, nil)
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
	predictions, _, _, _ := nm.ForwardPropagation(x, false)
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
