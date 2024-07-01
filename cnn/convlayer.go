package cnn

import (
	"sync"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type convLayer struct {
	InputShape      [3]int
	OutputShape     [3]int
	TrainableParams int
	Parameters      parameters
	Activation      activation
	Optimizer       convOptimizer
	NFilters        int
	FilterSize      int
	Stride          int
	Iter            float64
}

type parameters struct {
	W [][]*mat.Dense // weights with shape (nFilters, nChannels, filterSize, filterSize)
	B *mat.Dense     // biases with shape (nFilters, 1)
}

type convConfig struct {
	InputShape  [3]int
	OutputShape [3]int
	NFilters    int
	FilterSize  int
	Stride      int
}

func newConvLayer(nFilters, filterSize, stride int, activation activation, optType optimizerType,
	inputShape, outputShape [3]int) *convLayer {
	nChannels := inputShape[0]

	// initialize convolutional neural network
	// filters with shape (nFilters, nChannels, filterSize, filterSize)
	filters := make([][]*mat.Dense, nFilters)
	for i := range filters {
		filters[i] = make([]*mat.Dense, nChannels)
		for j := range filters[i] {
			filters[i][j] = ngo.Randn(filterSize, filterSize)
		}
	}

	// choice of optimization algorithm
	optimizer := convOptimizerSettings[optType]
	if optType == AdamOptimizer {
		optimizer.Adam = convInitializeAdam(filters)
	}

	return &convLayer{
		InputShape:      inputShape,
		OutputShape:     outputShape,
		TrainableParams: nFilters * (filterSize*filterSize*nChannels + 1),
		Parameters: parameters{
			W: filters,
			B: ngo.Randn(nFilters, 1),
		},
		Activation: activation,
		Optimizer:  optimizer,
		NFilters:   nFilters,
		FilterSize: filterSize,
		Stride:     stride,
		Iter:       1,
	}
}

// forward propagation step: convolution operation
// input x with shape (nTraining, nChannels, hIn, wIn)
func (cl *convLayer) ForwardPropagation(x [][]*mat.Dense) ([][]*mat.Dense, [][]*mat.Dense) {
	stride := cl.Stride
	W := cl.Parameters.W
	b := cl.Parameters.B

	nFilters := len(W)
	nChannels := len(W[0])
	filterSize, _ := W[0][0].Dims()

	// output dimension
	hOut := cl.OutputShape[1]
	wOut := cl.OutputShape[2]

	// conv output with shape (nTraining, nFilters, hOut, wOut)
	nTraining := len(x)
	Z := make([][]*mat.Dense, nTraining) // linear function
	A := make([][]*mat.Dense, nTraining) // activation function
	for i := range Z {
		Z[i] = make([]*mat.Dense, nFilters)
		A[i] = make([]*mat.Dense, nFilters)
		for j := range Z[i] {
			Z[i][j] = mat.NewDense(hOut, wOut, nil)
			A[i][j] = mat.NewDense(hOut, wOut, nil)
		}
	}

	applyActivationFunction := func(_, _ int, v float64) float64 { return cl.Activation.Function(v) }
	var wg sync.WaitGroup
	for t := 0; t < nTraining; t++ {
		for f := 0; f < nFilters; f++ {
			wg.Add(1)
			go func(t, f int) {
				defer wg.Done()
				for i := 0; i < hOut; i++ {
					for j := 0; j < wOut; j++ {
						sum := 0.0
						for c := 0; c < nChannels; c++ {
							filter := W[f][c]
							for k := 0; k < filterSize; k++ {
								for l := 0; l < filterSize; l++ {
									sum += x[t][c].At(i*stride+k, j*stride+l) * filter.At(k, l)
								}
							}
						}
						// compute the linear operation
						Z[t][f].Set(i, j, sum+b.At(f, 0))
					}
				}
				// compute the non linear operation
				A[t][f] = ngo.Apply(applyActivationFunction, Z[t][f])
			}(t, f)
		}
	}
	wg.Wait()

	return Z, A
}

// backward propagation step: reverse convolution operation
// input x with shape (nTraining, nChannels, hIn, wIn)
// gradient dA with shape (nTraining, nFilters, hOut, wOut)
func (cl *convLayer) BackwardPropagation(x [][]*mat.Dense, Z, dA [][]*mat.Dense, learningRate float64) [][]*mat.Dense {
	stride := cl.Stride
	W := cl.Parameters.W

	nFilters := len(W)
	nChannels := len(W[0])
	filterSize, _ := W[0][0].Dims()

	// output dimension
	hOut := cl.OutputShape[1]
	wOut := cl.OutputShape[2]

	// initialize gradients
	dW := make([][]*mat.Dense, nFilters) // derivatives of the weigths W
	db := mat.NewDense(nFilters, 1, nil) // derivatives of the biases b
	for f := 0; f < nFilters; f++ {
		dW[f] = make([]*mat.Dense, nChannels)
		for c := 0; c < nChannels; c++ {
			dW[f][c] = mat.NewDense(filterSize, filterSize, nil)
		}
	}

	// initialize gradient for input x
	// input dxPrev with shape (nTraining, nChannels, hIn, wIn)
	nTraining := len(x)
	dxPrev := make([][]*mat.Dense, nTraining)
	for i := range dxPrev {
		dxPrev[i] = make([]*mat.Dense, nChannels)
		for j := range dxPrev[i] {
			dxPrev[i][j] = mat.NewDense(cl.InputShape[1], cl.InputShape[2], nil)
		}
	}

	applyActivationDerivative := func(_, _ int, v float64) float64 { return cl.Activation.Derivative(v) }
	for t := 0; t < nTraining; t++ {
		// calculate gradient
		for f := 0; f < nFilters; f++ {
			// apply gradient of activation function to dA
			dZ := ngo.Multiply(dA[t][f], ngo.Apply(applyActivationDerivative, Z[t][f]))
			for i := 0; i < hOut; i++ {
				for j := 0; j < wOut; j++ {
					dZValue := dZ.At(i, j)
					for c := 0; c < nChannels; c++ {
						for k := 0; k < filterSize; k++ {
							for l := 0; l < filterSize; l++ {
								// gradient of Z with respect to W
								dW[f][c].Set(k, l, dW[f][c].At(k, l)+dZValue*x[t][c].At(i*stride+k, j*stride+l))

								// gradient of Z with respect to x
								dxPrev[t][c].Set(i*stride+k, j*stride+l, dxPrev[t][c].At(i*stride+k, j*stride+l)+dZValue*W[f][c].At(k, l))
							}
						}
					}

					// gradient of Z with respect to b
					db.Set(f, 0, db.At(f, 0)+dZValue)
				}
			}
		}
	}

	// update parameters (optimization algorithm)
	cl.Parameters = cl.Optimizer.Function(&cl.Optimizer, cl.Parameters, dW, db, learningRate, cl.Iter)
	cl.Iter++

	return dxPrev
}
