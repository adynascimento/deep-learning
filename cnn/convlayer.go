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
	Gradients       gradients
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

type gradients struct {
	DW [][]*mat.Dense // weights with shape (nFilters, nChannels, filterSize, filterSize)
	DB *mat.Dense     // biases with shape (nFilters, 1)
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

	// initialize gradients
	dW := make([][]*mat.Dense, nFilters) // derivatives of the weigths W
	for f := 0; f < nFilters; f++ {
		dW[f] = make([]*mat.Dense, nChannels)
		for c := 0; c < nChannels; c++ {
			dW[f][c] = mat.NewDense(filterSize, filterSize, nil)
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
			B: mat.NewDense(nFilters, 1, nil),
		},
		Activation: activation,
		Gradients: gradients{
			DW: dW,
			DB: mat.NewDense(nFilters, 1, nil),
		},
		Optimizer:  optimizer,
		NFilters:   nFilters,
		FilterSize: filterSize,
		Stride:     stride,
		Iter:       1,
	}
}

// forward propagation step: convolution operation
// input x with shape (nChannels, hIn, wIn)
func (cl *convLayer) ForwardPropagation(x []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	stride := cl.Stride
	W := cl.Parameters.W
	b := cl.Parameters.B

	nFilters := len(W)
	nChannels := len(W[0])

	// output dimension
	hOut := cl.OutputShape[1]
	wOut := cl.OutputShape[2]

	// conv output with shape (nFilters, hOut, wOut)
	Z := make([]*mat.Dense, nFilters) // linear function
	A := make([]*mat.Dense, nFilters) // activation function
	for i := range Z {
		Z[i] = mat.NewDense(hOut, wOut, nil)
		A[i] = mat.NewDense(hOut, wOut, nil)
	}

	var wg sync.WaitGroup
	workers := make(chan struct{}, 8)

	applyActivationFunction := func(_, _ int, v float64) float64 { return cl.Activation.Function(v) }
	for f := 0; f < nFilters; f++ {
		wg.Add(1)
		workers <- struct{}{}
		go func(f int) {
			defer wg.Done()
			defer func() { <-workers }()

			convolve := mat.NewDense(hOut, wOut, nil)
			for c := 0; c < nChannels; c++ {
				convolve = ngo.Add(convolve, Convolve2D(x[c], W[f][c], stride))
			}

			applyBias := func(_, _ int, v float64) float64 { return v + b.At(f, 0) }
			convolve = ngo.Apply(applyBias, convolve)

			Z[f] = convolve
			A[f] = ngo.Apply(applyActivationFunction, convolve)

		}(f)
	}
	wg.Wait()

	return Z, A
}

// backward propagation step: reverse convolution operation
// input x with shape (nChannels, hIn, wIn)
// gradient dA with shape (nFilters, hOut, wOut)
func (cl *convLayer) BackwardPropagation(x []*mat.Dense, Z, dA []*mat.Dense) []*mat.Dense {
	stride := cl.Stride
	W := cl.Parameters.W

	nFilters := len(W)
	nChannels := len(W[0])
	filterSize, _ := W[0][0].Dims()

	// initialize gradient for input x
	// input dxPrev with shape (nChannels, hIn, wIn)
	dxPrev := make([]*mat.Dense, nChannels)
	for i := range dxPrev {
		dxPrev[i] = mat.NewDense(cl.InputShape[1], cl.InputShape[2], nil)
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	workers := make(chan struct{}, 16)

	applyActivationDerivative := func(_, _ int, v float64) float64 { return cl.Activation.Derivative(v) }
	for f := 0; f < nFilters; f++ {
		wg.Add(1)
		workers <- struct{}{}
		go func(f int) {
			defer wg.Done()
			defer func() { <-workers }()

			// apply gradient of activation function to dA
			dZ := ngo.Multiply(dA[f], ngo.Apply(applyActivationDerivative, Z[f]))
			for c := 0; c < nChannels; c++ {
				// gradient of Z with respect to W
				convolve := Convolve2D(x[c], dZ, stride)
				mu.Lock()
				cl.Gradients.DW[f][c] = ngo.Add(cl.Gradients.DW[f][c], convolve)
				mu.Unlock()

				// gradient of Z with respect to x
				convolve = Convolve2D(
					ngo.ZeroPadding(dZ, filterSize-1),
					ngo.Rotate180(W[f][c]),
					stride)
				mu.Lock()
				dxPrev[c] = ngo.Add(dxPrev[c], convolve)
				mu.Unlock()
			}

			mu.Lock()
			cl.Gradients.DB.Set(f, 0, cl.Gradients.DB.At(f, 0)+mat.Sum(dZ))
			mu.Unlock()
		}(f)
	}
	wg.Wait()

	return dxPrev
}

// update parameters (optimization algorithm)
func (cl *convLayer) UpdateParameters(learningRate float64) {
	cl.Parameters = cl.Optimizer.Function(&cl.Optimizer, cl.Parameters, cl.Gradients.DW,
		cl.Gradients.DB, learningRate, cl.Iter)
	cl.Iter++

	// reset gradients
	for i := range cl.Gradients.DW {
		for j := range cl.Gradients.DW[i] {
			rows, cols := cl.Gradients.DW[i][j].Dims()
			cl.Gradients.DW[i][j] = mat.NewDense(rows, cols, nil)
		}
	}
	cl.Gradients.DB = mat.NewDense(cl.NFilters, 1, nil)
}
