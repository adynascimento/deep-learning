package cnn

import (
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/floats"
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
	NChannels       int
	FilterSize      int
	Stride          int
	Iter            float64
}

type parameters struct {
	W       [][]*mat.Dense // weights with shape (nFilters, nChannels, filterSize, filterSize)
	B       *mat.Dense     // biases with shape (nFilters, 1)
	wRotBig *mat.Dense     // rotated weights with shape (nChannels, nFilters*filterSize*filterSize)
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

	// K is the number of weights per input channel
	K := filterSize * filterSize

	// wRotBig will have a shape: nChannels x nFilters*K
	// this causes MatMul to sum, all at once, the contribution of all filters to channel c.
	// rotate weights for backpropagation
	wRotBig := mat.NewDense(nChannels, nFilters*K, nil)
	for c := 0; c < nChannels; c++ {
		row := wRotBig.RawRowView(c)
		for f := 0; f < nFilters; f++ {
			rot := ngo.Rotate180(filters[f][c])
			copy(
				row[f*K:(f+1)*K],
				rot.RawMatrix().Data,
			)
		}
	}

	// initialize gradients
	gradients := newGradients(nFilters, nChannels, filterSize)

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
			W:       filters,
			B:       mat.NewDense(nFilters, 1, nil),
			wRotBig: wRotBig,
		},
		Activation: activation,
		Gradients:  gradients,
		Optimizer:  optimizer,
		NFilters:   nFilters,
		NChannels:  nChannels,
		FilterSize: filterSize,
		Stride:     stride,
		Iter:       1,
	}
}

func newGradients(nFilters, nChannels, filterSize int) gradients {
	// initialize gradients
	dW := make([][]*mat.Dense, nFilters) // derivatives of the weigths W
	for f := 0; f < nFilters; f++ {
		dW[f] = make([]*mat.Dense, nChannels)
		for c := 0; c < nChannels; c++ {
			dW[f][c] = mat.NewDense(filterSize, filterSize, nil)
		}
	}

	return gradients{
		DW: dW,
		DB: mat.NewDense(nFilters, 1, nil),
	}
}

// forward propagation step: convolution operation
// input x with shape (nChannels, hIn, wIn)
func (cl *convLayer) ForwardPropagation(x []*mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	stride := cl.Stride
	W := cl.Parameters.W
	b := cl.Parameters.B

	nFilters := cl.NFilters
	nChannels := cl.NChannels

	// output dimension
	hOut := cl.OutputShape[1]
	wOut := cl.OutputShape[2]

	// conv output with shape (nFilters, hOut, wOut)
	Z := make([]*mat.Dense, nFilters) // linear function
	A := make([]*mat.Dense, nFilters) // activation function

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// P is the number of spatial positions in the output of the original convolution
	P := hOut * wOut

	// xBig will have a shape: nChannels*K x P.
	// each block of K rows contains the Im2Col representation of one input channel.
	// stacking all channels allows the convolution over every channel to be computed as one GEMM
	xBig := mat.NewDense(nChannels*K, P, nil)
	for c := 0; c < nChannels; c++ {
		// xCol will have a shape: K x P
		xCol := Im2Col(x[c], cl.FilterSize, cl.FilterSize, stride)

		raw := xCol.RawMatrix()
		for k := 0; k < K; k++ {
			copy(
				xBig.RawRowView(c*K+k),
				raw.Data[k*raw.Stride:k*raw.Stride+P],
			)
		}
	}

	// wBig will have a shape: nFilters x nChannels*K.
	// each row contains one filter flattened across all input channels.
	// each block of K values corresponds to one input channel.
	wBig := mat.NewDense(nFilters, nChannels*K, nil)
	for f := 0; f < nFilters; f++ {
		row := wBig.RawRowView(f)
		for c := 0; c < nChannels; c++ {
			copy(
				row[c*K:(c+1)*K],
				W[f][c].RawMatrix().Data,
			)
		}
	}

	// batched version of all convolutions using Im2Col + GEMM.
	// shapes:
	//     wBig: nFilters x nChannels*K
	//     xBig: nChannels*K x P
	// zBig will have a shape: nFilters x P.
	// each row contains the flattened output feature map of one filter.
	zBig := ngo.MatMul(wBig, xBig)

	for f := 0; f < nFilters; f++ {
		z := mat.NewDense(hOut, wOut, nil)
		copy(
			z.RawMatrix().Data,
			zBig.RawRowView(f),
		)

		// add the bias corresponding to filter f
		floats.AddConst(b.At(f, 0), z.RawMatrix().Data)

		Z[f] = z
		A[f] = cl.Activation.Function(z)
	}

	return Z, A
}

// backward propagation step: reverse convolution operation
// input x with shape (nChannels, hIn, wIn)
// gradient dA with shape (nFilters, hOut, wOut)
func (cl *convLayer) BackwardPropagation(x []*mat.Dense, Z, dA []*mat.Dense, workerGradient *gradients) []*mat.Dense {
	stride := cl.Stride
	nFilters := cl.NFilters

	// compute all dZ once in relation to the linear output Z of each filter
	dZs := make([]*mat.Dense, nFilters)
	for f := 0; f < nFilters; f++ {
		// apply gradient of activation function to dA: dZ[f] = dA[f] ⊙ activation'(Z[f])
		dZ := ngo.Multiply(dA[f], cl.Activation.Derivative(Z[f]))

		dZs[f] = dZ
		workerGradient.DB.Set(f, 0, workerGradient.DB.At(f, 0)+mat.Sum(dZ))
	}

	// propagate dZ to the filters W.
	// equivalent to dW[f][c] = Convolve2D(x[c], dZs[f]), computed as one GEMM
	// store the result in cl.Gradients.DW for all filters and channels.
	cl.BackwardWeightGradients(x, dZs, stride, workerGradient)

	// propagate dZ to the previous input x.
	// equivalent to summing Convolve2D(Pad(dZs[f]), Rotate180(W[f][c])) over filters.
	// Rotate180(W[f][c]) is precomputed in cl.Parameters.wRotBig for all filters and channels.
	dxPrev := cl.BackwardInputGradients(dZs, stride)

	return dxPrev
}

// compute dW with the convolution gradients for W.
// it is equivalent to: dW[f][c] = Convolve2D(x[c], dZs[f], stride),
// computed in batch with im2col + GEMM for performance
func (cl *convLayer) BackwardWeightGradients(x, dZs []*mat.Dense, stride int, workerGradient *gradients) {
	nFilters := cl.NFilters
	nChannels := cl.NChannels

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// P is the number of spatial positions in the output of the original convolution
	// it is also the number of elements in each dZ[f]
	dZRows, dZCols := dZs[0].Dims()
	P := dZRows * dZCols

	// dZBig will have a shape: nFilters x P
	// each line is a flattened dZ[f]
	dZBig := mat.NewDense(nFilters, P, nil)
	for f := 0; f < nFilters; f++ {
		copy(
			dZBig.RawRowView(f),
			dZs[f].RawMatrix().Data,
		)
	}

	// xBig will have a shape: nChannels*K x P
	xBig := mat.NewDense(nChannels*K, P, nil)
	for c := 0; c < nChannels; c++ {
		// xCol will have a shape: K x P
		xCol := Im2Col(x[c], cl.FilterSize, cl.FilterSize, stride)

		raw := xCol.RawMatrix()
		for k := 0; k < K; k++ {
			copy(
				xBig.RawRowView(c*K+k),
				raw.Data[k*raw.Stride:k*raw.Stride+P],
			)
		}
	}

	// batched version of all dW convolutions using im2col + GEMM.
	// shapes:
	//     dZStack: nFilters x P
	//     xBig.T:  P x nChannels*K
	// dWBig will have a shape: nFilters x nChannels*K
	// each line of dWBig contains all flattened dW[f][c], a block of K values per channel.
	dWBig := ngo.MatMul(dZBig, xBig.T())

	// dWBig stores all weight gradients flattened
	for f := 0; f < nFilters; f++ {
		row := dWBig.RawRowView(f)
		for c := 0; c < nChannels; c++ {
			dst := workerGradient.DW[f][c].RawMatrix().Data
			src := row[c*K : (c+1)*K] // block corresponding to channel c

			for i := range dst {
				dst[i] += src[i]
			}
		}
	}
}

// compute dxPrev with the convolution gradients for the input x
// it is equivalent to: dxPrev[c] += Convolve2D(ZeroPadding(dZs[f], filterSize-1), Rotate180(W[f][c]), stride)
// for every filter f and input channel c, computed in batch with im2col + GEMM for performance.
func (cl *convLayer) BackwardInputGradients(dZs []*mat.Dense, stride int) []*mat.Dense {
	nFilters := cl.NFilters
	nChannels := cl.NChannels

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// padding dZ and convolving with rotated filters gives the full convolution
	// necessary to recover a dx with the spatial size of the original input
	padCols := make([]*mat.Dense, nFilters)
	for f := 0; f < nFilters; f++ {
		dZ := dZs[f]
		if stride > 1 {
			dZ = UpSampleZeros2D(dZ, stride)
		}

		padDZ := ngo.ZeroPadding(dZ, cl.FilterSize-1)
		padCols[f] = Im2Col(padDZ, cl.FilterSize, cl.FilterSize, stride)
	}

	// Q is the quantity of spatial positions of dx.
	Q := padCols[0].RawMatrix().Cols

	// dZBig will have a shape: nFilters*K x Q
	// for each filter f, we stack all the patches of dZ[f] with padding.
	// each block of K lines corresponds to a filter f.
	dZBig := mat.NewDense(nFilters*K, Q, nil)
	for f := 0; f < nFilters; f++ {
		cols := padCols[f].RawMatrix()
		for k := 0; k < K; k++ {
			copy(
				dZBig.RawRowView(f*K+k),
				cols.Data[k*cols.Stride:k*cols.Stride+Q],
			)
		}
	}

	// this multiplication is the batch version.
	// shapes:
	//     wRotBig: nChannels x nFilters*K
	//     dZBig:   nFilters*K x Q
	// dxBig will have a shape: nChannels x Q
	// each line is the dx of a channel, flattened.
	dxBig := ngo.MatMul(cl.Parameters.wRotBig, dZBig)

	// initialize gradient for input x
	// input dxPrev with shape (nChannels, hIn, wIn)
	// copy the flattened result to the dxPrev
	dxPrev := make([]*mat.Dense, nChannels)
	for c := 0; c < nChannels; c++ {
		dxPrev[c] = mat.NewDense(cl.InputShape[1], cl.InputShape[2], nil)
		copy(
			dxPrev[c].RawMatrix().Data,
			dxBig.RawRowView(c),
		)
	}

	return dxPrev
}

// accumulate the local gradients from all workers into the layer gradients
func (cl *convLayer) ReduceWorkerGradients(workerGradients []gradients) {
	convGradients := &cl.Gradients

	for w := range workerGradients {
		workerGradient := &workerGradients[w]

		convGradients.DB.Add(convGradients.DB, workerGradient.DB)
		for f := 0; f < cl.NFilters; f++ {
			for c := 0; c < cl.NChannels; c++ {
				convGradients.DW[f][c].Add(convGradients.DW[f][c], workerGradient.DW[f][c])
			}
		}

		// reset worker gradients
		for f := range workerGradient.DW {
			for c := range workerGradient.DW[f] {
				workerGradient.DW[f][c].Zero()
			}
		}
		workerGradient.DB.Zero()
	}
}

// update parameters (optimization algorithm)
func (cl *convLayer) UpdateParameters(learningRate float64) {
	cl.Parameters = cl.Optimizer.Function(&cl.Optimizer, cl.Parameters, cl.Gradients.DW,
		cl.Gradients.DB, learningRate, cl.Iter)
	cl.Iter++

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// rotate weights after update for backpropagation
	for c := 0; c < cl.NChannels; c++ {
		row := cl.Parameters.wRotBig.RawRowView(c)
		for f := 0; f < cl.NFilters; f++ {
			rot := ngo.Rotate180(cl.Parameters.W[f][c])
			copy(
				row[f*K:(f+1)*K],
				rot.RawMatrix().Data,
			)
		}
	}

	// reset gradients
	for i := range cl.Gradients.DW {
		for j := range cl.Gradients.DW[i] {
			cl.Gradients.DW[i][j].Zero()
		}
	}
	cl.Gradients.DB.Zero()
}
