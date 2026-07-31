package cnn

import (
	"bytes"
	"encoding/gob"
	"math"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type convLayer struct {
	InputShape      [3]int
	OutputShape     [3]int
	TrainableParams int
	Parameters      parameters
	Activation      nncore.Activation
	Gradients       gradients
	Optimizer       nncore.Optimizer
	NFilters        int
	NChannels       int
	FilterSize      int
	Stride          int
	Iter            float64
}

type parameters struct {
	W    [][]*mat.Dense // weights with shape (nFilters, nChannels, filterSize, filterSize)
	B    *mat.Dense     // biases with shape (nFilters, 1)
	WBig *mat.Dense     // flattened weights with shape (nFilters, nChannels*filterSize*filterSize)
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

type convLayerConfig struct {
	convConfig
	Activation nncore.Activation
	Optimizer  nncore.OptimizerType
}

func newConvLayer(config convLayerConfig) *convLayer {
	nChannels := config.InputShape[0]

	// initialize convolutional neural network
	// filters with shape (nFilters, nChannels, filterSize, filterSize)
	filters := initializeConvParameters(config.NFilters, nChannels, config.FilterSize, config.Activation)

	// K is the number of weights per input channel
	K := config.FilterSize * config.FilterSize

	// wBig will have a shape: nFilters x nChannels*K.
	// each row contains one filter flattened across all input channels.
	// each block of K values corresponds to one input channel.
	wBig := mat.NewDense(config.NFilters, nChannels*K, nil)
	for f := 0; f < config.NFilters; f++ {
		row := wBig.RawRowView(f)
		for c := 0; c < nChannels; c++ {
			copy(
				row[c*K:(c+1)*K],
				filters[f][c].RawMatrix().Data,
			)
		}
	}

	// initialize gradients
	gradients := newGradients(config.NFilters, nChannels, config.FilterSize)

	// choice of optimization algorithm
	optimizer := nncore.NewOptimizer(config.Optimizer)

	return &convLayer{
		InputShape:      config.InputShape,
		OutputShape:     config.OutputShape,
		TrainableParams: config.NFilters * (config.FilterSize*config.FilterSize*nChannels + 1),
		Parameters: parameters{
			W:    filters,
			B:    mat.NewDense(config.NFilters, 1, nil),
			WBig: wBig,
		},
		Activation: config.Activation,
		Gradients:  gradients,
		Optimizer:  optimizer,
		NFilters:   config.NFilters,
		NChannels:  nChannels,
		FilterSize: config.FilterSize,
		Stride:     config.Stride,
		Iter:       1,
	}
}

func initializeConvParameters(nFilters, nChannels, filterSize int, activation nncore.Activation) [][]*mat.Dense {
	fanIn := nChannels * filterSize * filterSize
	fanOut := nFilters * filterSize * filterSize

	scalar := 1.0
	switch activation.Name {
	case nncore.ReLUActivation, nncore.EluActivation:
		scalar = math.Sqrt(2.0 / float64(fanIn)) // He (Kaiming)
	default: // tanh, sigmoid, softmax, linear...
		scalar = math.Sqrt(2.0 / float64(fanIn+fanOut)) // Xavier (Glorot)
	}

	// initialize convolutional neural network
	// filters with shape (nFilters, nChannels, filterSize, filterSize)
	filters := make([][]*mat.Dense, nFilters)
	for i := range filters {
		filters[i] = make([]*mat.Dense, nChannels)
		for j := range filters[i] {
			filters[i][j] = ngo.Scale(scalar, ngo.Randn(filterSize, filterSize))
		}
	}

	return filters
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
		xCol := ngo.Im2Col(x[c], cl.FilterSize, cl.FilterSize, stride)

		raw := xCol.RawMatrix()
		for k := 0; k < K; k++ {
			copy(
				xBig.RawRowView(c*K+k),
				raw.Data[k*raw.Stride:k*raw.Stride+P],
			)
		}
	}

	// batched version of all convolutions using Im2Col + GEMM.
	// shapes:
	//     wBig: nFilters x nChannels*K
	//     xBig: nChannels*K x P
	// zBig will have a shape: nFilters x P.
	// each row contains the flattened output feature map of one filter.
	zBig := ngo.MatMul(cl.Parameters.WBig, xBig)
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
func (cl *convLayer) BackwardPropagation(x []*mat.Dense, Z, dA []*mat.Dense, workerGradient *gradients, lambd float64) []*mat.Dense {
	stride := cl.Stride
	nFilters := cl.NFilters

	// P is the number of spatial positions in the output of the original convolution
	// it is also the number of columns produced by Im2Col in the forward pass
	dARows, dACols := dA[0].Dims()
	P := dARows * dACols

	// dZBig will have a shape: nFilters x P
	// each line is a flattened dZ[f]
	dZBig := mat.NewDense(nFilters, P, nil)

	// compute dZ and stack all filter gradients into dZBig
	// dA received from the dense layer is already scaled by 1./batchsize.
	for f := 0; f < nFilters; f++ {
		// apply gradient of activation function to dA: dZ[f] = dA[f] ⊙ activation'(Z[f])
		dZ := ngo.Multiply(dA[f], cl.Activation.Derivative(Z[f]))

		// stack dZ directly
		copy(
			dZBig.RawRowView(f),
			dZ.RawMatrix().Data,
		)

		workerGradient.DB.Set(f, 0, workerGradient.DB.At(f, 0)+mat.Sum(dZ))
	}

	// propagate dZ to the filters W.
	// equivalent to dW[f][c] = Convolve2D(x[c], dZs[f]), computed as one GEMM
	// store the result in cl.Gradients.DW for all filters and channels.
	cl.BackwardWeightGradients(x, dZBig, stride, workerGradient, lambd)

	// propagate dZ to the previous input x.
	// equivalent to summing Convolve2D(Pad(dZs[f]), Rotate180(W[f][c])) over filters.
	// but computed in batch with GEMM + Col2Im for performance.
	dxPrev := cl.BackwardInputGradients(dZBig, stride)

	return dxPrev
}

// compute dW with the convolution gradients for W.
// it is equivalent to: dW[f][c] = Convolve2D(x[c], dZs[f], stride),
// computed in batch with im2col + GEMM for performance
func (cl *convLayer) BackwardWeightGradients(x []*mat.Dense, dZBig *mat.Dense, stride int, workerGradient *gradients, lambd float64) {
	nFilters := cl.NFilters
	nChannels := cl.NChannels

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// P is the number of spatial positions in the output of the original convolution
	// it is also the number of columns produced by Im2Col in the forward pass
	P := dZBig.RawMatrix().Cols

	// xBig will have a shape: nChannels*K x P
	xBig := mat.NewDense(nChannels*K, P, nil)
	for c := 0; c < nChannels; c++ {
		// xCol will have a shape: K x P
		xCol := ngo.Im2Col(x[c], cl.FilterSize, cl.FilterSize, stride)

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

			wRaw := cl.Parameters.W[f][c].RawMatrix()
			for i := range dst {
				// add L2 regularization based on the current weight
				dst[i] += src[i] + lambd*wRaw.Data[i]
			}
		}
	}
}

// compute dxPrev with the convolution gradients for the input x
// it is equivalent to: dxPrev[c] += Convolve2D(ZeroPadding(dZs[f], filterSize-1), Rotate180(W[f][c]), stride)
// for every filter f and input channel c, but computed in batch with GEMM + Col2Im for performance.
func (cl *convLayer) BackwardInputGradients(dZBig *mat.Dense, stride int) []*mat.Dense {
	nChannels := cl.NChannels

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// P is the number of spatial positions in the output of the original convolution
	// it is also the number of columns produced by Im2Col in the forward pass
	P := dZBig.RawMatrix().Cols

	// this multiplication is the batch version.
	// shapes:
	//     wBigT:  nChannels*K x nFilters
	//     dZBig:  nFilters x P
	// dxBig will have shape: nChannels*K x P.
	// each input channel, a block of K rows stores the column representation
	// of the input gradients, before reconstruction with Col2Im.
	dxBig := ngo.MatMul(cl.Parameters.WBig.T(), dZBig)

	// initialize gradient for input x
	// input dxPrev with shape (nChannels, hIn, wIn)
	// copy the flattened result to the dxPrev
	dxPrev := make([]*mat.Dense, nChannels)
	for c := 0; c < nChannels; c++ {
		xCol := mat.NewDense(K, P, nil)
		for k := 0; k < K; k++ {
			copy(
				xCol.RawRowView(k),
				dxBig.RawRowView(c*K+k),
			)
		}

		// reconstruct the spatial gradient of this input channel.
		dxPrev[c] = ngo.Col2Im(xCol, cl.InputShape[1], cl.InputShape[2], cl.FilterSize, cl.FilterSize, stride)
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
	cl.Optimizer.Step(cl.TrainableParameters(), learningRate, cl.Iter)
	cl.Iter++

	// K is the number of weights per input channel
	K := cl.FilterSize * cl.FilterSize

	// flattened weights after update
	for f := 0; f < cl.NFilters; f++ {
		row := cl.Parameters.WBig.RawRowView(f)
		for c := 0; c < cl.NChannels; c++ {
			copy(
				row[c*K:(c+1)*K],
				cl.Parameters.W[f][c].RawMatrix().Data,
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

func (cl *convLayer) TrainableParameters() []*nncore.Parameter {
	nFilters := len(cl.Parameters.W)
	nChannels := len(cl.Parameters.W[0])

	params := []*nncore.Parameter{}
	for f := 0; f < nFilters; f++ {
		for c := 0; c < nChannels; c++ {
			params = append(params, &nncore.Parameter{
				Value:    cl.Parameters.W[f][c],
				Gradient: cl.Gradients.DW[f][c],
				Update: func(m *mat.Dense) {
					cl.Parameters.W[f][c] = m
				},
			})
		}
	}

	params = append(params, &nncore.Parameter{
		Value:    cl.Parameters.B,
		Gradient: cl.Gradients.DB,
		Update: func(m *mat.Dense) {
			cl.Parameters.B = m
		},
	})

	return params
}

func (cl *convLayer) MarshalParameters() ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(cl.Parameters); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (cl *convLayer) UnmarshalParameters(data []byte) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(&cl.Parameters)
}
