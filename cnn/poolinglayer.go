package cnn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type poolLayer struct {
	InputShape  [3]int
	OutputShape [3]int
	Size        int
	Stride      int
}

func newPoolLayer(size, stride int, inputShape [3]int) *poolLayer {
	// output dimension
	hOut := (inputShape[1]-size)/stride + 1
	wOut := (inputShape[2]-size)/stride + 1

	return &poolLayer{
		InputShape:  inputShape,
		OutputShape: [3]int{inputShape[0], hOut, wOut},
		Size:        size,
		Stride:      stride,
	}
}

// forward propagation step: pooling operation
// input x (conv output) with shape (nTraining, nFilters, hIn, wIn)
func (pl *poolLayer) ForwardPropagation(x [][]*mat.Dense) [][]*mat.Dense {
	size := pl.Size
	stride := pl.Stride

	// output dimension
	hOut := pl.OutputShape[1]
	wOut := pl.OutputShape[2]

	// pool output with shape (nTraining, nFilters, hOut, wOut)
	nTraining := len(x)
	nFilters := len(x[0])
	A := make([][]*mat.Dense, nTraining) // output of the pool layer
	for i := range A {
		A[i] = make([]*mat.Dense, nFilters)
		for j := range A[i] {
			A[i][j] = mat.NewDense(hOut, wOut, nil)
		}
	}

	for t := 0; t < nTraining; t++ {
		for f := 0; f < nFilters; f++ {
			xValue := x[t][f].RawMatrix()
			for i := 0; i < hOut; i++ {
				for j := 0; j < wOut; j++ {
					max := -math.MaxFloat64
					for k := 0; k < size; k++ {
						for l := 0; l < size; l++ {
							index := (i*stride+k)*xValue.Cols + (j*stride + l)
							if xValue.Data[index] > max {
								max = xValue.Data[index]
							}
						}
					}
					A[t][f].Set(i, j, max)
				}
			}
		}
	}

	return A
}

// backward propagation step: pooling operation
// input x (conv output) with shape (nTraining, nFilters, hIn, wIn)
// gradient dA with shape (nTraining, nFilters, hOut, wOut)
func (pl *poolLayer) BackwardPropagation(x [][]*mat.Dense, dA [][]*mat.Dense) [][]*mat.Dense {
	size := pl.Size
	stride := pl.Stride

	// output dimension
	hOut := pl.OutputShape[1]
	wOut := pl.OutputShape[2]

	// initialize gradient for input x
	// input dxPrev with shape (nTraining, nFilters, hIn, wIn)
	nTraining := len(x)
	nFilters := len(x[0])
	dxPrev := make([][]*mat.Dense, nTraining)
	for i := range dxPrev {
		dxPrev[i] = make([]*mat.Dense, nFilters)
		for j := range dxPrev[i] {
			dxPrev[i][j] = mat.NewDense(pl.InputShape[1], pl.InputShape[2], nil)
		}
	}

	for t := 0; t < nTraining; t++ {
		for f := 0; f < nFilters; f++ {
			xValue := x[t][f].RawMatrix()
			dAValue := dA[t][f].RawMatrix()
			for i := 0; i < hOut; i++ {
				for j := 0; j < wOut; j++ {
					max := -math.MaxFloat64
					maxIndex := [2]int{0, 0}
					for k := 0; k < size; k++ {
						for l := 0; l < size; l++ {
							index := (i*stride+k)*xValue.Cols + (j*stride + l)
							if xValue.Data[index] > max {
								max = xValue.Data[index]
								maxIndex = [2]int{i*stride + k, j*stride + l}
							}
						}
					}
					dxPrev[t][f].Set(maxIndex[0], maxIndex[1], dAValue.Data[i*wOut+j])
				}
			}
		}
	}

	return dxPrev
}
