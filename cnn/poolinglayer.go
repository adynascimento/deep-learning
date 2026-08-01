package cnn

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type poolLayer struct {
	InputShape  [3]int `json:"input_shape"`
	OutputShape [3]int `json:"output_shape"`
	Size        int    `json:"size"`
	Stride      int    `json:"stride"`
}

// cache of max indices for backward propagation, shape (nFilters, hOut*wOut)
type poolCache struct {
	MaxIndices [][]int
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
// input x (conv output) with shape (nFilters, hIn, wIn)
func (pl *poolLayer) ForwardPropagation(x []*mat.Dense) ([]*mat.Dense, *poolCache) {
	switch {
	case pl.Size == 2 && pl.Stride == 2:
		// optimized pooling operation for size=2 and stride=2
		return pl.MaxPool2x2(x)
	default:
		// default pooling operation for any size and stride
		return pl.MaxPoolGeneric(x)
	}
}

// forward propagation step: pooling operation for size=2 and stride=2 (optimized for this case)
func (pl *poolLayer) MaxPool2x2(x []*mat.Dense) ([]*mat.Dense, *poolCache) {
	// output dimension
	hOut := pl.OutputShape[1]
	wOut := pl.OutputShape[2]

	// pool output with shape (nFilters, hOut, wOut)
	nFilters := len(x)
	A := make([]*mat.Dense, nFilters) // output of the pool layer
	poolCache := &poolCache{          // cache of max indices for backward propagation
		MaxIndices: make([][]int, nFilters),
	}

	for f := 0; f < nFilters; f++ {
		A[f] = mat.NewDense(hOut, wOut, nil)
		poolCache.MaxIndices[f] = make([]int, hOut*wOut)
		cache := poolCache.MaxIndices[f]

		xRaw := x[f].RawMatrix()
		aRaw := A[f].RawMatrix()

		cols := xRaw.Cols
		xValue := xRaw.Data
		aValue := aRaw.Data
		outIdx := 0
		for i := 0; i < hOut; i++ {
			row := (i * 2) * cols
			for j := 0; j < wOut; j++ {
				idx := row + (j * 2)
				max := xValue[idx]
				maxIdx := idx
				if xValue[idx+1] > max {
					max = xValue[idx+1]
					maxIdx = idx + 1
				}
				if xValue[idx+cols] > max {
					max = xValue[idx+cols]
					maxIdx = idx + cols
				}
				if xValue[idx+cols+1] > max {
					max = xValue[idx+cols+1]
					maxIdx = idx + cols + 1
				}
				aValue[outIdx] = max
				cache[outIdx] = maxIdx
				outIdx++
			}
		}
	}

	return A, poolCache
}

func (pl *poolLayer) MaxPoolGeneric(x []*mat.Dense) ([]*mat.Dense, *poolCache) {
	size := pl.Size
	stride := pl.Stride

	// output dimension
	hOut := pl.OutputShape[1]
	wOut := pl.OutputShape[2]

	// pool output with shape (nFilters, hOut, wOut)
	nFilters := len(x)
	A := make([]*mat.Dense, nFilters) // output of the pool layer
	poolCache := &poolCache{          // cache of max indices for backward propagation
		MaxIndices: make([][]int, nFilters),
	}

	for f := 0; f < nFilters; f++ {
		A[f] = mat.NewDense(hOut, wOut, nil)
		poolCache.MaxIndices[f] = make([]int, hOut*wOut)
		cache := poolCache.MaxIndices[f]

		xValue := x[f].RawMatrix()
		aValue := A[f].RawMatrix()
		outIdx := 0
		for i := 0; i < hOut; i++ {
			row := i * stride
			for j := 0; j < wOut; j++ {
				maxIdx := 0
				max := -math.MaxFloat64
				col := j * stride
				for k := 0; k < size; k++ {
					for l := 0; l < size; l++ {
						idx := (row+k)*xValue.Cols + (col + l)
						if xValue.Data[idx] > max {
							max = xValue.Data[idx]
							maxIdx = idx
						}
					}
				}
				aValue.Data[outIdx] = max
				cache[outIdx] = maxIdx
				outIdx++
			}
		}
	}

	return A, poolCache
}

// backward propagation step: pooling operation
// gradient dA with shape (nFilters, hOut, wOut)
func (pl *poolLayer) BackwardPropagation(dA []*mat.Dense, cache *poolCache) []*mat.Dense {
	// initialize gradient for input x
	// input dxPrev with shape (nFilters, hIn, wIn)
	nFilters := len(dA)
	dxPrev := make([]*mat.Dense, nFilters)

	for f := 0; f < nFilters; f++ {
		dxPrev[f] = mat.NewDense(pl.InputShape[1], pl.InputShape[2], nil)

		dAValue := dA[f].RawMatrix().Data
		dxValue := dxPrev[f].RawMatrix().Data

		maxIndices := cache.MaxIndices[f]
		for i := range maxIndices {
			dxValue[maxIndices[i]] += dAValue[i]
		}
	}

	return dxPrev
}
