package cnn

import (
	"gonum.org/v1/gonum/mat"
)

type flatten struct {
	Shape []int // pool output with shape (nTraining, nFilters, hOut, wOut)
}

func newFlatten() *flatten {
	return &flatten{}
}

// forward propagation step: flatten operation
// input x (pool output) with shape (nTraining, nFilters, hOut, wOut)
func (fl *flatten) ForwardPropagation(x [][]*mat.Dense) *mat.Dense {
	nTraining := len(x)
	nFilters := len(x[0])
	hOut, wOut := x[0][0].Dims()

	// store the input shape
	fl.Shape = append(fl.Shape, nTraining, nFilters, hOut, wOut)

	result := mat.NewDense(nFilters*hOut*wOut, nTraining, nil)
	for i := 0; i < nTraining; i++ {
		for j := 0; j < nFilters; j++ {
			// Flatten the matrix (hOut, wOut) into a vector
			flatPatch := make([]float64, 0, hOut*wOut)
			for k := 0; k < hOut; k++ {
				for l := 0; l < wOut; l++ {
					flatPatch = append(flatPatch, x[i][j].At(k, l))
				}
			}

			// fills row i of the resulting matrix with the flattened vector
			rowOffset := j * hOut * wOut
			for k := 0; k < len(flatPatch); k++ {
				result.Set(rowOffset+k, i, flatPatch[k])
			}
		}
	}

	return result
}

// backward propagation step: flatten operation
// gradient dA with shape (nTraining, nFilters*rows*cols)
func (fl *flatten) BackwardPropagation(dA *mat.Dense) [][]*mat.Dense {
	nTraining := fl.Shape[0]
	nFilters := fl.Shape[1]
	hOut := fl.Shape[2]
	wOut := fl.Shape[3]

	// initialize the original structure
	x := make([][]*mat.Dense, nTraining)
	for i := range x {
		x[i] = make([]*mat.Dense, nFilters)
		for j := range x[i] {
			x[i][j] = mat.NewDense(hOut, wOut, nil)
		}
	}

	// reshape the gradient back to the original shape
	for i := 0; i < nTraining; i++ {
		for j := 0; j < nFilters; j++ {
			rowOffset := j * hOut * wOut
			for k := 0; k < hOut; k++ {
				for l := 0; l < wOut; l++ {
					value := dA.At(rowOffset+k*wOut+l, i)
					x[i][j].Set(k, l, value)
				}
			}
		}
	}

	return x
}
