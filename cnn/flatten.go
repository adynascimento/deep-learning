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
	fl.Shape = []int{nTraining, nFilters, hOut, wOut}

	result := mat.NewDense(nTraining, nFilters*hOut*wOut, nil)
	for i := 0; i < nTraining; i++ {
		feature := 0
		for j := 0; j < nFilters; j++ {
			// copy the flattened feature map into row i of the output matrix
			src := x[i][j].RawMatrix().Data
			copy(result.RawRowView(i)[feature:], src)
			feature += len(src)
		}
	}

	return result
}

// backward propagation step: flatten operation
// gradient dA with shape (nTraining, nFilters*hOut*wOut)
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

	// reshape the gradient back to the original shape (nTraining, nFilters, hOut, wOut)
	for i := 0; i < nTraining; i++ {
		feature := 0
		row := dA.RawRowView(i)
		for j := 0; j < nFilters; j++ {
			dst := x[i][j].RawMatrix().Data
			copy(dst, row[feature:feature+len(dst)])
			feature += len(dst)
		}
	}

	return x
}
