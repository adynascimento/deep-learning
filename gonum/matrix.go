package gonum

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type directionType string

const (
	OverRows    directionType = "rows"
	OverColumns directionType = "columns"
)

// sum over specific direction of a matrix
func Sum(a *mat.Dense, direction directionType) *mat.Dense {
	m := new(mat.Dense)
	vals := []float64{}

	switch direction {
	case OverRows:
		for j := 0; j < a.RawMatrix().Cols; j++ {
			var sum float64

			col := mat.Col(nil, j, a)
			for _, v := range col {
				sum = sum + v
			}
			vals = append(vals, sum)
		}
		m = mat.NewDense(1, a.RawMatrix().Cols, vals)
	case OverColumns:
		for i := 0; i < a.RawMatrix().Rows; i++ {
			var sum float64

			row := mat.Row(nil, i, a)
			for _, v := range row {
				sum = sum + v
			}
			vals = append(vals, sum)
		}
		m = mat.NewDense(a.RawMatrix().Rows, 1, vals)
	}

	return m
}

// add matrix with column vector row-wise
func AddMatrixVector(a *mat.Dense, b *mat.Dense) *mat.Dense {
	m := new(mat.Dense)
	fn := func(row, _ int, v float64) float64 { return v + b.At(row, 0) }
	m.Apply(fn, a)

	return m
}

// division matrix with row vector column-wise
func DivMatrixVector(a *mat.Dense, b *mat.Dense) *mat.Dense {
	m := new(mat.Dense)
	fn := func(_, col int, v float64) float64 { return v / b.At(0, col) }
	m.Apply(fn, a)

	return m
}

// generate a random slice of float64
func Randn(n, m int) *mat.Dense {
	random := []float64{}
	for i := 0; i < n*m; i++ {
		random = append(random, rand.NormFloat64())
	}

	return mat.NewDense(n, m, random)
}

// applies the function fn to each of the elements of a. The function fn takes a row/column
// index and element value and returns some function of that tuple
func Apply(fn func(i, j int, v float64) float64, a mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Apply(fn, a)

	return m
}

// multiply arguments element-wise by a scalar
func Scale(f float64, a mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Scale(f, a)

	return m
}

// addition arguments, element-wise.
func Add(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Add(a, b)

	return m
}

// division arguments, element-wise.
func DivElem(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.DivElem(a, b)

	return m
}

// subtract arguments, element-wise.
func Sub(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Sub(a, b)

	return m
}

// matrix product of two arrays
func MatMul(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Mul(a, b)

	return m
}

// return the element-wise square of the input.
func Square(a mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.MulElem(a, a)

	return m
}

// multiply arguments element-wise
func Multiply(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.MulElem(a, b)

	return m
}
