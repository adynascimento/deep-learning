package numeric

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// generate linearly spaced slice of float64
func Linspace(start, stop float64, num int) []float64 {
	var step float64
	if num == 1 {
		return []float64{start}
	}
	step = (stop - start) / float64(num-1)

	r := make([]float64, num)
	for i := 0; i < num; i++ {
		r[i] = start + float64(i)*step
	}
	return r
}

// sum rows of a matrix
func SumRows(a *mat.Dense) *mat.Dense {
	row := []float64{}
	for i := 0; i < a.RawMatrix().Rows; i++ {
		var sum float64
		for _, v := range a.RawRowView(i) {
			sum = sum + v
		}
		row = append(row, sum)
	}

	return mat.NewDense(a.RawMatrix().Rows, 1, row)
}

// add matrix with column vector
func AddMatrixVector(a *mat.Dense, b *mat.Dense) *mat.Dense {
	m := new(mat.Dense)
	fn := func(row, _ int, v float64) float64 { return v + b.At(row, 0) }
	m.Apply(fn, a)

	return m
}

// generate a random slice of float64
func Randn(n, m int) *mat.Dense {
	rand.Seed(time.Now().Unix())
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
