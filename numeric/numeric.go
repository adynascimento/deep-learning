package numeric

import (
	"math"
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
func AddMatrixVector(M mat.Dense, b *mat.Dense) *mat.Dense {
	out := []float64{}
	for i := 0; i < M.RawMatrix().Rows; i++ {
		row := []float64{}
		for _, v := range M.RawRowView(i) {
			row = append(row, v+b.RawMatrix().Data[i])
		}
		out = append(out, row...)
	}

	return mat.NewDense(M.RawMatrix().Rows, M.RawMatrix().Cols, out)
}

// calculate Tanh of a slice of float64
func Tanh(M *mat.Dense) *mat.Dense {
	values := []float64{}
	for _, v := range M.RawMatrix().Data {
		values = append(values, math.Tanh(v))
	}

	return mat.NewDense(M.RawMatrix().Rows, M.RawMatrix().Cols, values)
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
