package ngo

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type StandardScaler interface {
	Fit(m mat.Matrix)
	Transform(m mat.Matrix) *mat.Dense
	FitTransform(m mat.Matrix) *mat.Dense
	InverseTransform(m mat.Matrix) *mat.Dense
	GetMean() []float64
	GetStdDev() []float64
}

type standardScaler struct {
	mean   []float64
	stdDev []float64
}

func NewStandardScaler() StandardScaler {
	return &standardScaler{}
}

// performs a standardize features by removing the mean and
// scaling to unit variance on the matrix of the input data
// which is represented as an rows X cols matrix a where each
// row is a variable and each column is an observation.
// matrix shape (nFeatures, nSamples)
func (s *standardScaler) Fit(m mat.Matrix) {
	data := mat.DenseCopyOf(m)
	rows, _ := data.Dims()

	s.mean = make([]float64, rows)
	s.stdDev = make([]float64, rows)
	for i := 0; i < rows; i++ {
		s.mean[i], s.stdDev[i] = stat.PopMeanStdDev(mat.Row(nil, i, data), nil)
	}
}

// perform standardization by centering and scaling
func (s *standardScaler) Transform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	standardized := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		if s.stdDev[i] != 0 {
			for j := 0; j < cols; j++ {
				standardized.Set(i, j, (data.At(i, j)-s.mean[i])/s.stdDev[i])
			}
		}
	}

	return standardized
}

// FitTransform is exactly equivalent to calling Fit()
// followed by Transform()
func (s *standardScaler) FitTransform(m mat.Matrix) *mat.Dense {
	s.Fit(m)
	return s.Transform(m)
}

// scale back the data to the original representation
func (s *standardScaler) InverseTransform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	inversed := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			inversed.Set(i, j, (data.At(i, j)*s.stdDev[i])+s.mean[i])
		}
	}

	return inversed
}

// get the sample mean
func (s *standardScaler) GetMean() []float64 {
	return s.mean
}

// get the biased standard deviation
func (s *standardScaler) GetStdDev() []float64 {
	return s.stdDev
}
