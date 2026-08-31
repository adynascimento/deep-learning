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
// row is an observation and each column is a variable.
// matrix shape (nSamples, nFeatures)
func (s *standardScaler) Fit(m mat.Matrix) {
	data := mat.DenseCopyOf(m)
	_, cols := data.Dims()

	s.mean = make([]float64, cols)
	s.stdDev = make([]float64, cols)
	for j := 0; j < cols; j++ {
		s.mean[j], s.stdDev[j] = stat.PopMeanStdDev(mat.Col(nil, j, data), nil)
	}
}

// perform standardization by centering and scaling
func (s *standardScaler) Transform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	standardized := mat.NewDense(rows, cols, nil)
	for j := 0; j < cols; j++ {
		if s.stdDev[j] != 0 {
			for i := 0; i < rows; i++ {
				standardized.Set(i, j, (data.At(i, j)-s.mean[j])/s.stdDev[j])
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
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			inversed.Set(i, j, data.At(i, j)*s.stdDev[j]+s.mean[j])
		}
	}

	return inversed
}

// get the mean of each feature
func (s *standardScaler) GetMean() []float64 {
	return s.mean
}

// get the biased standard deviation of each feature
func (s *standardScaler) GetStdDev() []float64 {
	return s.stdDev
}
