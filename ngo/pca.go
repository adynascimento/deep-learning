package ngo

import (
	"log"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type PCA interface {
	Fit(m mat.Matrix)
	Transform(m mat.Matrix) *mat.Dense
	FitTransform(m mat.Matrix) *mat.Dense
	InverseTransform(m mat.Matrix) *mat.Dense
	GetComponents() *mat.Dense
	GetExplainedVariance() []float64
}

type pca struct {
	nComponents   int
	components    *mat.Dense
	varianceRatio []float64
	mean          []float64
}

func NewPCA(nComponents int) PCA {
	return &pca{
		nComponents: nComponents,
	}
}

// performs a principal components analysis on the matrix of the input data
// which is represented as an rows X cols matrix a where each
// row is an observation and each column is a variable.
// matrix shape (nSamples, nFeatures)
func (p *pca) Fit(m mat.Matrix) {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the mean of each feature
	p.mean = make([]float64, cols)
	for j := 0; j < cols; j++ {
		mean := stat.Mean(mat.Col(nil, j, data), nil)
		p.mean[j] = mean
		for i := 0; i < rows; i++ {
			data.Set(i, j, data.At(i, j)-mean)
		}
	}

	// calculate the SVD decomposition
	svd := &mat.SVD{}
	if ok := svd.Factorize(data, mat.SVDFull); !ok {
		log.Fatal("error in SVD decomposition")
	}

	// get the eigenvectors (principal components)
	var dst mat.Dense
	svd.VTo(&dst)

	// v contains the right singular vectors as columns.
	// store the principal components as rows.
	p.components = mat.DenseCopyOf(dst.Slice(0, cols, 0, p.nComponents).T())

	// calculate variance ratio
	singularValues := svd.Values(nil)
	floats.MulTo(singularValues, singularValues, singularValues)
	floats.ScaleTo(singularValues, 1.0/float64(rows-1), singularValues)

	totalVariance := floats.Sum(singularValues)
	p.varianceRatio = make([]float64, len(singularValues[:p.nComponents]))
	for i, singularValue := range singularValues[:p.nComponents] {
		p.varianceRatio[i] = singularValue / totalVariance
	}
}

// projects the data into principal component space
// the returned matrix will be of reduced dimensionality
// the returned matrix has shape (nSamples, nComponents)
func (p *pca) Transform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the feature means
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			data.Set(i, j, data.At(i, j)-p.mean[j])
		}
	}

	return MatMul(data, p.components.T())
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform()
func (p *pca) FitTransform(m mat.Matrix) *mat.Dense {
	p.Fit(m)
	return p.Transform(m)
}

// transforms data back to the original space
// the returned matrix has shape (nSamples, nFeatures)
func (p *pca) InverseTransform(m mat.Matrix) *mat.Dense {
	rows, _ := m.Dims()
	_, cols := p.components.Dims()

	reconstructed := MatMul(mat.DenseCopyOf(m), p.components)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			reconstructed.Set(i, j, reconstructed.At(i, j)+p.mean[j])
		}
	}

	return reconstructed
}

// get the principal components
func (p *pca) GetComponents() *mat.Dense {
	return p.components
}

// get the variances of the principal component scores
func (p *pca) GetExplainedVariance() []float64 {
	return p.varianceRatio
}
