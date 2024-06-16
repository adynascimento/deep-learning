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
// row is a variable and each column is an observation.
// matrix shape (nFeatures, nSamples)
func (p *pca) Fit(m mat.Matrix) {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the mean
	p.mean = make([]float64, rows)
	for i := 0; i < rows; i++ {
		mean := stat.Mean(mat.Row(nil, i, data), nil)
		p.mean[i] = mean
		for j := 0; j < cols; j++ {
			data.Set(i, j, data.At(i, j)-mean)
		}
	}

	// calculate the SVD decomposition
	svd := &mat.SVD{}
	if ok := svd.Factorize(data.T(), mat.SVDFull); !ok {
		log.Fatal("error in SVD decomposition")
	}

	// get the eigenvectors (principal components)
	var dst mat.Dense
	svd.VTo(&dst)
	p.components = mat.DenseCopyOf(dst.Slice(0, rows, 0, p.nComponents))

	// calculate variance ratio
	singularValues := svd.Values(nil)
	floats.MulTo(singularValues, singularValues, singularValues)
	floats.ScaleTo(singularValues, 1.0/float64(cols-1), singularValues)

	totalVariance := floats.Sum(singularValues)
	p.varianceRatio = make([]float64, len(singularValues[:p.nComponents]))
	for i, singularValue := range singularValues[:p.nComponents] {
		p.varianceRatio[i] = singularValue / totalVariance
	}
}

// projects the data into principal component space
// the returned matrix will be of reduced dimensionality
func (p *pca) Transform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the mean
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data.Set(i, j, data.At(i, j)-p.mean[i])
		}
	}

	return MatMul(p.components.T(), data)
}

// FitTransform is exactly equivalent to calling Fit()
// followed by Transform()
func (p *pca) FitTransform(m mat.Matrix) *mat.Dense {
	p.Fit(m)
	return p.Transform(m)
}

// transforms data back to the original space
func (p *pca) InverseTransform(m mat.Matrix) *mat.Dense {
	rows, _ := p.components.Dims()

	reconstructed := MatMul(p.components, mat.DenseCopyOf(m))
	for i := 0; i < rows; i++ {
		for j := 0; j < reconstructed.RawMatrix().Cols; j++ {
			reconstructed.Set(i, j, reconstructed.At(i, j)+p.mean[i])
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
