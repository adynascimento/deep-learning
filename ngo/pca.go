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
	GetComponents() *mat.Dense
	GetExplainedVariance() []float64
	InverseTransform(m mat.Matrix) *mat.Dense
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
// which is represented as an r × c matrix a where each
// row is an observation and each column is a variable.
func (p *pca) Fit(m mat.Matrix) {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the mean
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
	p.components = mat.DenseCopyOf(dst.T())
	p.components = mat.DenseCopyOf(p.components.Slice(0, p.nComponents, 0, cols))

	// calculate variance ratio
	singularValues := svd.Values(nil)[:p.nComponents]
	floats.MulTo(singularValues, singularValues, singularValues)
	floats.ScaleTo(singularValues, 1.0/float64(rows-1), singularValues)

	totalVariance := floats.Sum(singularValues)
	p.varianceRatio = make([]float64, len(singularValues))
	for i, singularValue := range singularValues {
		p.varianceRatio[i] = singularValue / totalVariance
	}
}

// projects the data into principal component space
// the returned matrix will be of reduced dimensionality
func (p *pca) Transform(m mat.Matrix) *mat.Dense {
	data := mat.DenseCopyOf(m)
	rows, cols := data.Dims()

	// center the data by subtracting the mean
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			data.Set(i, j, data.At(i, j)-p.mean[j])
		}
	}

	return MatMul(data, p.components.T())
}

// transforms data back to the original space
func (p *pca) InverseTransform(m mat.Matrix) *mat.Dense {
	_, cols := p.components.Dims()

	reconstructed := MatMul(mat.DenseCopyOf(m), p.components)
	for j := 0; j < cols; j++ {
		for i := 0; i < reconstructed.RawMatrix().Rows; i++ {
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
