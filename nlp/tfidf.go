package nlp

import (
	"math"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type TFIDFVectorizer interface {
	Fit(train ...string)
	Transform(docs ...string) *mat.Dense
	FitTransform(texts ...string) *mat.Dense
	GetVocabulary() map[string]int
}

type tfidfVectorizer struct {
	vectorizer CountVectorizer
	transform  *mat.Dense
}

// measures the importance of words in a document compared to the entire dataset
func NewTFIDFVectorizer(numWords int, stopWords ...string) TFIDFVectorizer {
	return &tfidfVectorizer{
		vectorizer: NewCountVectorizer(numWords, stopWords...),
	}
}

// counts term occurrences across all documents and constructs
// an inverse document frequency transform
func (t *tfidfVectorizer) Fit(texts ...string) {
	countMatrix := t.vectorizer.FitTransform(texts...)

	// calculating the number of documents that contain each term
	rows, cols := countMatrix.Dims()
	docsContainingTerm := make([]int, cols)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			if countMatrix.At(i, j) > 0 {
				docsContainingTerm[j]++
			}
		}
	}

	// calculating the IDF for each term
	idf := make([]float64, cols)
	for j := 0; j < cols; j++ {
		idf[j] = math.Log(float64(1+rows)/(1+float64(docsContainingTerm[j]))) + 1
	}

	// build a column vector from idf array
	t.transform = mat.NewDense(cols, 1, idf)
}

// applies the inverse document frequency (IDF) transform by multiplying
// each term frequency by its corresponding IDF value
func (t *tfidfVectorizer) Transform(texts ...string) *mat.Dense {
	// simply multiply the matrix by our idf transform (the column vector of term weights)
	countMatrix := t.vectorizer.Transform(texts...)
	tfidfMatrix := ngo.MulMatrixVector(countMatrix, t.transform)

	// L2 norm matrix to remove any bias caused by documents of different
	// lengths where longer documents naturally have more words and so higher word counts
	for i := 0; i < tfidfMatrix.RawMatrix().Rows; i++ {
		row := tfidfMatrix.RawRowView(i)
		norm := floats.Norm(row, 2)
		for j := range row {
			row[j] /= norm
		}
	}

	return tfidfMatrix
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform()
func (v *tfidfVectorizer) FitTransform(texts ...string) *mat.Dense {
	v.Fit(texts...)
	return v.Transform(texts...)
}

// return vocabulary map is populated by the Fit() or FitTransform() methods
func (v *tfidfVectorizer) GetVocabulary() map[string]int {
	return v.vectorizer.GetVocabulary()
}
