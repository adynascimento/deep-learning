package ngo

import (
	"math/rand"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type directionType string

const (
	OverRows    directionType = "rows"
	OverColumns directionType = "columns"
)

// split dataset into two matrix (training and testing)
func Split(a *mat.Dense, frac float64) (*mat.Dense, *mat.Dense) {
	nRows, nCols := a.Dims()

	jdx := int(frac * float64(nCols))
	m1 := a.Slice(0, nRows, 0, jdx)
	m2 := a.Slice(0, nRows, jdx, nCols)

	return m1.(*mat.Dense), m2.(*mat.Dense)
}

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

// add "a" matrix with "b" column vector row-wise
func AddMatrixVector(a *mat.Dense, b *mat.Dense) *mat.Dense {
	m := mat.DenseCopyOf(a)
	for i := range m.RawMatrix().Rows {
		floats.AddConst(b.At(i, 0), m.RawRowView(i))
	}

	return m
}

// multiply "a" matrix with "b" column vector row-wise
func MulMatrixVector(a, b *mat.Dense) *mat.Dense {
	m := mat.DenseCopyOf(a)
	for i := range m.RawMatrix().Rows {
		floats.Scale(b.At(i, 0), m.RawRowView(i))
	}

	return m
}

// division "a" matrix with "b" row vector column-wise
func DivMatrixVector(a *mat.Dense, b *mat.Dense) *mat.Dense {
	m := mat.DenseCopyOf(a)
	for i := range m.RawMatrix().Rows {
		floats.Div(m.RawRowView(i), b.RawMatrix().Data)
	}

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

// rotate matrix (2d component) by 180 degrees
func Rotate180(a *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()

	data := a.RawMatrix().Data
	rotated := make([]float64, len(data))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			rotated[i*cols+j] = data[(rows-i-1)*cols+(cols-j-1)]
		}
	}

	return mat.NewDense(rows, cols, rotated)
}

// applies zero padding to source matrix
// n number of columns to add and fill with zeros
func ZeroPadding(a *mat.Dense, n int) *mat.Dense {
	rows, cols := a.Dims()
	newRows := rows + 2*n
	newCols := cols + 2*n

	data := a.RawMatrix().Data
	padded := mat.NewDense(newRows, newCols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			padded.Set(i+n, j+n, data[i*cols+j])
		}
	}

	return padded
}

// finds the indices of the maximum value in a matrix
func MaxIndices(a *mat.Dense) (int, int) {
	maxIndex := floats.MaxIdx(a.RawMatrix().Data)

	maxIdx := maxIndex / a.RawMatrix().Cols
	maxJdx := maxIndex % a.RawMatrix().Cols

	return maxIdx, maxJdx
}

// flatten convert matrix to vector
func Flatten(x *mat.Dense) *mat.Dense {
	height, width := x.Dims()
	flattenMatrix := make([]float64, height*width)

	xValue := x.RawMatrix()
	for row := 0; row < height; row++ {
		for column := 0; column < width; column++ {
			flattenMatrix[row*width+column] = xValue.Data[row*width+column]
		}
	}

	return mat.NewDense(1, height*width, flattenMatrix)
}

// reshape matrix to given rows(height) and cols(width)
func Reshape(x *mat.Dense, rows, cols int) *mat.Dense {
	xValue := x.RawMatrix()
	newMat := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		newMat[i] = xValue.Data[i]
	}

	return mat.NewDense(rows, cols, newMat)
}

// insert stride-1 zeros between rows and columns.
// it is used in the input-gradient convolution when the forward convolution used stride > 1.
func UpSampleZeros2D(x *mat.Dense, stride int) *mat.Dense {
	if stride <= 1 {
		return mat.DenseCopyOf(x)
	}

	rows, cols := x.Dims()
	dilatedRows := (rows-1)*stride + 1
	dilatedCols := (cols-1)*stride + 1

	out := mat.NewDense(dilatedRows, dilatedCols, nil)

	xRaw := x.RawMatrix()
	outRaw := out.RawMatrix()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			outRow := r * stride
			outCol := c * stride
			outRaw.Data[outRow*outRaw.Stride+outCol] = xRaw.Data[r*xRaw.Stride+c]
		}
	}

	return out
}

// applies the function fn to each of the elements of a. The function fn takes a row/column
// index and element value and returns some function of that tuple
func Apply(fn func(i, j int, v float64) float64, a mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Apply(fn, a)

	return m
}

// applies the function fn to the underlying data of a raw matrix
func ApplyRaw(a *mat.Dense, fn func([]float64)) *mat.Dense {
	m := mat.DenseCopyOf(a)
	fn(m.RawMatrix().Data)

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

// appends the rows of b onto the rows of a
func Stack(a, b mat.Matrix) *mat.Dense {
	m := new(mat.Dense)
	m.Stack(a, b)

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
