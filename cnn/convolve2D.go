package cnn

import (
	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

// convolution operation between a matrix and a filter
func Convolve2D(x, filter *mat.Dense, stride int) *mat.Dense {
	xRows, xCols := x.Dims()
	dARows, dACols := filter.Dims()
	hOut := (xRows-dARows)/stride + 1
	wOut := (xCols-dACols)/stride + 1

	// convert image to column-based vector
	im2col := Im2Col(x, dARows, dACols, stride)
	flattenedFilter := Flatten(filter)

	out := ngo.MatMul(flattenedFilter, im2col)
	return Reshape(out, hOut, wOut)
}

// convert image to column-based vector
func Im2Col(x *mat.Dense, filterRows, filterCols, stride int) *mat.Dense {
	h, w := x.Dims()
	hOut := (h-filterRows)/stride + 1
	wOut := (w-filterCols)/stride + 1
	cols := mat.NewDense(filterRows*filterCols, hOut*wOut, nil)

	colIdx := 0
	xValue := x.RawMatrix()
	colsRaw := cols.RawMatrix()
	for i := 0; i <= h-filterRows; i += stride {
		for j := 0; j <= w-filterCols; j += stride {
			for k := 0; k < filterRows; k++ {
				for l := 0; l < filterCols; l++ {
					xIndex := (i+k)*xValue.Cols + (j + l)
					colsRaw.Data[(k*filterCols+l)*colsRaw.Stride+colIdx] = xValue.Data[xIndex]
				}
			}
			colIdx++
		}
	}
	return cols
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

