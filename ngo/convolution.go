package ngo

import (
	"gonum.org/v1/gonum/cmplxs"
	"gonum.org/v1/gonum/dsp/fourier"
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

	out := MatMul(flattenedFilter, im2col)
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

// convert the column-based vector back to the original image matrix
func Col2Im(cols *mat.Dense, inputRows, inputCols, filterRows, filterCols, stride int) *mat.Dense {
	x := mat.NewDense(inputRows, inputCols, nil)

	colIdx := 0
	xValue := x.RawMatrix()
	colsRaw := cols.RawMatrix()
	for i := 0; i <= inputRows-filterRows; i += stride {
		for j := 0; j <= inputCols-filterCols; j += stride {
			for k := 0; k < filterRows; k++ {
				for l := 0; l < filterCols; l++ {
					xIndex := (i+k)*xValue.Cols + (j + l)
					xValue.Data[xIndex] += colsRaw.Data[(k*filterCols+l)*colsRaw.Stride+colIdx]
				}
			}
			colIdx++
		}
	}

	return x
}

// convolution operation between a matrix and a filter using FFT
func ConvolveFFT2D(x, filter *mat.Dense, stride int) *mat.Dense {
	// pad input and filter to optimal size for FFT
	xRows, xCols := x.Dims()
	filterRows, filterCols := filter.Dims()
	padRows := xRows + filterRows - 1
	padCols := xCols + filterCols - 1

	// padding with zeros to make FFT more efficient
	paddedX := x.Grow(padRows-xRows, padCols-xCols).(*mat.Dense)
	paddedFilter := filter.Grow(padRows-filterRows, padCols-filterCols).(*mat.Dense)

	// perform FFT on both padded input and filter
	fft := fourier.NewCmplxFFT(padRows * padCols)
	xFFT := applyFFT(fft, paddedX)
	filterFFT := applyFFT(fft, paddedFilter)

	// element-wise multiplication in frequency domain
	outFreq := make([]complex128, padRows*padCols)
	cmplxs.MulTo(outFreq, xFFT, filterFFT)

	// perform inverse FFT to get back to spatial domain
	outSpatial := applyIFFT(fft, outFreq, padRows, padCols)

	// extract valid region and apply stride
	hOut := (xRows-filterRows)/stride + 1
	wOut := (xCols-filterCols)/stride + 1
	output := extractOutputMatrix(outSpatial, hOut, wOut, stride)

	return output
}

// applies FFT to the input matrix
func applyFFT(fft *fourier.CmplxFFT, m *mat.Dense) []complex128 {
	rows, cols := m.Dims()

	complexData := make([]complex128, rows*cols)
	for i, v := range m.RawMatrix().Data {
		complexData[i] = complex(v, 0)
	}

	return fft.Coefficients(nil, complexData)
}

// applies inverse FFT with normalization to the input matrix
func applyIFFT(fft *fourier.CmplxFFT, m []complex128, rows, cols int) *mat.Dense {
	inverseData := fft.Sequence(nil, m)

	out := make([]float64, rows*cols)
	for i, v := range inverseData {
		out[i] = real(v) / float64(rows*cols)
	}

	return mat.NewDense(rows, cols, out)
}

// extracts the valid region from the output and applies stride
func extractOutputMatrix(m *mat.Dense, hOut, wOut, stride int) *mat.Dense {
	rows, cols := m.Dims()
	startRow := (rows - hOut) / 2
	startCol := (cols - wOut) / 2

	data := m.RawMatrix().Data
	out := mat.NewDense(hOut, wOut, nil)
	for i := 0; i < hOut; i++ {
		for j := 0; j < wOut; j++ {
			index := (startRow+i*stride)*cols + startCol + j*stride
			out.Set(i, j, data[index])
		}
	}

	return out
}
