package cnn

import (
	"gonum.org/v1/gonum/cmplxs"
	"gonum.org/v1/gonum/dsp/fourier"
	"gonum.org/v1/gonum/mat"
)

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
