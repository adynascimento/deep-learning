package nncore

import (
	"gonum.org/v1/gonum/mat"
)

// creates a binary mask for inverted dropout.
// each element has probability (1-p) of being kept.
func DropoutMask(a *mat.Dense, p float64, rng *RNG) []bool {
	mask := make([]bool, len(a.RawMatrix().Data))
	keepProb := 1.0 - p
	for i := range mask {
		mask[i] = rng.Float64() < keepProb
	}

	return mask
}

// inverted dropout: during training, surviving activations are scaled by 1/(1-p)
// so that no scaling is required during inference.
func ApplyDropoutMask(a *mat.Dense, mask []bool, p float64) {
	data := a.RawMatrix().Data

	scale := 1.0 / (1.0 - p)
	for i := range data {
		if mask[i] {
			data[i] *= scale
		} else {
			data[i] = 0
		}
	}
}
