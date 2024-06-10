package ngo

import (
	"math"
	"math/rand"
)

// generate uniform random int [min, max]
func SuggestInt(min, max int) int {
	return rand.Intn(max-min) + min
}

// generate uniform random float [min, max]
func SuggestFloat(min, max float64) float64 {
	return (rand.Float64() * (max - min)) + min
}

// generate log uniform random float [min, max]
func SuggestLogFloat(min, max float64) float64 {
	logMin := math.Log(min)
	logMax := math.Log(max)
	logUniform := rand.Float64()*(logMax-logMin) + logMin

	return math.Exp(logUniform)
}

// generate linearly spaced slice of float64
func Linspace(start, stop float64, num int) []float64 {
	var step float64
	if num == 1 {
		return []float64{start}
	}
	step = (stop - start) / float64(num-1)

	r := make([]float64, num)
	for i := 0; i < num; i++ {
		r[i] = start + float64(i)*step
	}

	return r
}
