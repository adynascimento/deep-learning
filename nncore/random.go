package nncore

import (
	"math/rand/v2"
	"time"

	"gonum.org/v1/gonum/mat"
)

type RNG struct {
	*rand.Rand
}

func NewRand(seed *uint64) *RNG {
	if seed != nil {
		return &RNG{
			Rand: rand.New(rand.NewPCG(*seed, *seed)),
		}
	}

	return &RNG{
		Rand: rand.New(rand.NewPCG(
			uint64(time.Now().UnixNano()),
			uint64(time.Now().UnixNano()),
		)),
	}
}

// create a new rand with a seed derived from the current state
func (r *RNG) Spawn() *RNG {
	seed := r.Uint64()
	return NewRand(&seed)
}

// generate a random slice of float64
func (r *RNG) Randn(n, m int) *mat.Dense {
	random := make([]float64, n*m)
	for i := range random {
		random[i] = r.NormFloat64()
	}

	return mat.NewDense(n, m, random)
}
