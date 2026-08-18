package hyperopt

import (
	"time"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/tpe"
)

type Sampler int

const (
	RandomSearch Sampler = iota
	Bayesian
)

func NewSampler(sampler Sampler) goptuna.Sampler {
	switch sampler {
	case Bayesian:
		return tpe.NewSampler(
			tpe.SamplerOptionSeed(time.Now().UnixNano()),
		)

	case RandomSearch:
		return goptuna.NewRandomSampler(
			goptuna.RandomSamplerOptionSeed(time.Now().UnixNano()),
		)

	default:
		panic("unknown hyperparameter sampler")
	}
}
