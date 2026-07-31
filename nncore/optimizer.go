package nncore

import (
	"bytes"
	"encoding/gob"
	"math"

	"github.com/adynascimento/deep-learning/ngo"
	"gonum.org/v1/gonum/mat"
)

type OptimizerType string

const (
	AdamOptimizer            OptimizerType = "adam"
	GradientDescentOptimizer OptimizerType = "gradientdescent"
)

type Optimizer interface {
	Name() OptimizerType
	Step(parameters []*Parameter, learningRate, t float64)

	MarshalState() ([]byte, error)
	UnmarshalState([]byte) error
}

type Parameter struct {
	Value    *mat.Dense
	Gradient *mat.Dense
	Update   func(*mat.Dense) // callback function to update the source parameter
}

func NewOptimizer(optType OptimizerType) Optimizer {
	switch optType {
	case GradientDescentOptimizer:
		return &gradientDescentOptimizer{}

	case AdamOptimizer:
		return &adamOptimizer{}

	default:
		panic("optimizer not implemented")
	}
}

type gradientDescentOptimizer struct{}

func (opt *gradientDescentOptimizer) Name() OptimizerType {
	return GradientDescentOptimizer
}

// update the parameters (gradient descent)
func (opt *gradientDescentOptimizer) Step(parameters []*Parameter, learningRate, t float64) {
	for _, p := range parameters {
		p.Value = ngo.Sub(p.Value, ngo.Scale(learningRate, p.Gradient))
		p.Update(p.Value)
	}
}

func (opt *gradientDescentOptimizer) MarshalState() ([]byte, error) {
	return nil, nil
}

func (opt *gradientDescentOptimizer) UnmarshalState(data []byte) error {
	return nil
}

type adamOptimizer struct {
	adamState
}

type adamState struct {
	V []*mat.Dense
	S []*mat.Dense
}

func (opt *adamOptimizer) Name() OptimizerType {
	return AdamOptimizer
}

// update the parameters (adam optimizer)
func (opt *adamOptimizer) Step(parameters []*Parameter, learningRate, t float64) {
	// initializing adam states
	if len(opt.V) == 0 {
		opt.V = make([]*mat.Dense, len(parameters))
		opt.S = make([]*mat.Dense, len(parameters))

		for i, p := range parameters {
			r, c := p.Value.Dims()
			opt.V[i] = mat.NewDense(r, c, nil)
			opt.S[i] = mat.NewDense(r, c, nil)
		}
	}

	// default parameters
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-08

	for i, p := range parameters {
		// moving average of the gradients
		opt.V[i] = ngo.Add(ngo.Scale(beta1, opt.V[i]), ngo.Scale((1-beta1), p.Gradient))

		// moving average of the squared gradients
		opt.S[i] = ngo.Add(ngo.Scale(beta2, opt.S[i]), ngo.Scale((1.0-beta2), ngo.Square(p.Gradient)))

		// compute bias-corrected first moment estimate
		vCorr := ngo.Scale(1.0/(1.0-math.Pow(beta1, t)), opt.V[i])

		// compute bias-corrected second raw moment estimate
		sCorr := ngo.Scale(1.0/(1.0-math.Pow(beta2, t)), opt.S[i])

		// update parameter
		sqrtV := ngo.Apply(func(_, _ int, x float64) float64 { return math.Sqrt(x) + epsilon }, sCorr)
		p.Value = ngo.Sub(p.Value, ngo.Scale(learningRate, ngo.DivElem(vCorr, sqrtV)))

		// update source parameter
		p.Update(p.Value)
	}
}

func (opt *adamOptimizer) MarshalState() ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(opt.adamState); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (opt *adamOptimizer) UnmarshalState(data []byte) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(&opt.adamState)
}
