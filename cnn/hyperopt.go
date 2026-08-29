package cnn

import (
	"fmt"
	"log"
	"strconv"

	"github.com/adynascimento/deep-learning/hyperopt"
	"github.com/c-bata/goptuna"
)

type Params struct {
	NConvLayers      int
	HiddenLayers     []int
	LearningRate     float64
	L2Regularization float64
}

type IntRange struct {
	Min int
	Max int
}

type FloatRange struct {
	Min float64
	Max float64
}

type SearchSpace struct {
	NConvLayersRange   IntRange
	NHiddenLayersRange IntRange
	NHiddenRange       IntRange
	LearningRateRange  FloatRange
	L2Range            FloatRange
	NTrials            int
}

type Model func(int, Params) float64

type Hyperopt interface {
	Optimize(sampler hyperopt.Sampler, direction hyperopt.StudyDirection, model Model)
	GetBestParams() Params
}

type optimization struct {
	SearchSpace
	BestParams Params
}

func NewHyperopt(space SearchSpace) Hyperopt {
	return &optimization{
		SearchSpace: space,
	}
}

func (opt *optimization) Optimize(sampler hyperopt.Sampler, direction hyperopt.StudyDirection, model Model) {
	trial, err := hyperopt.Optimize(sampler, direction, opt.objective(model), opt.NTrials)
	if err != nil {
		log.Fatalln("error to optimize objective function:", err.Error())
	}

	nHiddenLayers := trial.Params["n_hidden_layers"].(int)

	hiddenLayers := make([]int, nHiddenLayers)
	for i := 0; i < nHiddenLayers; i++ {
		hiddenLayers[i] = trial.Params["hidden_layer_"+strconv.Itoa(i)].(int)
	}

	opt.BestParams.NConvLayers = trial.Params["n_conv_layers"].(int)
	opt.BestParams.HiddenLayers = hiddenLayers
	opt.BestParams.LearningRate = trial.Params["learning_rate"].(float64)
	opt.BestParams.L2Regularization = trial.Params["l2_regularization"].(float64)

	fmt.Printf("best trialID=%d with evaluation=%f \n", trial.ID, trial.Value)
}

func (opt *optimization) objective(model Model) hyperopt.Objective {
	return func(trial goptuna.Trial) (float64, error) {
		// define the search space via Suggest APIs
		nConvLayers, err := trial.SuggestInt(
			"n_conv_layers", opt.NConvLayersRange.Min, opt.NConvLayersRange.Max)
		if err != nil {
			return 0, err
		}

		nHiddenLayers, err := trial.SuggestInt(
			"n_hidden_layers", opt.NHiddenLayersRange.Min, opt.NHiddenLayersRange.Max)
		if err != nil {
			return 0, err
		}

		hiddenLayers := make([]int, nHiddenLayers)
		for j := 0; j < nHiddenLayers; j++ {
			hiddenLayers[j], err = trial.SuggestInt(
				"hidden_layer_"+strconv.Itoa(j), opt.NHiddenRange.Min, opt.NHiddenRange.Max)
			if err != nil {
				return 0, err
			}
		}

		learningRate, err := trial.SuggestLogFloat(
			"learning_rate", opt.LearningRateRange.Min, opt.LearningRateRange.Max)
		if err != nil {
			return 0, err
		}

		l2Regularization, err := trial.SuggestLogFloat(
			"l2_regularization", opt.L2Range.Min, opt.L2Range.Max)
		if err != nil {
			return 0, err
		}

		return model(
			trial.ID,
			Params{
				NConvLayers:      nConvLayers,
				HiddenLayers:     hiddenLayers,
				LearningRate:     learningRate,
				L2Regularization: l2Regularization,
			},
		), nil
	}
}

func (opt *optimization) GetBestParams() Params {
	return opt.BestParams
}
