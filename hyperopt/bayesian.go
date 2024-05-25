package hyperopt

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/tpe"
)

// bayesian optimization
func (hp *hyperparameter) BayesianOptimization(direction StudyDirection, model NeuralNetworkModel) {
	hp.NeuralNetworkModel = model

	// create a study which manages each experiment.
	study, _ := goptuna.CreateStudy(
		"neuralnetwork",
		goptuna.StudyOptionSampler(tpe.NewSampler(tpe.SamplerOptionSeed(time.Now().UnixNano()))),
		goptuna.StudyOptionDirection(goptuna.StudyDirection(direction)),
	)

	// evaluate objective function.
	study.Optimize(hp.objective, hp.NModels)

	// print the best evaluation parameters.
	trial, _ := study.Storage.GetBestTrial(study.ID)
	fmt.Printf("best trialID=%d with evaluation=%f \n", trial.ID, trial.Value)

	// best evaluation parameters
	keys := []string{}
	for k := range trial.Params {
		if strings.Contains(k, "hidden") {
			keys = append(keys, k)
		}
	}

	sort.Strings(keys)
	nnStructure := make([]interface{}, len(keys)+2)
	nnStructure[0] = hp.InputDim
	nnStructure[len(nnStructure)-1] = hp.OutputDim
	for k, v := range keys {
		nnStructure[k+1] = trial.Params[v]
	}

	hp.BestParams["NNStructure"] = nnStructure
	hp.BestParams["LearningRate"] = trial.Params["lr"]
	hp.BestParams["L2Regularization"] = trial.Params["lambd"]
}

// objective function which returns a value you want to minimize.
func (hp *hyperparameter) objective(trial goptuna.Trial) (float64, error) {
	// define the search space via Suggest APIs
	layersDims, _ := trial.SuggestInt("nlayers", hp.NLayersRange[0], hp.NLayersRange[1])

	nnStructure := make([]int, layersDims)   // dnn architecture
	nnStructure[0] = hp.InputDim             // input layer size
	nnStructure[layersDims-1] = hp.OutputDim // output layer size
	for j := 1; j < layersDims-1; j++ {
		nnStructure[j], _ = trial.SuggestInt("hidden"+strconv.Itoa(j), hp.NHiddenRange[0], hp.NHiddenRange[1]) // hidden layers
	}

	learningRate, _ := trial.SuggestLogUniform("lr", hp.LearningRateRange[0], hp.LearningRateRange[1])
	l2Regularization, _ := trial.SuggestLogUniform("lambd", hp.LambdRange[0], hp.LambdRange[1])

	// neural network model
	metric := hp.NeuralNetworkModel(trial.ID, Params{
		NNStructure:      nnStructure,
		LearningRate:     learningRate,
		L2Regularization: l2Regularization,
	})

	return metric, nil
}
