package hyperopt

import (
	"fmt"
	"sort"
	"strconv"
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

	params, _ := study.GetBestParams()
	printParams(hp.InputDim, hp.OutputDim, params)
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

	// define the search space via Suggest APIs
	l2Regularization, _ := trial.SuggestLogUniform("lambd", hp.LambdRange[0], hp.LambdRange[1])

	// neural network model
	metric := hp.NeuralNetworkModel(trial.ID, nnStructure, l2Regularization)

	return metric, nil
}

// print the best evaluation parameters.
func printParams(inputDim, outputDim int, params map[string]interface{}) {
	keys := []string{}
	for k := range params {
		if k != "lambd" && k != "nlayers" {
			keys = append(keys, k)
		}
	}

	sort.Strings(keys)
	nnStructure := make([]interface{}, len(keys)+2)
	nnStructure[0] = inputDim
	nnStructure[len(nnStructure)-1] = outputDim
	for k, v := range keys {
		nnStructure[k+1] = params[v]
	}

	fmt.Println("params:")
	fmt.Println("architecture:", nnStructure)
	fmt.Printf("lambd: %e \n", params["lambd"])
}
