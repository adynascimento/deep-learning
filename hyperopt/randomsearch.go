package hyperopt

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/floats"

	"github.com/adynascimento/deep-learning/ngo"
)

func (hp *hyperparameter) RandomSearchOptimization(direction StudyDirection, model NeuralNetworkModel) {
	hp.NeuralNetworkModel = model

	// lists
	metricList := []float64{}
	nnStructureList := [][]int{}
	learningRateList := []float64{}
	l2RegularizationList := []float64{}

	// random search optimization
	for i := 0; i < hp.NModels; i++ {
		// define the search space
		layersDims := ngo.SuggestInt(hp.NLayersRange[0], hp.NLayersRange[1])

		nnStructure := make([]int, layersDims)   // dnn architecture
		nnStructure[0] = hp.InputDim             // input layer size
		nnStructure[layersDims-1] = hp.OutputDim // output layer size
		for j := 1; j < layersDims-1; j++ {
			nnStructure[j] = ngo.SuggestInt(hp.NHiddenRange[0], hp.NHiddenRange[1]) // hidden layers
		}

		learningRate := ngo.SuggestLogFloat(hp.LearningRateRange[0], hp.LearningRateRange[1])
		l2Regularization := ngo.SuggestLogFloat(hp.LambdRange[0], hp.LambdRange[1])

		// neural network model
		metric := hp.NeuralNetworkModel(i, Params{
			NNStructure:      nnStructure,
			LearningRate:     learningRate,
			L2Regularization: l2Regularization,
		})

		// trials
		metricList = append(metricList, metric)
		nnStructureList = append(nnStructureList, nnStructure)
		learningRateList = append(learningRateList, learningRate)
		l2RegularizationList = append(l2RegularizationList, l2Regularization)

		fmt.Printf("%s \033[1m\033[38;5;27m[INFO]\033[0m Trial finished: trialID=%d, state=%s, evaluation=%f \n",
			time.Now().Format("2006-01-02 15:04:05"), i, "Complete", metric)
	}

	idx := 0
	if direction == Minimize {
		idx = floats.MinIdx(metricList)
	} else if direction == Maximize {
		idx = floats.MaxIdx(metricList)
	}
	fmt.Printf("best trialID=%d with evaluation=%f \n", idx, metricList[idx])

	// best params
	hp.BestParams["NNStructure"] = nnStructureList[idx]
	hp.BestParams["LearningRate"] = learningRateList[idx]
	hp.BestParams["L2Regularization"] = l2RegularizationList[idx]
}
