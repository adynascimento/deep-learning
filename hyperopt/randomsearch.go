package hyperopt

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/floats"

	ngo "github.com/adynascimento/deep-learning/numeric"
)

func (hp *hyperparameter) RandomSearchOptimization(direction StudyDirection, model NeuralNetworkModel) {
	hp.NeuralNetworkModel = model

	// lists
	metricList := []float64{}
	nnStructureList := [][]int{}
	l2RegularizationList := []float64{}

	// random search optimization
	for i := 0; i < hp.NModels; i++ {
		// define the search space
		layersDims := ngo.RandInt(hp.NLayersRange[0], hp.NLayersRange[1])

		nnStructure := make([]int, layersDims)   // dnn architecture
		nnStructure[0] = hp.InputDim             // input layer size
		nnStructure[layersDims-1] = hp.OutputDim // output layer size
		for j := 1; j < layersDims-1; j++ {
			nnStructure[j] = ngo.RandInt(hp.NHiddenRange[0], hp.NHiddenRange[1]) // hidden layers
		}
		nnStructureList = append(nnStructureList, nnStructure)

		// define the search space
		l2Regularization := ngo.RandFloat(hp.LambdRange[0], hp.LambdRange[1]) // regularization parameter
		l2RegularizationList = append(l2RegularizationList, l2Regularization)

		// neural network model
		metric := hp.NeuralNetworkModel(i, nnStructure, l2Regularization)
		metricList = append(metricList, metric)

		fmt.Printf("%s \033[1m\033[38;5;27m[INFO]\033[0m Trial finished: trialID=%d, state=%s, evaluation=%f \n",
			time.Now().Format("2006-01-02 15:04:05"), i, "Complete", metric)
	}

	index := 0
	if direction == Minimize {
		index = floats.MinIdx(metricList)
	} else {
		index = floats.MaxIdx(metricList)
	}
	fmt.Printf("best trialID=%d with evaluation=%f \n", index, metricList[index])
	fmt.Println("params:")
	fmt.Println("architecture:", nnStructureList[index])
	fmt.Printf("lambd: %e \n", l2RegularizationList[index])
}
