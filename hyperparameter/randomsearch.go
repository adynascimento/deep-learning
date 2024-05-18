package hyperparameter

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/floats"

	ngo "github.com/adynascimento/deep-learning/numeric"
)

func (hp *hyperparameter) RandomSearchOptimization(model NeuralNetworkModel) {
	hp.NeuralNetworkModel = model

	// lists
	nnStructureList := [][]int{}
	l2RegularizationList := []float64{}
	errorList := []float64{}

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
		err := hp.NeuralNetworkModel(i, nnStructure, l2Regularization)
		errorList = append(errorList, err)

		fmt.Printf("%s \033[1m\033[38;5;27m[INFO]\033[0m Trial finished: trialID=%d, state=%s, evaluation=%f \n",
			time.Now().Format("2006-01-02 15:04:05"), i, "Complete", err)
	}

	min_index := floats.MinIdx(errorList)
	fmt.Println("number of finished trials:", len(errorList))
	fmt.Println("best trial:", errorList[min_index])
	fmt.Println("params:")
	fmt.Println("architecture:", nnStructureList[min_index])
	fmt.Printf("lambd: %e \n", l2RegularizationList[min_index])
}
