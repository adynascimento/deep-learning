package neuralNetwork

import (
	"encoding/json"
	"io/ioutil"
	"log"
)

type model struct {
	NN_structure      []int                `json:"nn_structure"`
	Activation_name   activationType       `json:"activation"`
	Optimizer_name    optimizerType        `json:"optimizer"`
	Learning_rate     float64              `json:"learning_rate"`
	L2_regularization float64              `json:"l2_regularization"`
	Num_iterations    int                  `json:"num_iterations"`
	Parameters        map[string][]float64 `json:"parameters"`
}

func toModel(network neuralNetwork) model {
	parameters := make(map[string][]float64)
	for k, v := range network.parameters {
		parameters[k] = v.RawMatrix().Data
	}

	model := model{
		NN_structure:      network.nn_structure,
		Activation_name:   network.activation.name,
		Optimizer_name:    network.optimizer.name,
		Learning_rate:     network.learning_rate,
		L2_regularization: network.l2_regularization,
		Num_iterations:    network.num_iterations,
		Parameters:        parameters,
	}

	return model
}

// save a representation of v to the file at path.
func (network *neuralNetwork) Save(path string) {
	model := toModel(*network)

	b, err := json.MarshalIndent(model, "", "\t")
	if err != nil {
		log.Println("impossible to save neural network model on file:", err.Error())
	}

	err = ioutil.WriteFile(path, b, 0644)
	if err != nil {
		log.Println("impossible to save neural network model on file:", err.Error())
	}
}
