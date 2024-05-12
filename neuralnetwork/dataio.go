package neuralnetwork

import (
	"encoding/json"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type model struct {
	NNStructure      []int                `json:"nn_structure"`
	ActivationName   activationType       `json:"activation"`
	Mode             modeType             `json:"mode"`
	OptimizerName    optimizerType        `json:"optimizer"`
	LearningRate     float64              `json:"learning_rate"`
	L2Regularization float64              `json:"l2_regularization"`
	NIterations      int                  `json:"n_iterations"`
	Parameters       map[string][]float64 `json:"parameters"`
}

// save a representation of v to the file at path.
func (n *NeuralModel) Save(path string) {
	model := toModel(*n)

	b, err := json.MarshalIndent(model, "", "\t")
	if err != nil {
		log.Println("impossible to save neural network model on file:", err.Error())
	}

	err = os.WriteFile(path, b, 0644)
	if err != nil {
		log.Println("impossible to save neural network model on file:", err.Error())
	}
}

func toModel(network NeuralModel) model {
	parameters := make(map[string][]float64)
	for k, v := range network.Parameters {
		parameters[k] = v.RawMatrix().Data
	}

	model := model{
		NNStructure:      network.NNStructure,
		ActivationName:   network.Activation.Name,
		Mode:             network.OutputActivation.Mode,
		OptimizerName:    network.Optimizer.Name,
		LearningRate:     network.LearningRate,
		L2Regularization: network.L2Regularization,
		NIterations:      network.NIterations,
		Parameters:       parameters,
	}

	return model
}

func Load(path string) NeuralModel {
	b, err := os.ReadFile(path)
	if nil != err {
		log.Println("error loading neural network model from file: ", err.Error())
	}

	model := model{}
	err = json.Unmarshal(b, &model)
	if nil != err {
		log.Println("error loading neural network model from file: ", err.Error())
	}

	network := toNetwork(model)
	return network
}

func toNetwork(model model) NeuralModel {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(model.NNStructure) - 1           // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = mat.NewDense(model.NNStructure[l+1], model.NNStructure[l], model.Parameters["W"+strconv.Itoa(l+1)])
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(model.NNStructure[l+1], 1, model.Parameters["b"+strconv.Itoa(l+1)])
	}

	activationFunction := activation{}
	switch model.ActivationName {
	case ActivationTanh:
		activationFunction = activation{
			Name:       model.ActivationName,
			Function:   tanhActivation,
			Derivative: tanhActivationDerivative,
		}
	case ActivationSigmoid:
		activationFunction = activation{
			Name:       model.ActivationName,
			Function:   sigmoidActivation,
			Derivative: sigmoidActivationDerivative,
		}
	case ActivationElu:
		activationFunction = activation{
			Name:       model.ActivationName,
			Function:   eluActivation,
			Derivative: eluActivationDerivative,
		}
	}

	outputActivationFunction := outputActivation{}
	switch model.Mode {
	case ModeRegression:
		outputActivationFunction = outputActivation{
			Mode:     model.Mode,
			Function: applyLinear,
		}
	case ModeMultiClass:
		outputActivationFunction = outputActivation{
			Mode:     model.Mode,
			Function: applySoftmax,
		}
	case ModeMultiLabel:
		outputActivationFunction = outputActivation{
			Mode:     model.Mode,
			Function: applySigmoid,
		}
	case ModeBinary:
		outputActivationFunction = outputActivation{
			Mode:     model.Mode,
			Function: applySigmoid,
		}
	}

	network := NeuralModel{
		NeuralNetwork: NeuralNetwork{
			Activation:       activationFunction,
			OutputActivation: outputActivationFunction,
			Parameters:       parameters,
		},
	}

	return network
}
