package neuralNetwork

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func toNetwork(model model) neuralNetwork {
	parameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(model.NN_structure) - 1          // number of layers

	for l := 0; l < L; l++ {
		parameters["W"+strconv.Itoa(l+1)] = mat.NewDense(model.NN_structure[l+1], model.NN_structure[l], model.Parameters["W"+strconv.Itoa(l+1)])
		parameters["b"+strconv.Itoa(l+1)] = mat.NewDense(model.NN_structure[l+1], 1, model.Parameters["b"+strconv.Itoa(l+1)])
	}

	activation_function := activation{}
	switch model.Activation_name {
	case ActivationTanh:
		activation_function = activation{name: model.Activation_name, function: tanhActivation, derivative: tanhPrimeActivation}
	case ActivationSigmoid:
		activation_function = activation{name: model.Activation_name, function: sigmoidActivation, derivative: sigmoidPrimeActivation}
	case ActivationElu:
		activation_function = activation{name: model.Activation_name, function: eluActivation, derivative: eluPrimeActivation}
	}

	network := neuralNetwork{
		activation: activation_function,
		parameters: parameters,
	}

	return network
}

func Load(path string) neuralNetwork {
	b, err := ioutil.ReadFile(path)
	if nil != err {
		log.Println("impossible to load neural network model from file: ", err.Error())
	}

	model := model{}
	err = json.Unmarshal(b, &model)
	if nil != err {
		log.Println("impossible to load neural network model from file: ", err.Error())
	}

	network := toNetwork(model)
	return network
}
