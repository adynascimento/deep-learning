package cnn

import (
	"encoding/json"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type model struct {
	ActivationFunction activationType
	Mode               modeType
	Optimizer          optimizerType
	LearningRate       float64
	L2Regularization   float64
	NIterations        int
	BatchSize          int
	ConvLayers         []saveConvLayer
	PoolLayers         []*poolLayer
	DenseLayer         saveDenseLayer
}
type saveConvLayer struct {
	InputShape      [3]int
	OutputShape     [3]int
	TrainableParams int
	NFilters        int
	FilterSize      int
	Stride          int
	Parameters      convParameters
}

type convParameters struct {
	W [][][]float64
	B []float64
}

type saveDenseLayer struct {
	NNStructure []int
	Parameters  map[string][]float64
}

func (cm *cnnModel) Save(path string) {
	model := toModel(*cm)

	b, err := json.MarshalIndent(model, "", "\t")
	if err != nil {
		log.Println("error to save neural network model on file:", err.Error())
	}

	err = os.WriteFile(path, b, 0644)
	if err != nil {
		log.Println("error to save neural network model on file:", err.Error())
	}
}

func toModel(cm cnnModel) model {
	// conv layers parameters
	convLayers := []saveConvLayer{}
	for _, v := range cm.ConvLayers {
		var filters [][][]float64
		for _, f := range v.Parameters.W {
			var channels [][]float64
			for _, c := range f {
				channels = append(channels, c.RawMatrix().Data)
			}
			filters = append(filters, channels)
		}

		convLayers = append(convLayers, saveConvLayer{
			InputShape:      v.InputShape,
			OutputShape:     v.OutputShape,
			TrainableParams: v.TrainableParams,
			NFilters:        v.NFilters,
			FilterSize:      v.FilterSize,
			Stride:          v.Stride,
			Parameters: convParameters{
				W: filters,
				B: v.Parameters.B.RawMatrix().Data,
			},
		})
	}

	// dense layer parameters
	denseParameters := make(map[string][]float64)
	for k, v := range cm.DenseLayer.Parameters {
		denseParameters[k] = v.RawMatrix().Data
	}

	return model{
		ActivationFunction: cm.Activation.Name,
		Mode:               cm.Mode,
		Optimizer:          cm.Optimizer,
		LearningRate:       cm.LearningRate,
		L2Regularization:   cm.L2Regularization,
		NIterations:        cm.NIterations,
		BatchSize:          cm.BatchSize,
		ConvLayers:         convLayers,
		PoolLayers:         cm.PoolLayers,
		DenseLayer: saveDenseLayer{
			NNStructure: cm.DenseLayer.NNStructure,
			Parameters:  denseParameters,
		},
	}
}

func Load(path string) CNNModel {
	b, err := os.ReadFile(path)
	if nil != err {
		log.Println("error loading neural network model from file: ", err.Error())
	}

	model := model{}
	err = json.Unmarshal(b, &model)
	if nil != err {
		log.Println("error loading neural network model from file: ", err.Error())
	}

	return toNetwork(model)
}

func toNetwork(model model) CNNModel {
	// choice of activation function
	activationFunction := activationSettings[model.ActivationFunction]

	// choice of output layer activation function and loss function
	lossFunction := modeSettings[model.Mode].lossFunction
	outputActivationFunction := modeSettings[model.Mode].outputActivation

	// load conv layers parameters
	convLayers := []*convLayer{}
	for _, v := range model.ConvLayers {
		w := [][]*mat.Dense{}
		for _, f := range v.Parameters.W {
			channels := []*mat.Dense{}
			for _, c := range f {
				channels = append(channels, mat.NewDense(v.FilterSize, v.FilterSize, c))
			}
			w = append(w, channels)
		}

		convLayers = append(convLayers, &convLayer{
			InputShape:      v.InputShape,
			OutputShape:     v.OutputShape,
			TrainableParams: v.TrainableParams,
			Parameters: parameters{
				W: w,
				B: mat.NewDense(v.NFilters, 1, v.Parameters.B),
			},
			Activation: activationFunction,
			Stride:     v.Stride,
		})
	}

	// load dense layer parameters
	denseParameters := make(map[string]*mat.Dense) // map containing the parameters
	L := len(model.DenseLayer.NNStructure) - 1     // number of layers
	for l := 0; l < L; l++ {
		denseParameters["W"+strconv.Itoa(l+1)] = mat.NewDense(model.DenseLayer.NNStructure[l+1], model.DenseLayer.NNStructure[l],
			model.DenseLayer.Parameters["W"+strconv.Itoa(l+1)])
		denseParameters["b"+strconv.Itoa(l+1)] = mat.NewDense(model.DenseLayer.NNStructure[l+1], 1,
			model.DenseLayer.Parameters["b"+strconv.Itoa(l+1)])
	}

	return &cnnModel{
		cnn: &cnn{
			Activation:       activationFunction,
			Mode:             model.Mode,
			OutputActivation: outputActivationFunction,
			LossFunction:     lossFunction,
			ConvLayers:       convLayers,
			PoolLayers:       model.PoolLayers,
			DenseLayer: &denseLayer{
				NNStructure:      model.DenseLayer.NNStructure,
				Parameters:       denseParameters,
				Activation:       activationFunction,
				OutputActivation: outputActivationFunction,
			},
		},
	}
}
