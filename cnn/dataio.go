package cnn

import (
	"encoding/json"
	"log"
	"os"
	"regexp"
	"runtime"

	"github.com/adynascimento/deep-learning/nncore"
	"gonum.org/v1/gonum/mat"
)

var (
	arrayRegex        = regexp.MustCompile(`\[\s*([\d,\s]+?)\s*\]`)
	spaceRegex        = regexp.MustCompile(`\s+`)
	commaSpaceRegex   = regexp.MustCompile(`,\s*`)
	openBracketRegex  = regexp.MustCompile(`\[\s+`)
	closeBracketRegex = regexp.MustCompile(`\s+\]`)
)

type model struct {
	Activation   nncore.ActivationType `json:"activation"`
	Mode         nncore.ModeType       `json:"mode"`
	LearningRate float64               `json:"learning_rate"`
	Epochs       int                   `json:"epochs"`
	BatchSize    int                   `json:"batch_size"`
	ConvLayers   []convModel           `json:"conv"`
	PoolLayers   []*poolLayer          `json:"pool"`
	DenseLayer   denseModel            `json:"dense"`
}

type convModel struct {
	InputShape       [3]int    `json:"input_shape"`
	OutputShape      [3]int    `json:"output_shape"`
	TrainableParams  int       `json:"trainable_params"`
	NFilters         int       `json:"n_filters"`
	FilterSize       int       `json:"filter_size"`
	Stride           int       `json:"stride"`
	L2Regularization float64   `json:"l2_regularization"`
	Parameters       []byte    `json:"parameters"`
	Optimizer        optimizer `json:"optimizer"`
}

type denseModel struct {
	NNStructure      []int     `json:"nn_structure"`
	Parameters       []byte    `json:"parameters"`
	L2Regularization float64   `json:"l2_regularization"`
	Dropout          float64   `json:"dropout"`
	Optimizer        optimizer `json:"optimizer"`
}

type optimizer struct {
	Name  nncore.OptimizerType `json:"name"`
	Iter  float64              `json:"iter"`
	State []byte               `json:"state"`
}

func (cm *cnnModel) Save(path string) {
	model := toModel(*cm)

	b, err := json.MarshalIndent(model, "", "\t")
	if err != nil {
		log.Fatalln("error to save neural network model on file:", err.Error())
	}

	jsonStr := arrayRegex.ReplaceAllStringFunc(string(b), func(match string) string {
		internal := commaSpaceRegex.ReplaceAllString(spaceRegex.ReplaceAllString(match, " "), ", ")
		return openBracketRegex.ReplaceAllString(closeBracketRegex.ReplaceAllString(internal, "]"), "[")
	})

	err = os.WriteFile(path, []byte(jsonStr), 0644)
	if err != nil {
		log.Fatalln("error to save neural network model on file:", err.Error())
	}
}

func toModel(cm cnnModel) model {
	// conv layers parameters
	conv := []convModel{}
	for _, v := range cm.ConvLayers {
		convParameters, err := v.MarshalParameters()
		if err != nil {
			log.Fatalln("error to marshal conv parameters:", err.Error())
		}

		convOptimizerState, err := v.Optimizer.MarshalState()
		if err != nil {
			log.Fatalln("error to marshal conv optimizer state:", err.Error())
		}

		conv = append(conv, convModel{
			InputShape:       v.InputShape,
			OutputShape:      v.OutputShape,
			TrainableParams:  v.TrainableParams,
			NFilters:         v.NFilters,
			FilterSize:       v.FilterSize,
			Stride:           v.Stride,
			L2Regularization: v.L2Regularization,
			Parameters:       convParameters,
			Optimizer: optimizer{
				Name:  v.Optimizer.Name(),
				Iter:  v.Iter,
				State: convOptimizerState,
			},
		})
	}

	// dense layer parameters
	denseParameters, err := cm.DenseLayer.MarshalParameters()
	if err != nil {
		log.Fatalln("error to marshal dense parameters:", err.Error())
	}

	denseOptimizerState, err := cm.DenseLayer.Optimizer.MarshalState()
	if err != nil {
		log.Fatalln("error to marshal dense optimizer state:", err.Error())
	}

	return model{
		Activation:   cm.Activation.Name,
		Mode:         cm.Mode,
		LearningRate: cm.LearningRate,
		Epochs:       cm.Epochs,
		BatchSize:    cm.BatchSize,
		ConvLayers:   conv,
		PoolLayers:   cm.PoolLayers,
		DenseLayer: denseModel{
			NNStructure:      cm.DenseLayerStructure,
			Parameters:       denseParameters,
			L2Regularization: cm.DenseLayer.L2Regularization,
			Dropout:          cm.DenseLayer.Dropout,
			Optimizer: optimizer{
				Name:  cm.DenseLayer.Optimizer.Name(),
				Iter:  cm.DenseLayer.Iter,
				State: denseOptimizerState,
			},
		},
	}
}

func Load(path string) CNNModel {
	b, err := os.ReadFile(path)
	if nil != err {
		log.Fatalln("error loading neural network model from file: ", err.Error())
	}

	model := model{}
	err = json.Unmarshal(b, &model)
	if nil != err {
		log.Fatalln("error loading neural network model from file: ", err.Error())
	}

	return toNetwork(model)
}

func toNetwork(m model) CNNModel {
	// choice of activation function
	activation := nncore.NewActivation(m.Activation)

	// choice of output layer activation function and loss function
	configMode := nncore.NewMode(m.Mode)

	// load conv layers parameters
	convLayers := []*convLayer{}
	for _, v := range m.ConvLayers {
		// choice of optimization algorithm
		optimizer := nncore.NewOptimizer(v.Optimizer.Name)
		if err := optimizer.UnmarshalState(v.Optimizer.State); err != nil {
			log.Fatalln("error unmarshal conv optimizer state: ", err.Error())
		}

		nChannels := v.InputShape[0]
		gradients := newGradients(v.NFilters, nChannels, v.FilterSize)
		convLayer := &convLayer{
			InputShape:       v.InputShape,
			OutputShape:      v.OutputShape,
			TrainableParams:  v.TrainableParams,
			Activation:       activation,
			Gradients:        gradients,
			Optimizer:        optimizer,
			NFilters:         v.NFilters,
			NChannels:        nChannels,
			FilterSize:       v.FilterSize,
			Stride:           v.Stride,
			Iter:             v.Optimizer.Iter,
			L2Regularization: v.L2Regularization,
		}
		if err := convLayer.UnmarshalParameters(v.Parameters); err != nil {
			log.Fatalln("error unmarshal conv parameters: ", err.Error())
		}

		convLayers = append(convLayers, convLayer)
	}

	// load dense layer parameters
	denseOptimizer := nncore.NewOptimizer(m.DenseLayer.Optimizer.Name)
	if err := denseOptimizer.UnmarshalState(m.DenseLayer.Optimizer.State); err != nil {
		log.Fatalln("error unmarshal dense optimizer state: ", err.Error())
	}

	// initializing denseLayer layer
	denseLayer := &nncore.Dense{
		Activation:       activation,
		OutputActivation: configMode.OutputActivation,
		Optimizer:        denseOptimizer,
		Iter:             m.DenseLayer.Optimizer.Iter,
		L2Regularization: m.DenseLayer.L2Regularization,
		Dropout:          m.DenseLayer.Dropout,
	}
	if err := denseLayer.UnmarshalParameters(m.DenseLayer.Parameters); err != nil {
		log.Fatalln("error unmarshal dense parameters: ", err.Error())
	}

	return &cnnModel{
		cnn: &cnn{
			Activation:          activation,
			Mode:                m.Mode,
			OutputActivation:    configMode.OutputActivation,
			LossFunction:        configMode.LossFunction,
			ConvLayers:          convLayers,
			PoolLayers:          m.PoolLayers,
			FlattenLayer:        newFlatten(),
			DenseLayer:          denseLayer,
			DenseLayerStructure: m.DenseLayer.NNStructure,
		},
		cnnForwardOutputs: &cnnForwardOutputs{
			ConvOutputs: make(map[string][][]*mat.Dense),
			PoolOutputs: make(map[string][]*poolCache),
		},
		NWorkers:        runtime.GOMAXPROCS(0),
		WorkerGradients: newWorkerGradients(convLayers, runtime.GOMAXPROCS(0)),
		LearningRate:    m.LearningRate,
		Epochs:          m.Epochs,
		BatchSize:       m.BatchSize,
	}
}
