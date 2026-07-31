package mlp

import (
	"encoding/json"
	"log"
	"os"
	"regexp"

	"github.com/adynascimento/deep-learning/nncore"
)

var (
	arrayRegex        = regexp.MustCompile(`\[\s*([\d,\s]+?)\s*\]`)
	spaceRegex        = regexp.MustCompile(`\s+`)
	commaSpaceRegex   = regexp.MustCompile(`,\s*`)
	openBracketRegex  = regexp.MustCompile(`\[\s+`)
	closeBracketRegex = regexp.MustCompile(`\s+\]`)
)

type model struct {
	NNStructure  []int                 `json:"nn_structure"`
	Activation   nncore.ActivationType `json:"activation"`
	Mode         nncore.ModeType       `json:"mode"`
	LearningRate float64               `json:"learning_rate"`
	Epochs       int                   `json:"epochs"`
	BatchSize    int                   `json:"batch_size"`
	Dense        denseModel            `json:"dense"`
}

type denseModel struct {
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

func (nm *neuralModel) Save(path string) {
	model := toModel(*nm)

	b, err := json.MarshalIndent(model, "", "\t")
	if err != nil {
		log.Fatalln("error to save neural network model on file:", err.Error())
	}

	jsonStr := arrayRegex.ReplaceAllStringFunc(string(b), func(match string) string {
		internal := commaSpaceRegex.ReplaceAllString(spaceRegex.ReplaceAllString(match, " "), ", ")
		return openBracketRegex.ReplaceAllString(closeBracketRegex.ReplaceAllString(internal, "]"), "[")
	})

	if err := os.WriteFile(path, []byte(jsonStr), 0644); err != nil {
		log.Fatalln("error to save neural network model on file:", err.Error())
	}
}

func toModel(n neuralModel) model {
	denseParameters, err := n.Dense.MarshalParameters()
	if err != nil {
		log.Fatalln("error to marshal dense parameters:", err.Error())
	}

	optimizerState, err := n.Dense.Optimizer.MarshalState()
	if err != nil {
		log.Fatalln("error to marshal optimizer state:", err.Error())
	}

	return model{
		NNStructure:  n.NNStructure,
		Activation:   n.Dense.Activation.Name,
		Mode:         n.Dense.OutputActivation.Mode,
		LearningRate: n.LearningRate,
		Epochs:       n.Epochs,
		BatchSize:    n.BatchSize,
		Dense: denseModel{
			Parameters:       denseParameters,
			L2Regularization: n.Dense.L2Regularization,
			Dropout:          n.Dense.Dropout,
			Optimizer: optimizer{
				Name:  n.Dense.Optimizer.Name(),
				Iter:  n.Dense.Iter,
				State: optimizerState,
			},
		},
	}
}

func Load(path string) NeuralModel {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatalln("error loading neural network model from file: ", err.Error())
	}

	model := model{}
	if err := json.Unmarshal(b, &model); err != nil {
		log.Fatalln("error loading neural network model from file: ", err.Error())
	}

	return toNetwork(model)
}

func toNetwork(m model) NeuralModel {
	// choice of activation function
	activation := nncore.NewActivation(m.Activation)

	// choice of output layer activation function and loss function
	configMode := nncore.NewMode(m.Mode)

	// choice of optimization algorithm
	optimizer := nncore.NewOptimizer(m.Dense.Optimizer.Name)
	if err := optimizer.UnmarshalState(m.Dense.Optimizer.State); err != nil {
		log.Fatalln("error unmarshal optimizer state: ", err.Error())
	}

	// initializing denseLayer layer
	denseLayer := &nncore.Dense{
		Activation:       activation,
		OutputActivation: configMode.OutputActivation,
		Optimizer:        optimizer,
		Iter:             m.Dense.Optimizer.Iter,
		L2Regularization: m.Dense.L2Regularization,
		Dropout:          m.Dense.Dropout,
	}
	if err := denseLayer.UnmarshalParameters(m.Dense.Parameters); err != nil {
		log.Fatalln("error unmarshal dense parameters: ", err.Error())
	}

	return &neuralModel{
		neuralNetwork: &neuralNetwork{
			NNStructure:  m.NNStructure,
			Mode:         m.Mode,
			LossFunction: configMode.LossFunction,
			Dense:        denseLayer,
		},
		LearningRate: m.LearningRate,
		Epochs:       m.Epochs,
		BatchSize:    m.BatchSize,
	}
}
