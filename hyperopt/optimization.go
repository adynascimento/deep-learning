package hyperopt

type StudyDirection string
type NeuralNetworkModel func(int, Params) float64

const (
	Maximize StudyDirection = "maximize" // maximizes objective function value
	Minimize StudyDirection = "minimize" // minimizes objective function value
)

type Hyperparameter interface {
	GetBestParams() map[string]interface{}
	RandomSearchOptimization(direction StudyDirection, model NeuralNetworkModel)
	BayesianOptimization(direction StudyDirection, model NeuralNetworkModel)
}

type Params struct {
	NNStructure      []int
	LearningRate     float64
	L2Regularization float64
}

type SearchSpace struct {
	InputDim          int
	OutputDim         int
	NLayersRange      []int
	NHiddenRange      []int
	LearningRateRange []float64
	LambdRange        []float64
	NModels           int
}

type hyperparameter struct {
	BestParams map[string]interface{}
	SearchSpace
	NeuralNetworkModel
}

func NewHyperparameterOptimization(space SearchSpace) Hyperparameter {
	return &hyperparameter{
		BestParams:  make(map[string]interface{}),
		SearchSpace: space,
	}
}

func (hp *hyperparameter) GetBestParams() map[string]interface{} {
	return hp.BestParams
}
