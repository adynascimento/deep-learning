package hyperparameter

type StudyDirection string
type NeuralNetworkModel func(int, []int, float64) float64

const (
	Maximize StudyDirection = "maximize" // maximizes objective function value
	Minimize StudyDirection = "minimize" // minimizes objective function value
)

type Hyperparameter interface {
	RandomSearchOptimization(direction StudyDirection, model NeuralNetworkModel)
	BayesianOptimization(direction StudyDirection, model NeuralNetworkModel)
}

type Params struct {
	InputDim     int
	OutputDim    int
	NLayersRange []int
	NHiddenRange []int
	LambdRange   []float64
	NModels      int
}

type hyperparameter struct {
	Params
	NeuralNetworkModel
}

func NewHyperparameterOptimization(params Params) Hyperparameter {
	return &hyperparameter{
		Params: params,
	}
}
