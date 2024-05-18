package hyperparameter

type NeuralNetworkModel func(int, []int, float64) float64

type Hyperparameter interface {
	RandomSearchOptimization(model NeuralNetworkModel)
	BayesianOptimization(model NeuralNetworkModel)
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
