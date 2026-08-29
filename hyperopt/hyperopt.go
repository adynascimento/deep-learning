package hyperopt

import (
	"github.com/c-bata/goptuna"
)

type StudyDirection string
type Objective func(goptuna.Trial) (float64, error)

const (
	Maximize StudyDirection = "maximize" // maximizes objective function value
	Minimize StudyDirection = "minimize" // minimizes objective function value
)

func Optimize(sampler Sampler, direction StudyDirection, objective Objective, nTrials int) (goptuna.FrozenTrial, error) {
	goptunaSampler := NewSampler(sampler)

	// create a study which manages each experiment
	study, err := goptuna.CreateStudy(
		"hyperparameter-optimization",
		goptuna.StudyOptionSampler(goptunaSampler),
		goptuna.StudyOptionDirection(goptuna.StudyDirection(direction)))
	if err != nil {
		return goptuna.FrozenTrial{}, err
	}

	// evaluate objective function
	err = study.Optimize(goptuna.FuncObjective(objective), nTrials)
	if err != nil {
		return goptuna.FrozenTrial{}, err
	}

	// best evaluation parameters
	return study.Storage.GetBestTrial(study.ID)
}
