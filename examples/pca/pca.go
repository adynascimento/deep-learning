package main

import (
	"fmt"

	"github.com/adynascimento/deep-learning/ngo"
	"github.com/adynascimento/plot/plotter"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// training data
	xTrain := LoadDataFromFile("../dataset/mnist/train_x.csv")
	yTrain := LoadDataFromFile("../dataset/mnist/train_label.csv")

	applyNormalization := func(_, _ int, v float64) float64 { return v / 255.0 }
	xTrain = ngo.Apply(applyNormalization, xTrain)

	labels := make([]float64, yTrain.RawMatrix().Rows)
	for i := range labels {
		labels[i] = float64(floats.MaxIdx(yTrain.RawRowView(i)))
	}

	nComponents := 50
	pca := ngo.NewPCA(nComponents)
	xTrainReduced := pca.FitTransform(xTrain)
	variance := pca.GetExplainedVariance()

	x := ngo.Linspace(1, float64(nComponents), len(variance))
	y := make([]float64, len(variance))
	floats.CumSum(y, variance)

	plt := plotter.NewPlot()
	plt.FigSize(12, 11)

	pc1 := mat.Col(nil, 0, xTrainReduced)
	pc2 := mat.Col(nil, 1, xTrainReduced)
	plt.Scatter(pc1, pc2, labels,
		plotter.WithScatterColorMap(plotter.Tab10),
		plotter.WithScatterColorbar(plotter.Vertical),
	)
	plt.Title("pca dimensionality reduction")
	plt.XLabel("principal component 1")
	plt.YLabel("principal component 2")
	plt.XLim(-5, 9)
	plt.Grid()

	plt.Save("pca_reduction.png")

	plt = plotter.NewPlot()
	plt.FigSize(12, 11)

	plt.Plot(x, y)
	plt.Plot(x, ngo.Linspace(floats.Max(y), floats.Max(y), len(x)),
		plotter.WithLineColor(plotter.Red),
		plotter.WithLineStyle(plotter.Dashed),
	)
	plt.Title("cumulative explained variance")
	plt.XLabel("number of principal components")
	plt.YLabel("proportion of explained variance")
	plt.XLim(0, 53)
	plt.YLim(0, 0.9)
	plt.Legend("explained variance",
		fmt.Sprintf("retained variance (%.0f%%)", 100*floats.Max(y))).Location(plotter.LowerRight)
	plt.Grid()

	plt.Save("pca_variance.png")
}
