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

	labels := make([]float64, yTrain.RawMatrix().Cols)
	for j := range yTrain.RawMatrix().Cols {
		labels[j] = float64(floats.MaxIdx(mat.Col(nil, j, yTrain)))
	}

	nComponents := 50
	pca := ngo.NewPCA(nComponents)
	xTrainReduced := pca.FitTransform(xTrain)

	x := ngo.Linspace(0, float64(nComponents), len(pca.GetExplainedVariance()))
	y := make([]float64, len(pca.GetExplainedVariance()))
	floats.CumSum(y, pca.GetExplainedVariance())

	plt := plotter.NewPlot()
	plt.FigSize(12, 11)

	plt.Scatter(xTrainReduced.RawRowView(0), xTrainReduced.RawRowView(1), labels,
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
