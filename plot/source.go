package plot

type plotParameters struct {
	plotData  plotData  // data for lines plot
	title     string    // title plot
	legend    []string  // mainly used in lines plots
	axisLabel axisLabel // xlabel and ylabel for all plots
	axisLimit axisLimit // x-axis and y-axis limits
	figSize   figSize   // xwidth and ywidth of the saved figure
}

type plotData struct{ x, y [][]float64 }
type axisLabel struct{ xlabel, ylabel string }
type figSize struct{ xwidth, ywidth int }
type axisLimit struct {
	xmin, xmax, ymin, ymax float64
	useXLim, useYLim       bool
}

func NewPlot() plotParameters {
	return plotParameters{}
}

// parameters to lines plots
func (plt *plotParameters) Plot(x []float64, y []float64) {
	plt.plotData.x = append(plt.plotData.x, x)
	plt.plotData.y = append(plt.plotData.y, y)
}

// size of the saved figure
func (plt *plotParameters) FigSize(xwidth, ywidth int) {
	plt.figSize.xwidth = xwidth
	plt.figSize.ywidth = ywidth
}

// title for all plots
func (plt *plotParameters) Title(str string) {
	plt.title = str
}

// xlabel for all plots
func (plt *plotParameters) XLabel(xlabel string) {
	plt.axisLabel.xlabel = xlabel
}

// ylabel for all plots
func (plt *plotParameters) YLabel(ylabel string) {
	plt.axisLabel.ylabel = ylabel
}

// legend mainly used in lines plots
func (plt *plotParameters) Legend(str ...string) {
	plt.legend = append(plt.legend, str...)
}

// set the x-axis vies limits
func (plt *plotParameters) XLim(xmin, xmax float64) {
	if xmin < xmax {
		plt.axisLimit.xmin = xmin
		plt.axisLimit.xmax = xmax
		plt.axisLimit.useXLim = true
	}
}

// set the x-axis vies limits
func (plt *plotParameters) YLim(ymin, ymax float64) {
	if ymin < ymax {
		plt.axisLimit.ymin = ymin
		plt.axisLimit.ymax = ymax
		plt.axisLimit.useYLim = true
	}
}
