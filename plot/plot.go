package plot

type plotParameters struct {
	x, y   [][]float64
	title  string
	xlabel string
	ylabel string
	legend []string
	xwidth int
	ywidth int
}

func NewPlot() plotParameters {
	return plotParameters{}
}

func (plt *plotParameters) Plot(x []float64, y []float64) {
	plt.x = append(plt.x, x)
	plt.y = append(plt.y, y)
}

func (plt *plotParameters) FigSize(xwidth, ywidth int) {
	plt.xwidth = xwidth
	plt.ywidth = ywidth
}

func (plt *plotParameters) Title(str string) {
	plt.title = str
}

func (plt *plotParameters) XLabel(strx string) {
	plt.xlabel = strx
}

func (plt *plotParameters) YLabel(stry string) {
	plt.ylabel = stry
}

func (plt *plotParameters) Legend(str ...string) {
	plt.legend = append(plt.legend, str...)
}