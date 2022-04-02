package plot

import (
	"image/color"
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/font"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// default colors
var colors = []color.Color{
	color.RGBA{0.0, 0.0, 0.0, 255},
	color.RGBA{255, 0.0, 0.0, 255},
	color.RGBA{122, 195, 106, 255},
	color.RGBA{90, 155, 212, 255},
	color.RGBA{250, 167, 91, 255},
	color.RGBA{158, 103, 171, 255},
	color.RGBA{206, 112, 88, 255},
	color.RGBA{215, 127, 180, 255},
}

// save the plot to a PNG file
func (plt *plotParameters) Save(name string) {
	// create a new plot, set its title and axis labels
	p := plot.New()
	p.Title.Text = plt.title
	p.X.Label.Text = plt.axisLabel.xlabel
	p.Y.Label.Text = plt.axisLabel.ylabel

	// set the axis limits
	if plt.axisLimit.useXLim {
		p.X.Min = plt.axisLimit.xmin
		p.X.Max = plt.axisLimit.xmax
	}
	if plt.axisLimit.useYLim {
		p.Y.Min = plt.axisLimit.ymin
		p.Y.Max = plt.axisLimit.ymax
	}

	// make a line plotter
	plt.linePlot(p)

	// save the plot to a PNG file.
	xwdith := font.Length(plt.figSize.xwidth) * vg.Centimeter
	ywdith := font.Length(plt.figSize.ywidth) * vg.Centimeter
	err := p.Save(xwdith, ywdith, name)
	if err != nil {
		log.Panic(err)
	}
}

func (plt *plotParameters) linePlot(p *plot.Plot) {
	// draw a grid behind the data
	p.Add(plotter.NewGrid())

	// various plots to the figure
	lines := []*plotter.Line{}
	for nplot := 0; nplot < len(plt.plotData.x); nplot++ {
		pts := make(plotter.XYs, len(plt.plotData.x[nplot]))
		for j := range pts {
			pts[j].X = plt.plotData.x[nplot][j]
			pts[j].Y = plt.plotData.y[nplot][j]
		}

		// make a line plotter with points and set its style.
		line, _, _ := plotter.NewLinePoints(pts)
		line.Color = colors[nplot]
		line.LineStyle.Width = vg.Points(1.5)
		lines = append(lines, line)

		// add the plotters to the plot, with a legend
		p.Add(line)
	}

	// legend style
	for i, legend := range plt.legend {
		p.Legend.Add(legend, lines[i])
	}
	p.Legend.XOffs = vg.Millimeter
	p.Legend.YOffs = vg.Millimeter
	p.Legend.Padding = vg.Millimeter
	p.Legend.Left = true
}
