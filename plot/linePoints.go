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

// save the plot to a PNG file.
func (plt *plotParameters) Save(name string) {
	// create a new plot, set its title and axis labels
	p := plot.New()
	p.Title.Text = plt.title
	p.X.Label.Text = plt.xlabel
	p.Y.Label.Text = plt.ylabel

	// draw a grid behind the data
	p.Add(plotter.NewGrid())

	// various plots to the figure
	for nplot := 0; nplot < len(plt.x); nplot++ {
		pts := make(plotter.XYs, len(plt.x[nplot]))
		for j := range pts {
			pts[j].X = plt.x[nplot][j]
			pts[j].Y = plt.y[nplot][j]
		}

		// make a line plotter with points and set its style.
		line, _, _ := plotter.NewLinePoints(pts)
		line.Color = colors[nplot]
		line.LineStyle.Width = vg.Points(1.5)

		// legend style
		p.Legend.Add(plt.legend[nplot], line)
		p.Legend.XOffs = -5. * vg.Millimeter
		p.Legend.YOffs = -5. * vg.Millimeter
		p.Legend.Padding = vg.Millimeter
		p.Legend.Top = true

		// add the plotters to the plot, with a legend
		p.Add(line)
	}

	// save the plot to a PNG file.
	err := p.Save(font.Length(plt.xwidth)*vg.Centimeter, font.Length(plt.ywidth)*vg.Centimeter, name)
	if err != nil {
		log.Panic(err)
	}
}