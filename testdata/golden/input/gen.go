//go:build ignore

// Command gen writes testdata/golden/input/scan.png, the synthetic page image
// every L0 golden is derived from. Run it from the repository root:
//
//	go run testdata/golden/input/gen.go
//
// The image is deliberately synthetic: L0 tests only require that Go and
// Leptonica agree on the same pixels, and a generated input keeps the repo free
// of binary assets of uncertain provenance. All randomness comes from a fixed
// PCG seed, so regeneration is byte-identical.
package main

import (
	"image"
	"image/color"
	"image/png"
	"math/rand/v2"
	"os"
)

const (
	outPath = "testdata/golden/input/scan.png"
	width   = 1000
	height  = 1400
)

func main() {
	rng := rand.New(rand.NewPCG(1, 2))
	img := image.NewGray(image.Rect(0, 0, width, height))

	// Light background: a gentle horizontal gradient from 220 to 240 plus
	// low-amplitude noise, so thresholding is not trivially uniform.
	for y := range height {
		for x := range width {
			v := 220 + (20*x)/(width-1) + rng.IntN(7) - 3
			img.SetGray(x, y, color.Gray{Y: uint8(clamp(v))})
		}
	}

	// Two columns of dark bars standing in for text lines.
	columns := []int{80, 540}
	for row := range 20 {
		y := 100 + row*62
		for _, x0 := range columns {
			h := 12 + rng.IntN(9)
			w := 240 + rng.IntN(140)
			ink := 25 + rng.IntN(35)
			fill(img, x0, y, w, h, uint8(ink))
		}
	}

	// A 3px dark vertical rule, standing in for a table border or scan artifact.
	fill(img, 500, 60, 3, height-120, 20)

	// Isolated 1-2px specks, for despeckling and connected-component edges.
	for range 12 {
		x := 20 + rng.IntN(width-40)
		y := 20 + rng.IntN(height-40)
		s := 1 + rng.IntN(2)
		fill(img, x, y, s, s, uint8(30+rng.IntN(30)))
	}

	f, err := os.Create(outPath)
	if err != nil {
		panic(err)
	}
	if err := png.Encode(f, img); err != nil {
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
}

func fill(img *image.Gray, x0, y0, w, h int, v uint8) {
	for y := y0; y < y0+h && y < height; y++ {
		for x := x0; x < x0+w && x < width; x++ {
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
}

func clamp(v int) int {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}
