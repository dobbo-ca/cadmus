// This file is a Go translation of src/lstm/reconfig.cpp and
// src/lstm/maxpool.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Reconfig is NT_RECONFIG and, with Max set, NT_MAXPOOL. The two share their
// serialized fields and their traversal; Maxpool differs only in taking the
// elementwise maximum over the window instead of concatenating it, and in
// forcing the output depth back to ni.
//
// The output map is floor-divided in both dimensions (StrideMap::ScaleXY), so a
// partial trailing window is dropped rather than padded — and because the
// dimensions are floored first, every tap of every window is in range.
type Reconfig struct {
	name           string
	XScale, YScale int
	NI             int
	Max            bool
}

func NewReconfig(name string, ni, xScale, yScale int) *Reconfig {
	return &Reconfig{name: name, XScale: xScale, YScale: yScale, NI: ni}
}

func NewMaxpool(name string, ni, xScale, yScale int) *Reconfig {
	return &Reconfig{name: name, XScale: xScale, YScale: yScale, NI: ni, Max: true}
}

func (l *Reconfig) Name() string { return l.name }

func (l *Reconfig) NumOutputs() int {
	if l.Max {
		return l.NI
	}
	return l.NI * l.XScale * l.YScale
}

func (l *Reconfig) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	if l.XScale <= 0 || l.YScale <= 0 {
		return nil, fmt.Errorf("nn: %q has invalid scale %dx%d", l.name, l.XScale, l.YScale)
	}
	outMap := in.Map.ScaleXY(l.XScale, l.YScale)
	out := NewTensor(outMap, l.NumOutputs())
	for oy := range outMap.Height {
		for ox := range outMap.Width {
			ot := outMap.T(oy, ox)
			origin := in.Map.T(oy*l.YScale, ox*l.XScale)
			if l.Max {
				out.CopyTimeStep(ot, in, origin)
			}
			for dx := range l.XScale {
				for dy := range l.YScale {
					src := in.Map.T(oy*l.YScale+dy, ox*l.XScale+dx)
					if l.Max {
						out.MaxpoolTimeStep(ot, in, src)
					} else {
						out.CopyTimeStepPart(ot, (dx*l.YScale+dy)*l.NI, l.NI, in, src, 0)
					}
				}
			}
		}
	}
	return out, nil
}
