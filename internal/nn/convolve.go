// This file is a Go translation of src/lstm/convolve.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Convolve is NT_CONVOLVE: a stride-1 gather that stacks the
// (2*HalfX+1) x (2*HalfY+1) neighbourhood of every position into
// (2*HalfX+1)*(2*HalfY+1)*NI output features. It holds no weights — the
// learned part is the FullyConnected layer that follows it, so `Ct3,3,16`
// is this layer plus a 16x10 Tanh.
//
// Feature order is x-major, then y, then channel:
//
//	f = ((dx+HalfX) * (2*HalfY+1) + (dy+HalfY)) * NI + c
//
// Off-image taps are filled from the recognizer's LCG, not with zeros. Whether
// that matters for recognized text is unmeasured; what is certain is that
// substituting zeros will not match Tesseract's activations, so the border of
// every feature map would differ and Task 14's per-layer diff would be unusable.
type Convolve struct {
	name         string
	HalfX, HalfY int
	NI           int
	Rand         *Rand
}

func NewConvolve(name string, ni, halfX, halfY int, rnd *Rand) *Convolve {
	return &Convolve{name: name, HalfX: halfX, HalfY: halfY, NI: ni, Rand: rnd}
}

func (l *Convolve) Name() string { return l.name }

// NumOutputs recomputes no_ exactly as Convolve::DeSerialize does, overwriting
// whatever the serialized header claimed.
func (l *Convolve) NumOutputs() int {
	return l.NI * (2*l.HalfX + 1) * (2*l.HalfY + 1)
}

func (l *Convolve) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: convolve %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	yScale := 2*l.HalfY + 1
	out := NewTensor(in.Map, l.NumOutputs())
	for t := range in.Map.Len() {
		row := out.Row(t)
		outIX := 0
		for dx := -l.HalfX; dx <= l.HalfX; dx, outIX = dx+1, outIX+yScale*l.NI {
			if _, ok := in.Map.Offset(t, 0, dx); !ok {
				// The whole column of taps is outside the image.
				l.randomize(row[outIX : outIX+yScale*l.NI])
				continue
			}
			outIY := outIX
			for dy := -l.HalfY; dy <= l.HalfY; dy, outIY = dy+1, outIY+l.NI {
				src, ok := in.Map.Offset(t, dy, dx)
				if !ok {
					l.randomize(row[outIY : outIY+l.NI])
					continue
				}
				copy(row[outIY:outIY+l.NI], in.Row(src))
			}
		}
	}
	return out, nil
}

// randomize is NetworkIO::Randomize on the float path.
func (l *Convolve) randomize(dst []float32) {
	for i := range dst {
		dst[i] = float32(l.Rand.SignedRand(1.0))
	}
}
