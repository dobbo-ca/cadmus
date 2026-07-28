// This file is a Go translation of src/lstm/fullyconnected.cpp from Tesseract
// OCR (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Activation selects the nonlinearity FullyConnected::ForwardTimeStep applies.
type Activation int

const (
	ActLinear   Activation = iota // NT_LINEAR
	ActTanh                       // NT_TANH
	ActLogistic                   // NT_LOGISTIC
	ActRelu                       // NT_RELU
	ActSoftmax                    // NT_SOFTMAX and NT_SOFTMAX_NO_CTC
	ActPosClip                    // NT_POSCLIP
	ActSymClip                    // NT_SYMCLIP
)

// FullyConnected is Tesseract's FullyConnected: one weight matrix applied per
// timestep, followed by the type's nonlinearity.
//
// Tesseract iterates t over the raw buffer width and zeroes the padding
// afterwards with ZeroInvalidElements. Cadmus has one batch and therefore no
// padding, so every timestep is valid and the cleanup pass does not exist.
type FullyConnected struct {
	name string
	Act  Activation
	W    *Matrix
}

func NewFullyConnected(name string, act Activation, w *Matrix) *FullyConnected {
	return &FullyConnected{name: name, Act: act, W: w}
}

func (l *FullyConnected) Name() string    { return l.name }
func (l *FullyConnected) NumOutputs() int { return l.W.Outputs }

func (l *FullyConnected) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.W.Inputs {
		return nil, fmt.Errorf("nn: %q got %d input features, want %d", l.name, in.Features, l.W.Inputs)
	}
	out := NewTensor(in.Map, l.W.Outputs)
	u := make([]float64, l.W.Inputs)
	v := make([]float64, l.W.Outputs)
	for t := range in.Map.Len() {
		in.ReadTimeStep(t, u)
		l.W.DotVector(u, v)
		l.activate(v)
		out.WriteTimeStep(t, v)
	}
	return out, nil
}

func (l *FullyConnected) activate(v []float64) {
	switch l.Act {
	case ActLinear:
	case ActTanh:
		TanhInPlace(v)
	case ActLogistic:
		LogisticInPlace(v)
	case ActRelu:
		for i, x := range v {
			if x < 0 {
				v[i] = 0
			}
		}
	case ActSoftmax:
		SoftmaxInPlace(v)
	case ActPosClip:
		clipInPlace(v, 0, 1)
	case ActSymClip:
		clipInPlace(v, -1, 1)
	}
}

func clipInPlace(v []float64, lo, hi float64) {
	for i, x := range v {
		if x < lo {
			v[i] = lo
		} else if x > hi {
			v[i] = hi
		}
	}
}
