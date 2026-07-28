// This file is a Go translation of src/lstm/network.cpp, src/lstm/plumbing.cpp,
// src/lstm/series.cpp, src/lstm/parallel.cpp, src/lstm/reversed.cpp and
// src/lstm/input.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Layer is one node of a runnable network graph.
//
// Forward returns a freshly allocated tensor rather than filling a caller-owned
// buffer. Tesseract threads a NetworkScratch through every call to recycle
// buffers; that is a performance concern, and reintroducing it before there is
// a measurement would be premature.
type Layer interface {
	// Name is the layer's serialized name. Activation dumps key on it.
	Name() string
	// NumOutputs is the feature count of the tensor Forward produces.
	NumOutputs() int
	// Forward propagates in and returns the layer's output.
	Forward(in *Tensor) (*Tensor, error)
}

// Input is NT_INPUT. Input::Forward is `*output = input`; returning the input
// tensor unchanged is safe because no layer mutates its own input.
type Input struct {
	name     string
	features int
}

func NewInput(name string, features int) *Input { return &Input{name: name, features: features} }

func (l *Input) Name() string                        { return l.name }
func (l *Input) NumOutputs() int                     { return l.features }
func (l *Input) Forward(in *Tensor) (*Tensor, error) { return in, nil }

// Series is NT_SERIES: each layer's output is the next layer's input.
type Series struct {
	name  string
	Stack []Layer
}

func NewSeries(name string, stack []Layer) (*Series, error) {
	if len(stack) == 0 {
		return nil, fmt.Errorf("nn: series %q has no layers", name)
	}
	return &Series{name: name, Stack: stack}, nil
}

func (l *Series) Name() string    { return l.name }
func (l *Series) NumOutputs() int { return l.Stack[len(l.Stack)-1].NumOutputs() }

func (l *Series) Forward(in *Tensor) (*Tensor, error) {
	cur := in
	for _, sub := range l.Stack {
		out, err := sub.Forward(cur)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", l.name, err)
		}
		cur = out
	}
	return cur, nil
}

// Parallel is NT_PARALLEL and NT_REPLICATED: every child sees the same input
// and their outputs are concatenated along the feature axis, in stack order.
type Parallel struct {
	name  string
	Stack []Layer
}

func NewParallel(name string, stack []Layer) (*Parallel, error) {
	if len(stack) == 0 {
		return nil, fmt.Errorf("nn: parallel %q has no layers", name)
	}
	return &Parallel{name: name, Stack: stack}, nil
}

func (l *Parallel) Name() string { return l.name }

func (l *Parallel) NumOutputs() int {
	n := 0
	for _, sub := range l.Stack {
		n += sub.NumOutputs()
	}
	return n
}

func (l *Parallel) Forward(in *Tensor) (*Tensor, error) {
	parts := make([]*Tensor, len(l.Stack))
	for i, sub := range l.Stack {
		out, err := sub.Forward(in)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", l.name, err)
		}
		if i > 0 && out.Map != parts[0].Map {
			return nil, fmt.Errorf("nn: parallel %q child %d produced map %v; want %v", l.name, i, out.Map, parts[0].Map)
		}
		parts[i] = out
	}
	out := NewTensor(parts[0].Map, l.NumOutputs())
	for t := range out.Map.Len() {
		dst := out.Row(t)
		off := 0
		for _, p := range parts {
			copy(dst[off:off+p.Features], p.Row(t))
			off += p.Features
		}
	}
	return out, nil
}

// ReversalKind selects which of Reversed's three transforms applies.
type ReversalKind int

const (
	// ReverseX is NT_XREVERSED (RTLReversed): mirror x within each row.
	ReverseX ReversalKind = iota
	// ReverseY is NT_YREVERSED (TTBReversed): mirror y within each column.
	ReverseY
	// TransposeXY is NT_XYTRANSPOSE: swap the two spatial dimensions.
	TransposeXY
)

// Reversed is Tesseract's Reversed plumbing: it applies its transform to the
// input, runs its single child, and applies the same transform to the child's
// output. Both transforms are involutions, so output positions realign with
// input positions.
type Reversed struct {
	name string
	Kind ReversalKind
	Sub  Layer
}

func NewReversed(name string, kind ReversalKind, sub Layer) *Reversed {
	return &Reversed{name: name, Kind: kind, Sub: sub}
}

func (l *Reversed) Name() string    { return l.name }
func (l *Reversed) NumOutputs() int { return l.Sub.NumOutputs() }

func (l *Reversed) Forward(in *Tensor) (*Tensor, error) {
	out, err := l.Sub.Forward(reverseData(in, l.Kind))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", l.name, err)
	}
	return reverseData(out, l.Kind), nil
}

// reverseData is NetworkIO::CopyWithXReversal, CopyWithYReversal and
// CopyWithXYTranspose. Cadmus has one batch, so the per-batch valid width
// Tesseract mirrors within is simply the map width.
func reverseData(src *Tensor, kind ReversalKind) *Tensor {
	m := src.Map
	if kind == TransposeXY {
		m = m.TransposeXY()
	}
	dst := NewTensor(m, src.Features)
	for y := range src.Map.Height {
		for x := range src.Map.Width {
			var dt int
			switch kind {
			case ReverseX:
				dt = m.T(y, src.Map.Width-1-x)
			case ReverseY:
				dt = m.T(src.Map.Height-1-y, x)
			case TransposeXY:
				dt = m.T(x, y)
			}
			dst.CopyTimeStep(dt, src, src.Map.T(y, x))
		}
	}
	return dst
}
