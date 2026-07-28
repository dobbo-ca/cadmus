// This file is a Go translation of src/lstm/lstm.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Gate indices, from LSTM::WeightType in src/lstm/lstm.h. GFS, the fifth gate,
// exists only for 2-D LSTMs and is rejected at construction.
const (
	GateCI  = iota // cell input,  activated with Tanh
	GateGI         // input gate,  activated with Logistic
	GateGF1        // forget gate, activated with Logistic
	GateGO         // output gate, activated with Logistic
	numGates
)

// stateClip is kStateClip from src/lstm/lstm.cpp.
const stateClip = 100.0

// LSTM is NT_LSTM and, with Summary set, NT_LSTM_SUMMARY.
//
// NI is the layer input count, NS the internal state count (= NumOutputs), and
// NA the padded gate-input width, so every gate matrix is NS x (NA+1). The
// column layout of a gate row is
//
//	[0, NI)      the layer input at this timestep
//	[NI, NA)     the recurrent output from the previous timestep
//	[NA]         the bias, against an implicit 1.0
//
// Tesseract also supports a softmax-feedback block between those two and a
// second recurrent block for the 2-D case; neither occurs in a tessdata_best
// Latin model and both are rejected by NewLSTM.
type LSTM struct {
	name    string
	NI      int
	NS      int
	NA      int
	Summary bool
	Gates   [numGates]*Matrix

	// lastState is the cell state as computed at each row's final timestep,
	// captured BEFORE the end-of-row reset zeroes it. Kept only so tests can
	// assert the kStateClip behaviour; it is not part of the forward
	// computation. Reading `state` after the reset would always be zero.
	lastState []float64
}

func NewLSTM(name string, ni, na int, summary bool, g [numGates]*Matrix) (*LSTM, error) {
	for i, m := range g {
		if m == nil {
			return nil, fmt.Errorf("nn: lstm %q gate %d is nil", name, i)
		}
		if m.Inputs != na {
			return nil, fmt.Errorf("nn: lstm %q gate %d has %d input columns, want na=%d", name, i, m.Inputs, na)
		}
		if m.Outputs != g[GateCI].Outputs {
			return nil, fmt.Errorf("nn: lstm %q gate %d has %d outputs, want %d", name, i, m.Outputs, g[GateCI].Outputs)
		}
	}
	ns := g[GateCI].Outputs
	// LSTM::DeSerialize derives is_2d_ as na_ - nf_ == ni_ + 2*ns_. With no
	// softmax feedback nf_ is zero, so a 1-D layer has na == ni + ns exactly.
	if na == ni+2*ns {
		return nil, fmt.Errorf("nn: lstm %q is 2-D (na=%d, ni=%d, ns=%d); 2-D LSTMs and their GFS gate are out of scope for L1b", name, na, ni, ns)
	}
	if na != ni+ns {
		return nil, fmt.Errorf("nn: lstm %q has na=%d but ni+ns=%d; softmax-feedback LSTMs are out of scope for L1b", name, na, ni+ns)
	}
	return &LSTM{name: name, NI: ni, NS: ns, NA: na, Summary: summary, Gates: g}, nil
}

func (l *LSTM) Name() string    { return l.name }
func (l *LSTM) NumOutputs() int { return l.NS }

// LastState exposes the cell state as computed at the last timestep of the last
// row, captured before that row's reset, so tests can assert the kStateClip
// behaviour. It is not used by the forward pass.
func (l *LSTM) LastState() []float64 { return l.lastState }

func (l *LSTM) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: lstm %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	m := in.Map
	outMap := m
	if l.Summary {
		outMap = m.ReduceWidthTo1()
	}
	out := NewTensor(outMap, l.NS)

	// source_ is a NetworkIO, so the recurrent output is narrowed to float32
	// before it is read back into the matvec. That rounding is inside the
	// recurrence and must not be optimised away.
	source := NewTensor(m, l.NA)

	u := make([]float64, l.NA)
	var gate [numGates][]float64
	for i := range gate {
		gate[i] = make([]float64, l.NS)
	}
	state := make([]float64, l.NS)
	output := make([]float64, l.NS)
	l.lastState = make([]float64, l.NS)

	destT := 0
	for y := range m.Height {
		for x := range m.Width {
			t := m.T(y, x)
			source.CopyTimeStepPart(t, 0, l.NI, in, t, 0)
			source.WriteTimeStepPart(t, l.NI, l.NS, output)
			source.ReadTimeStep(t, u)

			l.Gates[GateCI].DotVector(u, gate[GateCI])
			TanhInPlace(gate[GateCI])
			for _, g := range [3]int{GateGI, GateGF1, GateGO} {
				l.Gates[g].DotVector(u, gate[g])
				LogisticInPlace(gate[g])
			}

			for i := range state {
				// Two statements in Tesseract: MultiplyVectorsInPlace then
				// MultiplyAccumulate. The explicit float64 conversions keep Go
				// from fusing either product into the addition.
				keep := float64(state[i] * gate[GateGF1][i])
				add := float64(gate[GateCI][i] * gate[GateGI][i])
				s := keep + add
				if s < -stateClip {
					s = -stateClip
				} else if s > stateClip {
					s = stateClip
				}
				state[i] = s
				output[i] = Tanh(s) * gate[GateGO][i]
			}

			if l.Summary {
				if x == m.Width-1 {
					out.WriteTimeStep(destT, output)
					destT++
				}
			} else {
				out.WriteTimeStep(t, output)
			}

			if x == m.Width-1 {
				// Capture before zeroing: every row is an independent sequence,
				// so the state at the row's last timestep is the only place the
				// kStateClip behaviour is observable.
				copy(l.lastState, state)
				for i := range state {
					state[i] = 0
					output[i] = 0
				}
			}
		}
	}
	return out, nil
}
