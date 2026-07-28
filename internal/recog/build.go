// This file is a Go translation of the Network::CreateFromFile dispatch in
// src/lstm/network.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// tfIntMode is TF_INT_MODE from src/lstm/lstmrecognizer.h.
const tfIntMode = 1

// Network is a loaded, runnable recognizer graph plus the scalars the decoder
// needs out of the LSTM component's trailer.
type Network struct {
	Root        nn.Layer
	InputHeight int
	NumOutputs  int
	NullChar    int
	XScale      int
	// Rand is the recognizer's TRand. Normalize draws from it first (for rows
	// the scaler left short) and every Convolve continues from there, exactly
	// as Copy2DImage precedes Convolve::Forward in Tesseract.
	Rand *nn.Rand
}

// Build converts a parsed layer tree into a runnable nn graph.
func Build(rec *tessdata.Recognizer) (*Network, error) {
	if rec.TrainingFlags&tfIntMode != 0 {
		return nil, fmt.Errorf("recog: model has TF_INT_MODE set (training_flags=%d); L1b supports float weights from tessdata_best only", rec.TrainingFlags)
	}
	rnd := nn.NewRand(uint64(int64(rec.SampleIteration) * 0x10000001))

	root, err := buildLayer(rec.Network, rnd)
	if err != nil {
		return nil, err
	}

	shape := findInputShape(rec.Network)
	if shape == nil {
		return nil, fmt.Errorf("recog: model has no Input layer")
	}
	if shape.Height <= 0 {
		return nil, fmt.Errorf("recog: input shape height is %d; a fixed height is required", shape.Height)
	}
	if shape.Depth != 1 {
		return nil, fmt.Errorf("recog: input depth %d; only 1-channel grey input is supported", shape.Depth)
	}

	n := &Network{
		Root:        root,
		InputHeight: shape.Height,
		NumOutputs:  root.NumOutputs(),
		// tessdata.Recognizer.NullChar is int32 (L1a Task 4).
		NullChar: int(rec.NullChar),
		XScale:   xScaleFactor(rec.Network),
		Rand:     rnd,
	}
	// null_char_ is the authority for the CTC blank; it is structurally free to
	// differ from NumOutputs-1, so it is read rather than assumed. It must
	// nonetheless be a valid output index.
	if n.NullChar < 0 || n.NullChar >= n.NumOutputs {
		return nil, fmt.Errorf("recog: null_char %d is outside the %d network outputs", n.NullChar, n.NumOutputs)
	}
	return n, nil
}

func buildLayer(l *tessdata.Layer, rnd *nn.Rand) (nn.Layer, error) {
	switch l.Type {
	case tessdata.LayerInput:
		return nn.NewInput(l.Name, l.NumOutputs), nil

	case tessdata.LayerSeries, tessdata.LayerParallel, tessdata.LayerReplicated:
		stack, err := buildStack(l, rnd)
		if err != nil {
			return nil, err
		}
		if l.Type == tessdata.LayerSeries {
			return nn.NewSeries(l.Name, stack)
		}
		return nn.NewParallel(l.Name, stack)

	case tessdata.LayerXReversed, tessdata.LayerYReversed, tessdata.LayerXYTranspose:
		stack, err := buildStack(l, rnd)
		if err != nil {
			return nil, err
		}
		if len(stack) != 1 {
			return nil, fmt.Errorf("recog: %v %q has %d children; want exactly 1", l.Type, l.Name, len(stack))
		}
		kind := map[tessdata.LayerType]nn.ReversalKind{
			tessdata.LayerXReversed:   nn.ReverseX,
			tessdata.LayerYReversed:   nn.ReverseY,
			tessdata.LayerXYTranspose: nn.TransposeXY,
		}[l.Type]
		return nn.NewReversed(l.Name, kind, stack[0]), nil

	case tessdata.LayerConvolve:
		return nn.NewConvolve(l.Name, l.NumInputs, l.HalfX, l.HalfY, rnd), nil

	case tessdata.LayerMaxpool:
		return nn.NewMaxpool(l.Name, l.NumInputs, l.XScale, l.YScale), nil

	case tessdata.LayerReconfig:
		return nn.NewReconfig(l.Name, l.NumInputs, l.XScale, l.YScale), nil

	case tessdata.LayerSoftmax, tessdata.LayerSoftmaxNoCTC, tessdata.LayerTanh,
		tessdata.LayerRelu, tessdata.LayerLinear, tessdata.LayerLogistic,
		tessdata.LayerPosClip, tessdata.LayerSymClip:
		if len(l.Matrices) != 1 {
			return nil, fmt.Errorf("recog: %v %q has %d weight matrices; want 1", l.Type, l.Name, len(l.Matrices))
		}
		m, err := convertMatrix(l.Name, l.Matrices[0])
		if err != nil {
			return nil, err
		}
		// FullyConnected's matrix is no x (ni+1). Both dimensions are checked
		// because a mismatch means the graph parse or the weight load slipped.
		if m.Outputs != l.NumOutputs || m.Inputs != l.NumInputs {
			return nil, fmt.Errorf("recog: %v %q matrix is %dx%d; want %dx%d from the layer header",
				l.Type, l.Name, m.Outputs, m.Inputs+1, l.NumOutputs, l.NumInputs+1)
		}
		act := map[tessdata.LayerType]nn.Activation{
			tessdata.LayerSoftmax:      nn.ActSoftmax,
			tessdata.LayerSoftmaxNoCTC: nn.ActSoftmax,
			tessdata.LayerTanh:         nn.ActTanh,
			tessdata.LayerRelu:         nn.ActRelu,
			tessdata.LayerLinear:       nn.ActLinear,
			tessdata.LayerLogistic:     nn.ActLogistic,
			tessdata.LayerPosClip:      nn.ActPosClip,
			tessdata.LayerSymClip:      nn.ActSymClip,
		}[l.Type]
		if err := checkNumWeights(l); err != nil {
			return nil, err
		}
		return nn.NewFullyConnected(l.Name, act, m), nil

	case tessdata.LayerLSTM, tessdata.LayerLSTMSummary:
		if len(l.Matrices) != 4 {
			return nil, fmt.Errorf("recog: LSTM %q has %d gate matrices; want 4 (a 5th means a 2-D layer, which is out of scope)", l.Name, len(l.Matrices))
		}
		var gates [4]*nn.Matrix
		for i, src := range l.Matrices {
			m, err := convertMatrix(l.Name, src)
			if err != nil {
				return nil, err
			}
			if m.Inputs != l.NA {
				return nil, fmt.Errorf("recog: LSTM %q gate %d has %d input columns; want na=%d", l.Name, i, m.Inputs, l.NA)
			}
			gates[i] = m
		}
		if err := checkNumWeights(l); err != nil {
			return nil, err
		}
		return nn.NewLSTM(l.Name, l.NumInputs, l.NA, l.Type == tessdata.LayerLSTMSummary, gates)

	default:
		return nil, fmt.Errorf("recog: layer type %v (%q) is not supported by the L1b runtime", l.Type, l.Name)
	}
}

func buildStack(l *tessdata.Layer, rnd *nn.Rand) ([]nn.Layer, error) {
	stack := make([]nn.Layer, 0, len(l.Children))
	for i, c := range l.Children {
		sub, err := buildLayer(c, rnd)
		if err != nil {
			return nil, fmt.Errorf("%v %q child %d: %w", l.Type, l.Name, i, err)
		}
		stack = append(stack, sub)
	}
	return stack, nil
}

// checkNumWeights cross-checks a weight-bearing layer's serialized num_weights
// against the element count of the matrices actually loaded, bias columns
// included. It is free validation that the weight load stayed aligned:
// ConvNL 160 = 16x10, Lfx96 61824 = 4x96x161, Output 56943 = 111x513.
func checkNumWeights(l *tessdata.Layer) error {
	total := 0
	for _, m := range l.Matrices {
		total += m.Rows * m.Cols
	}
	if total != l.NumWeights {
		return fmt.Errorf("recog: layer %q matrices hold %d weights but the header says %d", l.Name, total, l.NumWeights)
	}
	return nil
}

// convertMatrix reshapes a loader matrix into a runtime one. The loader's Cols
// counts the bias column; the runtime's Inputs does not. The loader's field is
// named Values (L1a Task 3), not W.
func convertMatrix(name string, m tessdata.Matrix) (*nn.Matrix, error) {
	if m.Cols < 1 {
		return nil, fmt.Errorf("recog: layer %q has a matrix with %d columns", name, m.Cols)
	}
	return nn.NewMatrix(m.Rows, m.Cols-1, m.Values)
}

func findInputShape(l *tessdata.Layer) *tessdata.InputShape {
	if l.Shape != nil {
		return l.Shape
	}
	for _, c := range l.Children {
		if s := findInputShape(c); s != nil {
			return s
		}
	}
	return nil
}

// xScaleFactor is Network::XScaleFactor, and the Series case is the ONLY one
// that multiplies.
//
//	src/lstm/network.h:211   Network::XScaleFactor()  -> 1
//	src/lstm/reconfig.cpp    Reconfig::XScaleFactor() -> x_scale_ (Maxpool derives)
//	src/lstm/plumbing.cpp    Plumbing::XScaleFactor() -> stack_[0]->XScaleFactor()
//	src/lstm/series.cpp:91   Series::XScaleFactor()   -> product over the stack
//
// Parallel, Replicated and the three Reversed variants inherit Plumbing's
// version, which takes the FIRST child only. Treating them as products is
// latent for eng — every plumbing node there has exactly one child, so the
// product equals stack[0] — but it is wrong for any Parallel with two or more
// children, and Task 1 Step 2 tells the reader to fall back on this function.
func xScaleFactor(l *tessdata.Layer) int {
	switch l.Type {
	case tessdata.LayerMaxpool, tessdata.LayerReconfig:
		return l.XScale

	case tessdata.LayerSeries:
		f := 1
		for _, c := range l.Children {
			f *= xScaleFactor(c)
		}
		return f

	case tessdata.LayerParallel, tessdata.LayerReplicated, tessdata.LayerParRLLSTM,
		tessdata.LayerParUDLSTM, tessdata.LayerPar2DLSTM,
		tessdata.LayerXReversed, tessdata.LayerYReversed, tessdata.LayerXYTranspose:
		if len(l.Children) == 0 {
			return 1
		}
		return xScaleFactor(l.Children[0])

	default:
		return 1
	}
}
