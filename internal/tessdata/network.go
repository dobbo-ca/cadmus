// This file is a Go translation of src/lstm/network.cpp, src/lstm/plumbing.cpp,
// src/lstm/lstm.cpp, src/lstm/fullyconnected.cpp, src/lstm/convolve.cpp,
// src/lstm/maxpool.cpp, src/lstm/reconfig.cpp, src/lstm/input.cpp,
// src/lstm/lstmrecognizer.cpp, src/lstm/weightmatrix.cpp and
// src/ccstruct/matrix.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

// Serialization layouts this file depends on, transcribed from the Tesseract
// sources so that later readers do not have to re-derive them.
//
// Network::Serialize (src/lstm/network.cpp) — every layer, whatever its type:
//
//	int8    type_tag           ; 0 (NT_NONE) means a type *name* string follows
//	string  type_name          ; only when type_tag == 0
//	int8    training_state     ; 1 (TS_ENABLED) means the file is a training dump
//	int8    needs_to_backprop
//	int32   network_flags
//	int32   ni
//	int32   no
//	int32   num_weights
//	string  name
//	<type-specific body>
//
// GENERIC_2D_ARRAY<T>::DeSerialize + DeSerializeSize (src/ccstruct/matrix.h).
// This is the piece the plan flagged as unspecified; field order is exactly:
//
//	int32   dim1               ; rejected above UINT16_MAX
//	int32   dim2               ; rejected above UINT16_MAX
//	T       empty_             ; ONE padding element, before the array proper
//	T       array_[dim1*dim2]  ; column-major, dim1 is the row/output count
//
// The lone `empty_` element between the dimensions and the payload is the easy
// thing to miss; omitting it desynchronises every subsequent read.
//
// WeightMatrix::DeSerialize (src/lstm/weightmatrix.cpp):
//
//	uint8   mode               ; bit0 int8 weights, bit2 Adam, bit7 new format
//	if !(mode & 128)           → DeSerializeOld, see parseMatrixShapeOld
//	if mode & 1                → GENERIC_2D_ARRAY<int8>, then uint32 n + n float64 scales
//	else                       → GENERIC_2D_ARRAY<float64>
//	                             if training: updates_, and if Adam: dw_sq_sum_
//
// LSTMRecognizer::DeSerialize (src/lstm/lstmrecognizer.cpp) wraps the root
// layer; see parseRecognizerTrailer for the fields that follow it.

import (
	"fmt"
	"io"
	"strings"
)

// LayerType mirrors Tesseract's NetworkType enum in src/lstm/network.h.
type LayerType int8

const (
	LayerNone               LayerType = 0
	LayerInput              LayerType = 1
	LayerConvolve           LayerType = 2
	LayerMaxpool            LayerType = 3
	LayerParallel           LayerType = 4
	LayerReplicated         LayerType = 5
	LayerParRLLSTM          LayerType = 6
	LayerParUDLSTM          LayerType = 7
	LayerPar2DLSTM          LayerType = 8
	LayerSeries             LayerType = 9
	LayerReconfig           LayerType = 10
	LayerXReversed          LayerType = 11
	LayerYReversed          LayerType = 12
	LayerXYTranspose        LayerType = 13
	LayerLSTM               LayerType = 14
	LayerLSTMSummary        LayerType = 15
	LayerLogistic           LayerType = 16
	LayerPosClip            LayerType = 17
	LayerSymClip            LayerType = 18
	LayerTanh               LayerType = 19
	LayerRelu               LayerType = 20
	LayerLinear             LayerType = 21
	LayerSoftmax            LayerType = 22
	LayerSoftmaxNoCTC       LayerType = 23
	LayerLSTMSoftmax        LayerType = 24
	LayerLSTMSoftmaxEncoded LayerType = 25
	LayerTensorFlow         LayerType = 26

	numLayerTypes LayerType = 27
)

// layerTypeNames is kTypeNames from src/lstm/network.cpp, in enum order. The
// strings are load-bearing: a layer whose type tag is 0 identifies itself by
// name, and the name must round-trip through this table.
var layerTypeNames = [numLayerTypes]string{
	"Invalid", "Input",
	"Convolve", "Maxpool",
	"Parallel", "Replicated",
	"ParBidiLSTM", "DepParUDLSTM",
	"Par2dLSTM", "Series",
	"Reconfig", "RTLReversed",
	"TTBReversed", "XYTranspose",
	"LSTM", "SummLSTM",
	"Logistic", "LinLogistic",
	"LinTanh", "Tanh",
	"Relu", "Linear",
	"Softmax", "SoftmaxNoCTC",
	"LSTMSoftmax", "LSTMBinarySoftmax",
	"TensorFlow",
}

func (t LayerType) String() string {
	if t >= 0 && t < numLayerTypes {
		return layerTypeNames[t]
	}
	return fmt.Sprintf("LayerType(%d)", int8(t))
}

const (
	// tsEnabled is TS_ENABLED from src/lstm/network.h. Only a training dump
	// carries the extra per-matrix update arrays.
	tsEnabled int8 = 1

	// nfLayerSpecificLR is NF_LAYER_SPECIFIC_LR from src/lstm/network.h.
	nfLayerSpecificLR int32 = 64

	// maxStackSize mirrors Plumbing::DeSerialize's guard.
	maxStackSize = 10000

	// maxMatrixDim is the UINT16_MAX cap in GENERIC_2D_ARRAY::DeSerializeSize.
	maxMatrixDim = 65535

	// maxScaleCount mirrors WeightMatrix::DeSerialize's scale-vector guard.
	maxScaleCount = 100000000
)

// MatrixShape is the shape of one serialized weight matrix. Rows is the
// matrix's output count (GENERIC_2D_ARRAY dim1), Cols its input count plus one
// for the bias (dim2).
type MatrixShape struct {
	Rows, Cols int
	Int8       bool
}

// Layer is one node of a deserialized LSTM network graph. Weight values are
// skipped; only shapes are recorded.
type Layer struct {
	Type       LayerType
	Name       string
	NumInputs  int
	NumOutputs int
	NumWeights int
	Flags      int32
	Matrices   []MatrixShape
	Children   []*Layer
}

// ParseNetwork deserializes the graph in a .traineddata LSTM component. swap
// comes from Container.Swapped.
//
// The component holds an LSTMRecognizer, not a bare network: the root layer is
// followed by the recognizer's own fields. Those are consumed and discarded so
// that the buffer can be required to end exactly where the format says it
// should — a parser that desynchronises mid-graph produces a plausible-looking
// tree and a non-empty tail, and that tail is what catches it.
func ParseNetwork(data []byte, swap bool) (*Layer, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	root, err := parseLayer(r)
	if err != nil {
		return nil, err
	}
	if r.Remaining() > 0 {
		if err := parseRecognizerTrailer(r); err != nil {
			return nil, fmt.Errorf("tessdata: %d bytes follow the network graph and are not a recognizer trailer (the graph parse desynchronised): %w", r.Remaining(), err)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the network graph; the parse desynchronised", r.Remaining())
	}
	return root, nil
}

// parseRecognizerTrailer consumes the LSTMRecognizer fields that follow the
// root layer. Models that ship lstm-unicharset and lstm-recoder as separate
// container components — every model in tessdata, tessdata_best and
// tessdata_fast — write only these eight fields here.
func parseRecognizerTrailer(r *Reader) error {
	if _, err := r.String(); err != nil { // network_str_, the build spec string
		return fmt.Errorf("reading network spec string: %w", err)
	}
	for _, field := range []string{"training_flags", "training_iteration", "sample_iteration", "null_char"} {
		if _, err := r.Int32(); err != nil {
			return fmt.Errorf("reading %s: %w", field, err)
		}
	}
	// adam_beta_, learning_rate_ and momentum_ are declared `float` in
	// src/lstm/lstmrecognizer.h — 4 bytes each, not 8.
	for _, field := range []string{"adam_beta", "learning_rate", "momentum"} {
		if _, err := r.Bytes(4); err != nil {
			return fmt.Errorf("reading %s: %w", field, err)
		}
	}
	return nil
}

// parseLayer reads the common Network header and dispatches on layer type.
func parseLayer(r *Reader) (*Layer, error) {
	typ, err := parseLayerType(r)
	if err != nil {
		return nil, err
	}
	training, err := r.Int8()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v training state: %w", typ, err)
	}
	if _, err := r.Int8(); err != nil { // needs_to_backprop
		return nil, fmt.Errorf("tessdata: reading %v backprop flag: %w", typ, err)
	}
	flags, err := r.Int32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v flags: %w", typ, err)
	}
	ni, err := r.Int32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v input count: %w", typ, err)
	}
	no, err := r.Int32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v output count: %w", typ, err)
	}
	numWeights, err := r.Int32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v weight count: %w", typ, err)
	}
	name, err := r.String()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading %v name: %w", typ, err)
	}
	if ni < 0 || no < 0 || numWeights < 0 {
		return nil, fmt.Errorf("tessdata: invalid %v layer parameters: ni=%d no=%d num_weights=%d", typ, ni, no, numWeights)
	}

	l := &Layer{
		Type:       typ,
		Name:       name,
		NumInputs:  int(ni),
		NumOutputs: int(no),
		NumWeights: int(numWeights),
		Flags:      flags,
	}
	isTraining := training == tsEnabled

	switch typ {
	case LayerSeries, LayerParallel, LayerReplicated, LayerParRLLSTM, LayerParUDLSTM,
		LayerPar2DLSTM, LayerXReversed, LayerYReversed, LayerXYTranspose:
		err = parsePlumbing(r, l)
	case LayerSoftmax, LayerSoftmaxNoCTC, LayerRelu, LayerTanh, LayerLinear,
		LayerLogistic, LayerPosClip, LayerSymClip:
		err = parseFullyConnected(r, l, isTraining)
	case LayerLSTM, LayerLSTMSummary, LayerLSTMSoftmax, LayerLSTMSoftmaxEncoded:
		err = parseLSTM(r, l, isTraining)
	case LayerConvolve:
		// half_x_, half_y_ (src/lstm/convolve.cpp).
		err = skipInt32s(r, 2)
	case LayerMaxpool, LayerReconfig:
		// x_scale_, y_scale_ (src/lstm/reconfig.cpp; Maxpool delegates to it).
		err = skipInt32s(r, 2)
	case LayerInput:
		// StaticShape: batch_, height_, width_, depth_, loss_type_.
		err = skipInt32s(r, 5)
	default:
		// NT_NONE and NT_TENSORFLOW have no deserializer in Tesseract either.
		return nil, fmt.Errorf("tessdata: unsupported network layer type %v", typ)
	}
	if err != nil {
		return nil, fmt.Errorf("tessdata: %v layer %q: %w", typ, name, err)
	}
	return l, nil
}

// parseLayerType mirrors getNetworkType in src/lstm/network.cpp: current
// Tesseract always writes a 0 tag followed by the type name, so that layer
// types can be reordered without invalidating existing models.
func parseLayerType(r *Reader) (LayerType, error) {
	tag, err := r.Int8()
	if err != nil {
		return 0, fmt.Errorf("tessdata: reading layer type tag: %w", err)
	}
	if tag != 0 {
		if tag < 0 || LayerType(tag) >= numLayerTypes {
			return 0, fmt.Errorf("tessdata: layer type tag %d out of range", tag)
		}
		return LayerType(tag), nil
	}
	name, err := r.String()
	if err != nil {
		return 0, fmt.Errorf("tessdata: reading layer type name: %w", err)
	}
	for i, n := range layerTypeNames {
		if n == name {
			return LayerType(i), nil
		}
	}
	return 0, fmt.Errorf("tessdata: unknown network layer type %q", name)
}

// parsePlumbing mirrors Plumbing::DeSerialize in src/lstm/plumbing.cpp.
func parsePlumbing(r *Reader, l *Layer) error {
	size, err := r.Uint32()
	if err != nil {
		return fmt.Errorf("reading stack size: %w", err)
	}
	if size > maxStackSize {
		return fmt.Errorf("stack size %d exceeds maximum %d", size, maxStackSize)
	}
	for i := uint32(0); i < size; i++ {
		child, err := parseLayer(r)
		if err != nil {
			return fmt.Errorf("child %d: %w", i, err)
		}
		l.Children = append(l.Children, child)
	}
	if l.Flags&nfLayerSpecificLR != 0 {
		// Plumbing::learning_rates_ is std::vector<float> — float32, not the
		// float64 that weights and scales use.
		if err := skipFloat32Slice(r); err != nil {
			return fmt.Errorf("reading layer-specific learning rates: %w", err)
		}
	}
	return nil
}

// skipFloat32Slice consumes a TFile-serialized std::vector<float>: a uint32
// count followed by that many 4-byte values.
func skipFloat32Slice(r *Reader) error {
	n, err := r.Uint32()
	if err != nil {
		return fmt.Errorf("reading count: %w", err)
	}
	if n > maxStringLen {
		return fmt.Errorf("count %d exceeds maximum %d", n, maxStringLen)
	}
	if _, err := r.Bytes(4 * int(n)); err != nil {
		return fmt.Errorf("skipping %d values: %w", n, err)
	}
	return nil
}

// parseFullyConnected mirrors FullyConnected::DeSerialize.
func parseFullyConnected(r *Reader, l *Layer, training bool) error {
	shape, err := parseMatrixShape(r, training)
	if err != nil {
		return fmt.Errorf("reading weights: %w", err)
	}
	l.Matrices = append(l.Matrices, shape)
	return nil
}

// gate weight indices, from LSTM::WeightType in src/lstm/lstm.h.
const (
	gateCI = iota
	gateGI
	gateGF1
	gateGO
	gateGFS
	numGates
)

// parseLSTM mirrors LSTM::DeSerialize in src/lstm/lstm.cpp.
func parseLSTM(r *Reader, l *Layer, training bool) error {
	na, err := r.Int32()
	if err != nil {
		return fmt.Errorf("reading na: %w", err)
	}
	// nf_ is the number of softmax outputs fed back into the input.
	var nf int
	switch l.Type {
	case LayerLSTMSoftmax:
		nf = l.NumOutputs
	case LayerLSTMSoftmaxEncoded:
		nf = ceilLog2(uint32(l.NumOutputs))
	}

	is2D := false
	for w := 0; w < numGates; w++ {
		if w == gateGFS && !is2D {
			continue
		}
		shape, err := parseMatrixShape(r, training)
		if err != nil {
			return fmt.Errorf("reading gate weights %d: %w", w, err)
		}
		l.Matrices = append(l.Matrices, shape)
		if w == gateCI {
			// ns_ = gate_weights_[CI].NumOutputs(); the 2-D test is exactly
			// Tesseract's, and decides whether a GFS matrix follows.
			ns := shape.Rows
			is2D = int(na)-nf == l.NumInputs+2*ns
		}
	}

	if l.Type == LayerLSTMSoftmax || l.Type == LayerLSTMSoftmaxEncoded {
		softmax, err := parseLayer(r)
		if err != nil {
			return fmt.Errorf("reading built-in softmax: %w", err)
		}
		l.Children = append(l.Children, softmax)
	}
	return nil
}

// parseMatrixShape mirrors WeightMatrix::DeSerialize, recording the shape and
// skipping the payload.
func parseMatrixShape(r *Reader, training bool) (MatrixShape, error) {
	mode, err := r.Uint8()
	if err != nil {
		return MatrixShape{}, fmt.Errorf("reading matrix mode: %w", err)
	}
	intMode := mode&1 != 0
	useAdam := mode&4 != 0
	if mode&128 == 0 {
		return parseMatrixShapeOld(r, training, intMode)
	}

	if intMode {
		rows, cols, err := skip2DArray(r, 1)
		if err != nil {
			return MatrixShape{}, fmt.Errorf("reading int8 weights: %w", err)
		}
		n, err := r.Uint32()
		if err != nil {
			return MatrixShape{}, fmt.Errorf("reading scale count: %w", err)
		}
		if n > maxScaleCount {
			return MatrixShape{}, fmt.Errorf("scale count %d exceeds maximum %d", n, maxScaleCount)
		}
		if _, err := r.Bytes(8 * int(n)); err != nil {
			return MatrixShape{}, fmt.Errorf("reading scales: %w", err)
		}
		return MatrixShape{Rows: rows, Cols: cols, Int8: true}, nil
	}

	rows, cols, err := skip2DArray(r, 8)
	if err != nil {
		return MatrixShape{}, fmt.Errorf("reading float64 weights: %w", err)
	}
	if training {
		if _, _, err := skip2DArray(r, 8); err != nil {
			return MatrixShape{}, fmt.Errorf("reading updates: %w", err)
		}
		if useAdam {
			if _, _, err := skip2DArray(r, 8); err != nil {
				return MatrixShape{}, fmt.Errorf("reading dw_sq_sum: %w", err)
			}
		}
	}
	return MatrixShape{Rows: rows, Cols: cols}, nil
}

// parseMatrixShapeOld mirrors WeightMatrix::DeSerializeOld, which stores
// weights as float32 and scales as a float32 vector.
func parseMatrixShapeOld(r *Reader, training, intMode bool) (MatrixShape, error) {
	var shape MatrixShape
	if intMode {
		rows, cols, err := skip2DArray(r, 1)
		if err != nil {
			return MatrixShape{}, fmt.Errorf("reading legacy int8 weights: %w", err)
		}
		if err := skipFloat32Slice(r); err != nil {
			return MatrixShape{}, fmt.Errorf("reading legacy scales: %w", err)
		}
		shape = MatrixShape{Rows: rows, Cols: cols, Int8: true}
	} else {
		rows, cols, err := skip2DArray(r, 4)
		if err != nil {
			return MatrixShape{}, fmt.Errorf("reading legacy float32 weights: %w", err)
		}
		shape = MatrixShape{Rows: rows, Cols: cols}
	}
	if training {
		// updates_, then the dead errs array. Both float32.
		for i := 0; i < 2; i++ {
			if _, _, err := skip2DArray(r, 4); err != nil {
				return MatrixShape{}, fmt.Errorf("reading legacy training array %d: %w", i, err)
			}
		}
	}
	return shape, nil
}

// skip2DArray reads a GENERIC_2D_ARRAY<T> header, skips its payload, and
// returns its dimensions. elemSize is sizeof(T).
func skip2DArray(r *Reader, elemSize int) (rows, cols int, err error) {
	dim1, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading dim1: %w", err)
	}
	dim2, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading dim2: %w", err)
	}
	if dim1 < 0 || dim1 > maxMatrixDim || dim2 < 0 || dim2 > maxMatrixDim {
		return 0, 0, fmt.Errorf("implausible matrix dimensions %dx%d", dim1, dim2)
	}
	// One empty_ element precedes the dim1*dim2 payload elements.
	if _, err := r.Bytes(elemSize * (1 + int(dim1)*int(dim2))); err != nil {
		return 0, 0, fmt.Errorf("skipping %dx%d payload: %w", dim1, dim2, err)
	}
	return int(dim1), int(dim2), nil
}

func skipInt32s(r *Reader, n int) error {
	for i := 0; i < n; i++ {
		if _, err := r.Int32(); err != nil {
			return fmt.Errorf("reading field %d: %w", i, err)
		}
	}
	return nil
}

// ceilLog2 is ceil_log2 from src/lstm/lstm.cpp.
func ceilLog2(n uint32) int {
	if n == 0 {
		return 0
	}
	l2 := 0
	for v := n; v > 1; v >>= 1 {
		l2++
	}
	if n == 1<<l2 {
		return l2
	}
	return l2 + 1
}

// Tree writes an indented one-line-per-layer rendering of the graph.
func (l *Layer) Tree(w io.Writer) { l.tree(w, 0) }

func (l *Layer) tree(w io.Writer, depth int) {
	fmt.Fprintf(w, "%s%v %q ni=%d no=%d weights=%d",
		strings.Repeat("  ", depth), l.Type, l.Name, l.NumInputs, l.NumOutputs, l.NumWeights)
	if len(l.Matrices) > 0 {
		shapes := make([]string, len(l.Matrices))
		for i, m := range l.Matrices {
			shapes[i] = fmt.Sprintf("%dx%d", m.Rows, m.Cols)
			if m.Int8 {
				shapes[i] += "(int8)"
			}
		}
		fmt.Fprintf(w, " [matrices: %s]", strings.Join(shapes, ", "))
	}
	fmt.Fprintln(w)
	for _, c := range l.Children {
		c.tree(w, depth+1)
	}
}
