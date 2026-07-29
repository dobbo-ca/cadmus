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
//	if !(mode & 128)           → DeSerializeOld (float32). REJECTED by parseMatrix.
//	if mode & 1                → GENERIC_2D_ARRAY<int8> + scales. REJECTED by parseMatrix.
//	else                       → GENERIC_2D_ARRAY<float64>, see readFloat64Matrix
//	                             (training dumps, which add updates_/dw_sq_sum_,
//	                             are rejected in parseLayer on the TS_ENABLED
//	                             training-state byte)
//
// LSTMRecognizer::DeSerialize (src/lstm/lstmrecognizer.cpp:133) wraps the root
// layer: the lstm component is a serialized LSTMRecognizer, not a bare Network,
// and eight more fields follow the graph. With include_charsets == false — the
// case for every model that ships lstm-unicharset and lstm-recoder as separate
// container components — the trailer is exactly:
//
//	string  network_str_        ; the build spec, e.g. "[1,36,0,1Ct3,...O1c1]"
//	int32   training_flags_     ; TF_INT_MODE=1, TF_COMPRESS_UNICHARSET=64
//	int32   training_iteration_
//	int32   sample_iteration_
//	int32   null_char_
//	f32     adam_beta_          ; 4 bytes, not 8
//	f32     learning_rate_
//	f32     momentum_
//
// See ParseRecognizer, which reads and retains all eight.

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
)

const (
	// Flags on WeightMatrix::DeSerialize's leading mode byte
	// (src/lstm/weightmatrix.cpp:229).
	modeInt8Flag uint8 = 1
	// modeAdamFlag is declared for documentation and is deliberately NOT
	// consulted: whether the Adam arrays follow a matrix is decided by the
	// layer's training-state byte, not by this flag, and parseLayer rejects
	// training dumps outright. Keying off it would desynchronise the stream —
	// every matrix in tessdata_best eng has mode 132, kAdamFlag set, and no
	// Adam arrays.
	modeAdamFlag   uint8 = 4
	modeDoubleFlag uint8 = 128
)

// Matrix is one deserialized weight matrix.
//
// Values is row-major with row stride Cols: element (row, col) is
// Values[row*Cols+col]. GENERIC_2D_ARRAY's index() is
// `column*dim2_ + row`, and its "column" is Tesseract's *output* index — so
// despite the header comment in src/ccstruct/matrix.h calling the storage
// column-major, the effective address math is dim1 contiguous runs of dim2.
//
// Rows is the output count (dim1). Cols is the input count plus ONE trailing
// bias column (dim2): MatrixDotVectorInternal computes the dot product over
// dim2-1 elements and then adds w[i][dim2-1] against an implicit 1.0
// (src/lstm/weightmatrix.cpp:99).
type Matrix struct {
	Rows, Cols int
	Values     []float64
}

// At returns the weight connecting input col to output row.
func (m *Matrix) At(row, col int) float64 { return m.Values[row*m.Cols+col] }

// Stats returns the minimum, maximum and arithmetic mean of the matrix's
// values. An empty matrix returns zeroes. Used by cadmusdump and by the tests
// that check the payload is real numbers rather than misaligned garbage.
func (m *Matrix) Stats() (min, max, mean float64) {
	if len(m.Values) == 0 {
		return 0, 0, 0
	}
	min, max = m.Values[0], m.Values[0]
	var sum float64
	for _, v := range m.Values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}
	return min, max, sum / float64(len(m.Values))
}

// InputShape is Tesseract's StaticShape, the 4-D tensor description an Input
// layer carries (src/lstm/static_shape.h). Width 0 means "determined at
// runtime".
type InputShape struct {
	Batch, Height, Width, Depth, LossType int
}

// Layer is one node of a deserialized LSTM network graph, with its weight
// values.
type Layer struct {
	Type       LayerType
	Name       string
	NumInputs  int
	NumOutputs int
	NumWeights int
	Flags      int32
	Matrices   []Matrix
	Children   []*Layer

	// Type-specific fields, zero/nil unless the layer type sets them.
	HalfX, HalfY int         // Convolve:          half_x_, half_y_
	XScale       int         // Maxpool, Reconfig: x_scale_
	YScale       int         // Maxpool, Reconfig: y_scale_
	Shape        *InputShape // Input only
	NA           int         // LSTM family:       na_
}

// TrainingFlags bits, from src/lstm/lstmrecognizer.h:44.
const (
	tfIntMode            int32 = 1
	tfCompressUnicharset int32 = 64
)

// Recognizer is a parsed lstm component: the network graph plus the
// LSTMRecognizer fields serialized after it.
type Recognizer struct {
	Network *Layer

	// NetworkStr is the spec string the model was built from, e.g.
	// "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]". Its output count is
	// a PLACEHOLDER: NetworkBuilder::ParseOutput overrides the number after
	// O1c with the recoder's code range, which is why released models persist
	// "O1c1". Never derive the output count from this string.
	NetworkStr        string
	TrainingFlags     int32
	TrainingIteration int32

	// SampleIteration seeds LSTMRecognizer::SetRandomSeed, which drives the
	// random edge padding Convolve applies at image borders. L1b needs it.
	SampleIteration int32

	// NullChar is the network output index of the CTC blank. It is the
	// authority: UnicharCompress::DefragmentCodeValues relocates the null code
	// to the top of the range, so it is code_range-1 in practice, but the
	// format does not require that and this field does.
	NullChar int32

	AdamBeta     float32
	LearningRate float32
	Momentum     float32
}

// IsIntMode reports whether the model carries int8-quantized weights.
func (rec *Recognizer) IsIntMode() bool { return rec.TrainingFlags&tfIntMode != 0 }

// ParseRecognizer deserializes a .traineddata lstm component. swap comes from
// Container.Swapped.
//
// The component holds an LSTMRecognizer, not a bare network: the root layer is
// followed by the recognizer's own fields. Requiring the buffer to end exactly
// where the format says it should is what catches a parser that desynchronised
// mid-graph and returned a plausible-looking tree.
func ParseRecognizer(data []byte, swap bool) (*Recognizer, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	root, err := parseLayer(r)
	if err != nil {
		return nil, err
	}
	rec := &Recognizer{Network: root}
	if rec.NetworkStr, err = r.String(); err != nil {
		return nil, fmt.Errorf("tessdata: reading network spec string (the graph parse may have desynchronised): %w", err)
	}
	for _, f := range []struct {
		name string
		dst  *int32
	}{
		{"training_flags", &rec.TrainingFlags},
		{"training_iteration", &rec.TrainingIteration},
		{"sample_iteration", &rec.SampleIteration},
		{"null_char", &rec.NullChar},
	} {
		if *f.dst, err = r.Int32(); err != nil {
			return nil, fmt.Errorf("tessdata: reading %s: %w", f.name, err)
		}
	}
	// adam_beta_, learning_rate_ and momentum_ are declared `float` in
	// src/lstm/lstmrecognizer.h:350 — 4 bytes each, not 8.
	for _, f := range []struct {
		name string
		dst  *float32
	}{
		{"adam_beta", &rec.AdamBeta},
		{"learning_rate", &rec.LearningRate},
		{"momentum", &rec.Momentum},
	} {
		if *f.dst, err = r.Float32(); err != nil {
			return nil, fmt.Errorf("tessdata: reading %s: %w", f.name, err)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the lstm component; the parse desynchronised", r.Remaining())
	}

	if rec.IsIntMode() {
		return nil, fmt.Errorf("tessdata: training_flags %d has TF_INT_MODE set: this is an int8-quantized model. Cadmus L1 implements the float weight path only; run ./testdata/fetch.sh to get the tessdata_best model", rec.TrainingFlags)
	}
	if rec.TrainingFlags&tfCompressUnicharset == 0 {
		return nil, fmt.Errorf("tessdata: training_flags %d has TF_COMPRESS_UNICHARSET clear: the model has no recoder and LSTMRecognizer::LoadRecoder would synthesise a pass-through one. Cadmus requires a recoding model", rec.TrainingFlags)
	}
	if rec.NullChar < 0 || int(rec.NullChar) >= root.NumOutputs {
		return nil, fmt.Errorf("tessdata: null_char %d is outside the network's %d outputs", rec.NullChar, root.NumOutputs)
	}
	return rec, nil
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
	// TS_ENABLED means the file is a training dump, whose weight matrices carry
	// an extra updates_ array (and dw_sq_sum_ when kAdamFlag is set) that this
	// parser does not read. LSTMTrainer::SaveRecognitionDump flips the state to
	// TS_TEMP_DISABLE before writing, so no released model has TS_ENABLED;
	// every layer of tessdata_best eng has state 2.
	if training == tsEnabled {
		return nil, fmt.Errorf("tessdata: %v layer %q has training state TS_ENABLED: training dumps are not supported", typ, name)
	}

	l := &Layer{
		Type:       typ,
		Name:       name,
		NumInputs:  int(ni),
		NumOutputs: int(no),
		NumWeights: int(numWeights),
		Flags:      flags,
	}

	switch typ {
	case LayerSeries, LayerParallel, LayerReplicated, LayerParRLLSTM, LayerParUDLSTM,
		LayerPar2DLSTM, LayerXReversed, LayerYReversed, LayerXYTranspose:
		err = parsePlumbing(r, l)
	case LayerSoftmax, LayerSoftmaxNoCTC, LayerRelu, LayerTanh, LayerLinear,
		LayerLogistic, LayerPosClip, LayerSymClip:
		err = parseFullyConnected(r, l)
	case LayerLSTM, LayerLSTMSummary, LayerLSTMSoftmax, LayerLSTMSoftmaxEncoded:
		err = parseLSTM(r, l)
	case LayerConvolve:
		// half_x_, half_y_ (src/lstm/convolve.cpp:45). Convolve holds no
		// weights: it is a pure im2col gather and recomputes
		// no_ = ni_*(2*half_x+1)*(2*half_y+1).
		l.HalfX, l.HalfY, err = parseInt32Pair(r)
	case LayerMaxpool, LayerReconfig:
		// x_scale_, y_scale_ (src/lstm/reconfig.cpp:60). Maxpool serializes
		// identical bytes and then overrides no_ = ni_ (src/lstm/maxpool.cpp:29).
		l.XScale, l.YScale, err = parseInt32Pair(r)
	case LayerInput:
		l.Shape, err = parseInputShape(r)
	default:
		// NT_NONE and NT_TENSORFLOW have no deserializer in Tesseract either.
		return nil, fmt.Errorf("tessdata: unsupported network layer type %v", typ)
	}
	if err != nil {
		return nil, fmt.Errorf("tessdata: %v layer %q: %w", typ, name, err)
	}

	// A layer's num_weights is the total element count of its own matrices
	// (bias columns included) plus its children's num_weights. Verified exactly
	// on every layer of tessdata_best eng: ConvNL 160 == 16*10,
	// Lfx96 61824 == 4*96*161, Output 56943 == 111*513, root Series 1461007 ==
	// the sum of its children. A mismatch means the parse desynchronised.
	//
	// The identity holds by construction for every layer type, including
	// NT_LSTM_SOFTMAX / NT_LSTM_SOFTMAX_ENCODED, whose nested softmax lives in
	// Children and which eng does not use:
	//
	//	LSTM::InitWeights           src/lstm/lstm.cpp:175   sums the gate
	//	                            matrices, then adds softmax_->InitWeights()
	//	                            when softmax_ != nullptr.
	//	Plumbing::InitWeights       src/lstm/plumbing.cpp:50 sums its stack.
	//	FullyConnected::InitWeights src/lstm/fullyconnected.cpp:86 is
	//	                            no_ * (ni_ + 1), i.e. rows*cols.
	total := 0
	for i := range l.Matrices {
		total += l.Matrices[i].Rows * l.Matrices[i].Cols
	}
	for _, c := range l.Children {
		total += c.NumWeights
	}
	if total != l.NumWeights {
		return nil, fmt.Errorf("tessdata: %v layer %q declares num_weights=%d but its matrices and children total %d", typ, name, l.NumWeights, total)
	}
	return l, nil
}

func parseInt32Pair(r *Reader) (int, int, error) {
	a, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading first field: %w", err)
	}
	b, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading second field: %w", err)
	}
	return int(a), int(b), nil
}

// parseInputShape reads StaticShape's batch_, height_, width_, depth_ and
// loss_type_, in that order (src/lstm/static_shape.h:83).
func parseInputShape(r *Reader) (*InputShape, error) {
	var s InputShape
	for i, dst := range []*int{&s.Batch, &s.Height, &s.Width, &s.Depth, &s.LossType} {
		v, err := r.Int32()
		if err != nil {
			return nil, fmt.Errorf("reading input shape field %d: %w", i, err)
		}
		*dst = int(v)
	}
	return &s, nil
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
func parseFullyConnected(r *Reader, l *Layer) error {
	m, err := parseMatrix(r)
	if err != nil {
		return fmt.Errorf("reading weights: %w", err)
	}
	l.Matrices = append(l.Matrices, m)
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
func parseLSTM(r *Reader, l *Layer) error {
	na, err := r.Int32()
	if err != nil {
		return fmt.Errorf("reading na: %w", err)
	}
	l.NA = int(na)
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
		m, err := parseMatrix(r)
		if err != nil {
			return fmt.Errorf("reading gate weights %d: %w", w, err)
		}
		l.Matrices = append(l.Matrices, m)
		if w == gateCI {
			// ns_ = gate_weights_[CI].NumOutputs(); the 2-D test is exactly
			// Tesseract's, and decides whether a GFS matrix follows.
			ns := m.Rows
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

// parseMatrix mirrors WeightMatrix::DeSerialize for the only configuration
// cadmus supports: modern format (kDoubleFlag), float64 weights (kInt8Flag
// clear), not a training dump. Every other configuration is an error, loudly.
func parseMatrix(r *Reader) (Matrix, error) {
	mode, err := r.Uint8()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading matrix mode: %w", err)
	}
	if mode&modeDoubleFlag == 0 {
		return Matrix{}, fmt.Errorf("matrix mode %d has kDoubleFlag clear, selecting WeightMatrix::DeSerializeOld (float32 weights); cadmus supports only the modern float64 format", mode)
	}
	if mode&modeInt8Flag != 0 {
		return Matrix{}, fmt.Errorf("matrix mode %d has kInt8Flag set: this is an int8-quantized model. Cadmus L1 implements the float weight path only; run ./testdata/fetch.sh to get the tessdata_best model", mode)
	}
	return readFloat64Matrix(r)
}

// readFloat64Matrix mirrors GENERIC_2D_ARRAY<double>::DeSerialize +
// DeSerializeSize (src/ccstruct/matrix.h:197, :567):
//
//	int32   dim1
//	int32   dim2
//	f64     empty_             ; ONE padding element, before the payload
//	f64     array_[dim1*dim2]
//
// empty_ is read and discarded. It is 0.0 in every matrix of tessdata_best
// eng, but nothing in the format requires that, so its value is not asserted.
func readFloat64Matrix(r *Reader) (Matrix, error) {
	dim1, err := r.Int32()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading dim1: %w", err)
	}
	dim2, err := r.Int32()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading dim2: %w", err)
	}
	// Tesseract checks only the UINT16_MAX upper bound; a negative size passes
	// and then misbehaves inside Resize(). Reject it here.
	if dim1 < 0 || dim1 > maxMatrixDim || dim2 < 0 || dim2 > maxMatrixDim {
		return Matrix{}, fmt.Errorf("implausible matrix dimensions %dx%d", dim1, dim2)
	}
	n := int(dim1) * int(dim2)
	if r.Remaining() < 8*(1+n) {
		return Matrix{}, fmt.Errorf("matrix %dx%d needs %d bytes, %d remain", dim1, dim2, 8*(1+n), r.Remaining())
	}
	if _, err := r.Float64(); err != nil { // empty_
		return Matrix{}, fmt.Errorf("reading empty_: %w", err)
	}
	m := Matrix{Rows: int(dim1), Cols: int(dim2), Values: make([]float64, n)}
	for i := range m.Values {
		if m.Values[i], err = r.Float64(); err != nil {
			return Matrix{}, fmt.Errorf("reading element %d of %dx%d: %w", i, dim1, dim2, err)
		}
	}
	return m, nil
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

// gateNames labels an LSTM layer's matrices in serialization order
// (LSTM::WeightType, src/lstm/lstm.h:32). Non-LSTM layers get a bare index.
var gateNames = [...]string{"CI", "GI", "GF1", "GO", "GFS"}

func (l *Layer) tree(w io.Writer, depth int) {
	pad := strings.Repeat("  ", depth)
	_, _ = fmt.Fprintf(w, "%s%v %q ni=%d no=%d", pad, l.Type, l.Name, l.NumInputs, l.NumOutputs)
	if l.NA != 0 {
		_, _ = fmt.Fprintf(w, " na=%d", l.NA)
	}
	_, _ = fmt.Fprintf(w, " weights=%d\n", l.NumWeights)

	switch l.Type {
	case LayerInput:
		if s := l.Shape; s != nil {
			_, _ = fmt.Fprintf(w, "%s    shape batch=%d height=%d width=%d depth=%d loss=%d\n",
				pad, s.Batch, s.Height, s.Width, s.Depth, s.LossType)
		}
	case LayerConvolve:
		_, _ = fmt.Fprintf(w, "%s    half_x=%d half_y=%d\n", pad, l.HalfX, l.HalfY)
	case LayerMaxpool, LayerReconfig:
		_, _ = fmt.Fprintf(w, "%s    x_scale=%d y_scale=%d\n", pad, l.XScale, l.YScale)
	}

	isLSTM := l.Type == LayerLSTM || l.Type == LayerLSTMSummary ||
		l.Type == LayerLSTMSoftmax || l.Type == LayerLSTMSoftmaxEncoded
	for i := range l.Matrices {
		m := &l.Matrices[i]
		label := ""
		if isLSTM && i < len(gateNames) {
			label = gateNames[i]
		}
		min, max, mean := m.Stats()
		_, _ = fmt.Fprintf(w, "%s    [%d] %-3s %dx%d min=%f max=%f mean=%f\n",
			pad, i, label, m.Rows, m.Cols, min, max, mean)
	}
	for _, c := range l.Children {
		c.tree(w, depth+1)
	}
}
