package tessdata

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadRealRecognizer parses the LSTM component of the fetched fixture,
// skipping the test when the fixture has not been fetched.
func loadRealRecognizer(t *testing.T) *Recognizer {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(TypeLSTM)
	if !ok {
		t.Fatal("eng.traineddata has no lstm component")
	}
	rec, err := ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	return rec
}

func loadRealNetwork(t *testing.T) *Layer { return loadRealRecognizer(t).Network }

// The whole point of the spike: parse the real model's graph.
func TestParseNetworkRealModel(t *testing.T) {
	root := loadRealNetwork(t)

	if root.Type != LayerSeries {
		t.Errorf("root.Type = %v; want %v", root.Type, LayerSeries)
	}
	if len(root.Children) == 0 {
		t.Fatal("root has no children; the graph did not deserialize")
	}

	var b strings.Builder
	root.Tree(&b)
	t.Logf("network graph:\n%s", b.String())

	// The tessdata_best English model is a CRNN: it must contain at least one
	// convolution and at least one LSTM somewhere in the tree.
	var conv, lstmCount int
	var walk func(*Layer)
	walk = func(l *Layer) {
		switch l.Type {
		case LayerConvolve:
			conv++
		case LayerLSTM, LayerLSTMSummary, LayerLSTMSoftmax, LayerLSTMSoftmaxEncoded:
			lstmCount++
		}
		for _, ch := range l.Children {
			walk(ch)
		}
	}
	walk(root)
	if conv == 0 {
		t.Error("no convolution layers found; expected a CRNN")
	}
	if lstmCount == 0 {
		t.Error("no LSTM layers found; expected a CRNN")
	}
}

func TestParseNetworkRejectsAbsurdStackSize(t *testing.T) {
	// A Series header claiming 99999 children must be rejected, matching
	// Plumbing::DeSerialize's 10000 guard.
	var b []byte
	b = append(b, byte(LayerSeries), 0, 0)            // type, training, backprop
	b = append(b, 0, 0, 0, 0)                         // flags
	b = append(b, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) // ni, no, num_weights
	b = append(b, 0, 0, 0, 0)                         // empty name
	b = append(b, 0x9f, 0x86, 0x01, 0x00)             // stack size 99999
	if _, err := ParseRecognizer(b, false); err == nil {
		t.Fatal("ParseRecognizer() with 99999-child stack: want error, got nil")
	}
}

// buildLayerHeader emits the common Network header: type tag, training state,
// backprop flag, flags, ni, no, num_weights, name.
func buildLayerHeader(typ LayerType, training int8, flags, ni, no, numWeights int32, name string) []byte {
	b := []byte{byte(typ), byte(training), 0}
	b = binary.LittleEndian.AppendUint32(b, uint32(flags))
	b = binary.LittleEndian.AppendUint32(b, uint32(ni))
	b = binary.LittleEndian.AppendUint32(b, uint32(no))
	b = binary.LittleEndian.AppendUint32(b, uint32(numWeights))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(name)))
	return append(b, name...)
}

// buildFloatMatrix emits a modern float64 WeightMatrix: mode byte, dim1, dim2,
// the lone empty_ element, then dim1*dim2 values in row-major order.
func buildFloatMatrix(mode uint8, rows, cols int, values []float64) []byte {
	b := []byte{mode}
	b = binary.LittleEndian.AppendUint32(b, uint32(rows))
	b = binary.LittleEndian.AppendUint32(b, uint32(cols))
	b = binary.LittleEndian.AppendUint64(b, math.Float64bits(0)) // empty_
	for _, v := range values {
		b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
	}
	return b
}

func TestParseMatrixReadsValuesRowMajor(t *testing.T) {
	// A 2x3 Tanh layer: 6 weights, num_weights must equal 6.
	vals := []float64{1, 2, 3, 4, 5, 6}
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, vals)...)

	r := NewReader(data)
	l, err := parseLayer(r)
	if err != nil {
		t.Fatalf("parseLayer() error = %v", err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
	if len(l.Matrices) != 1 {
		t.Fatalf("len(Matrices) = %d; want 1", len(l.Matrices))
	}
	m := l.Matrices[0]
	if m.Rows != 2 || m.Cols != 3 {
		t.Fatalf("matrix = %dx%d; want 2x3", m.Rows, m.Cols)
	}
	// Row-major: At(row, col) == Values[row*Cols+col].
	for row := range 2 {
		for col := range 3 {
			want := vals[row*3+col]
			if got := m.At(row, col); got != want {
				t.Errorf("At(%d,%d) = %v; want %v", row, col, got, want)
			}
		}
	}
	min, max, mean := m.Stats()
	if min != 1 || max != 6 || mean != 3.5 {
		t.Errorf("Stats() = %v, %v, %v; want 1, 6, 3.5", min, max, mean)
	}
}

func TestParseMatrixRejectsInt8Mode(t *testing.T) {
	// mode 133 == kDoubleFlag|kAdamFlag|kInt8Flag: Homebrew's tessdata build.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(133, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	_, err := parseLayer(NewReader(data))
	if err == nil {
		t.Fatal("parseLayer() with kInt8Flag: want error, got nil")
	}
	if !strings.Contains(err.Error(), "int8") {
		t.Errorf("error %q does not name int8 as the cause", err)
	}
}

func TestParseMatrixRejectsLegacyFormat(t *testing.T) {
	// mode 4 has kDoubleFlag clear => WeightMatrix::DeSerializeOld, float32.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(4, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	if _, err := parseLayer(NewReader(data)); err == nil {
		t.Fatal("parseLayer() with kDoubleFlag clear: want error, got nil")
	}
}

func TestParseLayerRejectsTrainingDump(t *testing.T) {
	// Training state 1 == TS_ENABLED: the matrices carry updates_ (and, with
	// kAdamFlag, dw_sq_sum_) arrays this parser deliberately does not read.
	data := buildLayerHeader(LayerTanh, 1, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	_, err := parseLayer(NewReader(data))
	if err == nil {
		t.Fatal("parseLayer() with TS_ENABLED: want error, got nil")
	}
	// Match the flag NAME, not the word "training": the trailer field names
	// training_flags and training_iteration also contain it, so "training"
	// alone would be satisfied by an unrelated failure.
	if !strings.Contains(err.Error(), "TS_ENABLED") {
		t.Errorf("error %q does not name the training state as the cause", err)
	}
}

func TestParseLayerRejectsNumWeightsMismatch(t *testing.T) {
	// num_weights claims 99 but the 2x3 matrix holds 6 elements.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 99, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	if _, err := parseLayer(NewReader(data)); err == nil {
		t.Fatal("parseLayer() with a bad num_weights: want error, got nil")
	}
}

func TestParseMatrixRejectsNegativeDimension(t *testing.T) {
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	// A non-constant conversion: uint32(int32(-1)) as a constant expression is
	// a compile-time overflow.
	dim1 := int32(-1)
	m := []byte{132}
	m = binary.LittleEndian.AppendUint32(m, uint32(dim1))
	m = binary.LittleEndian.AppendUint32(m, 3)
	if _, err := parseLayer(NewReader(append(data, m...))); err == nil {
		t.Fatal("parseLayer() with dim1 = -1: want error, got nil")
	}
}

// The real model: every weight value present, in the shapes and ranges
// measured from tessdata_best 4.1.0 eng.traineddata during planning.
func TestParseNetworkRealModelWeights(t *testing.T) {
	root := loadRealNetwork(t)

	byName := map[string]*Layer{}
	var walk func(*Layer)
	walk = func(l *Layer) {
		byName[l.Name] = l
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(root)

	for _, tc := range []struct {
		name       string
		matrices   int
		rows, cols int
		na         int
	}{
		{"ConvNL", 1, 16, 10, 0},
		{"Lfys64", 4, 64, 81, 80},
		{"Lfx96", 4, 96, 161, 160},
		{"Lrx96", 4, 96, 193, 192},
		{"Lfx512", 4, 512, 609, 608},
		{"Output", 1, 111, 513, 0},
	} {
		l, ok := byName[tc.name]
		if !ok {
			t.Errorf("layer %q missing from the parsed tree", tc.name)
			continue
		}
		if len(l.Matrices) != tc.matrices {
			t.Errorf("%s: len(Matrices) = %d; want %d", tc.name, len(l.Matrices), tc.matrices)
			continue
		}
		if l.NA != tc.na {
			t.Errorf("%s: NA = %d; want %d", tc.name, l.NA, tc.na)
		}
		for i, m := range l.Matrices {
			if m.Rows != tc.rows || m.Cols != tc.cols {
				t.Errorf("%s matrix %d = %dx%d; want %dx%d", tc.name, i, m.Rows, m.Cols, tc.rows, tc.cols)
			}
			if len(m.Values) != tc.rows*tc.cols {
				t.Errorf("%s matrix %d: len(Values) = %d; want %d", tc.name, i, len(m.Values), tc.rows*tc.cols)
			}
			min, max, mean := m.Stats()
			if math.IsNaN(min) || math.IsInf(min, 0) || math.IsNaN(max) || math.IsInf(max, 0) {
				t.Errorf("%s matrix %d: non-finite bounds min=%v max=%v", tc.name, i, min, max)
			}
			// Every matrix in this model straddles zero and stays well inside
			// +/-100. A matrix of zeros, or one full of huge values, means the
			// payload read is misaligned.
			if !(min < 0 && max > 0) {
				t.Errorf("%s matrix %d: range [%v, %v] does not straddle zero", tc.name, i, min, max)
			}
			if min < -100 || max > 100 {
				t.Errorf("%s matrix %d: range [%v, %v] outside the plausible +/-100", tc.name, i, min, max)
			}
			if math.Abs(mean) > 1 {
				t.Errorf("%s matrix %d: mean %v is implausibly large", tc.name, i, mean)
			}
		}
	}

	// Type-specific scalars.
	wantShape := InputShape{Batch: 1, Height: 36, Width: 0, Depth: 1, LossType: 0}
	if in := byName["Input"]; in == nil {
		t.Error("Input layer missing")
	} else if in.Shape == nil {
		t.Error("Input.Shape = nil; want the StaticShape")
	} else if *in.Shape != wantShape {
		t.Errorf("Input.Shape = %+v; want %+v", *in.Shape, wantShape)
	}
	if c := byName["Convolve"]; c == nil {
		t.Error("Convolve layer missing")
	} else if c.HalfX != 1 || c.HalfY != 1 {
		t.Errorf("Convolve half = %d,%d; want 1,1", c.HalfX, c.HalfY)
	}
	if mp := byName["Maxpool"]; mp == nil {
		t.Error("Maxpool layer missing")
	} else if mp.XScale != 3 || mp.YScale != 3 {
		t.Errorf("Maxpool scale = %d,%d; want 3,3", mp.XScale, mp.YScale)
	}

	// Spot-check one exact statistic against the value measured during
	// planning. Softmax "Output" has the widest range in the model; if the
	// payload were shifted by even one float64 this would not match.
	out := byName["Output"]
	min, max, mean := out.Matrices[0].Stats()
	assertClose(t, "Output min", min, -29.279424)
	assertClose(t, "Output max", max, 35.155323)
	assertClose(t, "Output mean", mean, 0.051364)
}

func assertClose(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 5e-6 {
		t.Errorf("%s = %.6f; want %.6f", what, got, want)
	}
}

// Exact trailer values, read out of tessdata_best 4.1.0 eng.traineddata during
// planning and independently confirmed by `combine_tessdata -l`.
func TestParseRecognizerRealModelTrailer(t *testing.T) {
	rec := loadRealRecognizer(t)

	const wantSpec = "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"
	if rec.NetworkStr != wantSpec {
		t.Errorf("NetworkStr = %q; want %q", rec.NetworkStr, wantSpec)
	}
	if rec.TrainingFlags != 64 {
		t.Errorf("TrainingFlags = %d; want 64 (TF_COMPRESS_UNICHARSET, TF_INT_MODE clear)", rec.TrainingFlags)
	}
	if rec.TrainingIteration != 814100 {
		t.Errorf("TrainingIteration = %d; want 814100", rec.TrainingIteration)
	}
	if rec.SampleIteration != 814136 {
		t.Errorf("SampleIteration = %d; want 814136", rec.SampleIteration)
	}
	if rec.NullChar != 110 {
		t.Errorf("NullChar = %d; want 110", rec.NullChar)
	}
	if rec.AdamBeta != 0.999 {
		t.Errorf("AdamBeta = %v; want 0.999", rec.AdamBeta)
	}
	if rec.LearningRate != 0.001 {
		t.Errorf("LearningRate = %v; want 0.001", rec.LearningRate)
	}
	if rec.Momentum != 0.5 {
		t.Errorf("Momentum = %v; want 0.5", rec.Momentum)
	}
	// The spec string is a placeholder for the output count: ParseOutput in
	// src/training/common/networkbuilder.cpp overrides whatever number appears
	// after O1c with the recoder's code range, which is why the persisted
	// string reads O1c1. Never derive the output count from it.
	if rec.Network.NumOutputs != 111 {
		t.Errorf("Network.NumOutputs = %d; want 111", rec.Network.NumOutputs)
	}
}

// buildTrailer emits the eight LSTMRecognizer fields that follow the root layer.
func buildTrailer(spec string, flags, trainIter, sampleIter, nullChar int32) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(len(spec)))
	b = append(b, spec...)
	for _, v := range []int32{flags, trainIter, sampleIter, nullChar} {
		b = binary.LittleEndian.AppendUint32(b, uint32(v))
	}
	for _, v := range []float32{0.999, 0.001, 0.5} {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}

// A minimal single-layer "network" plus a trailer, for the flag assertions.
//
// null_char MUST be < the layer's no (2 here), or ParseRecognizer's range check
// rejects the fixture and TestParseRecognizerAcceptsTiny can never pass. The
// real model's 110 belongs to a 111-output network, not to this one.
func buildTinyRecognizer(flags int32) []byte {
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	return append(data, buildTrailer("[tiny]", flags, 1, 2, 1)...)
}

func TestParseRecognizerRejectsIntMode(t *testing.T) {
	// training_flags 65 == TF_COMPRESS_UNICHARSET|TF_INT_MODE: Homebrew's
	// tessdata build of eng.
	_, err := ParseRecognizer(buildTinyRecognizer(65), false)
	if err == nil {
		t.Fatal("ParseRecognizer() with TF_INT_MODE: want error, got nil")
	}
	// Match the flag NAME. "int" alone is satisfied by "point", "printing" or
	// any int-typed field name, so it cannot distinguish this cause from an
	// unrelated failure.
	if !strings.Contains(err.Error(), "TF_INT_MODE") {
		t.Errorf("error %q does not name int mode as the cause", err)
	}
}

func TestParseRecognizerRejectsNonRecodingModel(t *testing.T) {
	// TF_COMPRESS_UNICHARSET clear means LoadRecoder builds a pass-through
	// recoder from the unicharset instead of reading the lstm-recoder
	// component. No stock model does this and cadmus does not implement it.
	if _, err := ParseRecognizer(buildTinyRecognizer(0), false); err == nil {
		t.Fatal("ParseRecognizer() with TF_COMPRESS_UNICHARSET clear: want error, got nil")
	}
}

func TestParseRecognizerRejectsTrailingBytes(t *testing.T) {
	data := append(buildTinyRecognizer(64), 0xde, 0xad)
	if _, err := ParseRecognizer(data, false); err == nil {
		t.Fatal("ParseRecognizer() with 2 trailing bytes: want error, got nil")
	}
}

func TestParseRecognizerAcceptsTiny(t *testing.T) {
	rec, err := ParseRecognizer(buildTinyRecognizer(64), false)
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	if rec.NetworkStr != "[tiny]" || rec.NullChar != 1 {
		t.Fatalf("trailer = %q, null %d; want \"[tiny]\", 1", rec.NetworkStr, rec.NullChar)
	}
}
