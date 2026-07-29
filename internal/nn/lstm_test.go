package nn

import (
	"math"
	"strings"
	"testing"
)

// gates builds four 1x(na+1) matrices with the given input weight, recurrent
// weight and bias, so the cell can be driven analytically.
func gates(t *testing.T, in, rec, bias [4]float64) [4]*Matrix {
	t.Helper()
	var out [4]*Matrix
	for g := range out {
		m, err := NewMatrix(1, 2, []float64{in[g], rec[g], bias[g]})
		if err != nil {
			t.Fatalf("NewMatrix() error = %v", err)
		}
		out[g] = m
	}
	return out
}

func TestLSTMCellMatchesTheEquations(t *testing.T) {
	// Recurrent weights zero, so h_{t-1} does not feed back and each timestep
	// depends only on the input and the carried state.
	g := gates(t,
		[4]float64{1, 1, 1, 1}, // input weights
		[4]float64{0, 0, 0, 0}, // recurrent weights
		[4]float64{0, 0, 0, 0}) // biases
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 2}, 1)
	in.WriteTimeStep(0, []float64{0.5})
	in.WriteTimeStep(1, []float64{0.25})
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Reference, computed with the same tabulated activations.
	state := 0.0
	got := make([]float64, 1)
	for tt, x := range []float64{0.5, 0.25} {
		ci, gi, gf, go_ := Tanh(x), Logistic(x), Logistic(x), Logistic(x)
		state = state*gf + ci*gi
		want := float64(float32(Tanh(state) * go_))
		out.ReadTimeStep(tt, got)
		if math.Abs(got[0]-want) > 1e-9 {
			t.Errorf("t=%d output = %v; want %v", tt, got[0], want)
		}
	}
}

// The state is clipped to +/-100 every timestep, and the clipped value carries
// forward. A huge constant input drives it to the clip and pins it there.
func TestLSTMClipsTheStateEveryTimestep(t *testing.T) {
	g := gates(t,
		[4]float64{100, 100, 100, 100},
		[4]float64{0, 0, 0, 0},
		[4]float64{0, 0, 0, 0})
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 400}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{1})
	}
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	// Tanh(100) saturates to exactly 1 and GO saturates to 1, so a clipped
	// state gives an output of exactly 1. Without the clip the state still
	// saturates the tanh, so instead assert the state directly via the
	// exported probe.
	got := make([]float64, 1)
	out.ReadTimeStep(399, got)
	if got[0] != 1 {
		t.Errorf("saturated output = %v; want exactly 1", got[0])
	}
	// LastState is the state as computed at the final timestep, BEFORE the
	// end-of-row reset. Reading it after the reset would always give 0 and the
	// assertion would be vacuous — the map is one row 400 wide, so the last
	// timestep IS the end of a row.
	if s := l.LastState()[0]; s != 100 {
		t.Errorf("final cell state = %v; want exactly 100 (kStateClip)", s)
	}
}

// Every row is an independent sequence: the state and output are zeroed at the
// end of each row, so two identical rows must produce identical outputs.
func TestLSTMResetsStateAtEndOfRow(t *testing.T) {
	g := gates(t,
		[4]float64{1, 1, 1, 1},
		[4]float64{0.5, 0.5, 0.5, 0.5},
		[4]float64{0, 0, 0, 0})
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 2, Width: 3}, 1)
	for y := range 2 {
		for x := range 3 {
			in.WriteTimeStep(in.Map.T(y, x), []float64{float64(x) * 0.3})
		}
	}
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	a := make([]float64, 1)
	b := make([]float64, 1)
	for x := range 3 {
		out.ReadTimeStep(out.Map.T(0, x), a)
		out.ReadTimeStep(out.Map.T(1, x), b)
		if a[0] != b[0] {
			t.Fatalf("x=%d: row 0 = %v, row 1 = %v; the state was not reset between rows", x, a[0], b[0])
		}
	}
}

// SummLSTM emits only the last h of each row, into a width-1 map.
func TestLSTMSummaryEmitsOnlyTheLastStep(t *testing.T) {
	g := gates(t,
		[4]float64{1, 1, 1, 1},
		[4]float64{0, 0, 0, 0},
		[4]float64{0, 0, 0, 0})
	plain, _ := NewLSTM("L", 1, 2, false, g)
	summ, err := NewLSTM("Lfys", 1, 2, true, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 2, Width: 4}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{0.1 * float64(tt%4+1)})
	}
	full, err := plain.Forward(in)
	if err != nil {
		t.Fatalf("plain Forward() error = %v", err)
	}
	got, err := summ.Forward(in)
	if err != nil {
		t.Fatalf("summary Forward() error = %v", err)
	}
	if got.Map != (StrideMap{Height: 2, Width: 1}) {
		t.Fatalf("summary map = %v; want {2 1}", got.Map)
	}
	a := make([]float64, 1)
	b := make([]float64, 1)
	for y := range 2 {
		full.ReadTimeStep(full.Map.T(y, 3), a)
		got.ReadTimeStep(got.Map.T(y, 0), b)
		if a[0] != b[0] {
			t.Errorf("row %d summary = %v; want the plain LSTM's last step %v", y, b[0], a[0])
		}
	}
}

// gatesNA builds four 1x(na+1) matrices of zeros, so a construction-time shape
// rejection can be provoked at an arbitrary na. Using `gates` here would not
// work: its matrices are always 1x3 (Inputs == 2), so NewLSTM's earlier
// "gate has N input columns, want na=M" check fires first and the shape branch
// under test is never reached.
func gatesNA(t *testing.T, na int) [4]*Matrix {
	t.Helper()
	var out [4]*Matrix
	for g := range out {
		m, err := NewMatrix(1, na, make([]float64, na+1))
		if err != nil {
			t.Fatalf("NewMatrix() error = %v", err)
		}
		out[g] = m
	}
	return out
}

func TestNewLSTMRejectsUnsupportedShapes(t *testing.T) {
	// ns is gate CI's output count, 1 here. For ni=1 a 1-D layer has
	// na == ni+ns == 2; na == ni+2*ns == 3 is the 2-D case.
	if _, err := NewLSTM("L", 1, 3, false, gatesNA(t, 3)); err == nil {
		t.Fatal("NewLSTM with na = ni + 2*ns: want a 2-D unsupported error, got nil")
	} else if !strings.Contains(err.Error(), "2-D") {
		t.Errorf("NewLSTM with na = ni + 2*ns: error = %q; want it to name the 2-D case", err)
	}
	// Anything that is neither ni+ns nor ni+2*ns is softmax feedback.
	if _, err := NewLSTM("L", 1, 5, false, gatesNA(t, 5)); err == nil {
		t.Fatal("NewLSTM with na = 5 (ni=1, ns=1): want a softmax-feedback error, got nil")
	}
	// And the column-count guard still fires when the gates disagree with na.
	if _, err := NewLSTM("L", 1, 2, false, gatesNA(t, 3)); err == nil {
		t.Fatal("NewLSTM with gates wider than na: want an error, got nil")
	}
}
