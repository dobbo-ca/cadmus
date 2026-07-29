package nn

import (
	"math"
	"testing"
)

func TestFullyConnectedAppliesMatrixThenActivation(t *testing.T) {
	// One output, one input: y = tanh(2*x + 0).
	w, err := NewMatrix(1, 1, []float64{2, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	fc := NewFullyConnected("ConvNL", ActTanh, w)
	if fc.NumOutputs() != 1 {
		t.Fatalf("NumOutputs() = %d; want 1", fc.NumOutputs())
	}
	in := NewTensor(StrideMap{Height: 1, Width: 2}, 1)
	in.WriteTimeStep(0, []float64{0.25})
	in.WriteTimeStep(1, []float64{-0.25})
	out, err := fc.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 1)
	out.ReadTimeStep(0, got)
	if want := float64(float32(Tanh(0.5))); got[0] != want {
		t.Errorf("t=0 = %v; want %v", got[0], want)
	}
	out.ReadTimeStep(1, got)
	if want := float64(float32(Tanh(-0.5))); got[0] != want {
		t.Errorf("t=1 = %v; want %v", got[0], want)
	}
	if out.Map != in.Map {
		t.Errorf("FullyConnected changed the map to %v", out.Map)
	}
}

func TestFullyConnectedSoftmaxNormalisesPerTimestep(t *testing.T) {
	w, err := NewMatrix(3, 1, []float64{1, 0, 2, 0, 3, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	in.WriteTimeStep(0, []float64{1})
	out, err := NewFullyConnected("Output", ActSoftmax, w).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 3)
	out.ReadTimeStep(0, got)
	var sum float64
	for _, p := range got {
		sum += p
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("softmax outputs sum to %v; want 1", sum)
	}
}

func TestFullyConnectedRejectsWrongInputWidth(t *testing.T) {
	w, _ := NewMatrix(1, 3, []float64{1, 1, 1, 0})
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 2)
	if _, err := NewFullyConnected("x", ActLinear, w).Forward(in); err == nil {
		t.Fatal("Forward with 2 features into a 3-input matrix: want error, got nil")
	}
}
