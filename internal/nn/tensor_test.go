package nn

import (
	"math"
	"testing"
)

func TestTensorRoundsThroughFloat32(t *testing.T) {
	x := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	// A value that is representable in float64 but not float32. Storing and
	// reloading must lose the low bits, exactly as NetworkIO does.
	const v = 1.0 + 1e-12
	x.WriteTimeStep(0, []float64{v})
	got := make([]float64, 1)
	x.ReadTimeStep(0, got)
	if got[0] == v {
		t.Fatalf("ReadTimeStep returned %v unchanged; the store is not float32", got[0])
	}
	if got[0] != float64(float32(v)) {
		t.Fatalf("ReadTimeStep() = %v; want %v", got[0], float64(float32(v)))
	}
}

func TestTensorWriteTimeStepPart(t *testing.T) {
	x := NewTensor(StrideMap{Height: 1, Width: 2}, 5)
	x.WriteTimeStep(0, []float64{1, 2, 3, 4, 5})
	x.WriteTimeStepPart(0, 2, 2, []float64{9, 8})
	got := make([]float64, 5)
	x.ReadTimeStep(0, got)
	want := []float64{1, 2, 9, 8, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("after WriteTimeStepPart = %v; want %v", got, want)
		}
	}
	// Timestep 1 must be untouched.
	x.ReadTimeStep(1, got)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("timestep 1 feature %d = %v; want 0", i, v)
		}
	}
}

func TestTensorCopyTimeStepPart(t *testing.T) {
	src := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	src.WriteTimeStep(0, []float64{7, 8, 9})
	dst := NewTensor(StrideMap{Height: 1, Width: 1}, 6)
	dst.CopyTimeStepPart(0, 3, 3, src, 0, 0)
	got := make([]float64, 6)
	dst.ReadTimeStep(0, got)
	want := []float64{0, 0, 0, 7, 8, 9}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("CopyTimeStepPart = %v; want %v", got, want)
		}
	}
}

func TestTensorMaxpoolTimeStep(t *testing.T) {
	a := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	a.WriteTimeStep(0, []float64{1, 5, -3})
	b := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	b.WriteTimeStep(0, []float64{4, 2, -1})
	a.MaxpoolTimeStep(0, b, 0)
	got := make([]float64, 3)
	a.ReadTimeStep(0, got)
	want := []float64{4, 5, -1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 0 {
			t.Fatalf("MaxpoolTimeStep = %v; want %v", got, want)
		}
	}
}
