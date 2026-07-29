package nn

import "testing"

func TestMaxpoolTakesTheBlockMaximum(t *testing.T) {
	in := NewTensor(StrideMap{Height: 3, Width: 6}, 2)
	// Feature 0 counts up, feature 1 counts down, so the max of each is at a
	// different corner of the window.
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt), float64(in.Map.Len() - tt)})
	}
	mp := NewMaxpool("Maxpool", 2, 3, 3)
	if mp.NumOutputs() != 2 {
		t.Fatalf("Maxpool.NumOutputs() = %d; want 2 (ni, not ni*xs*ys)", mp.NumOutputs())
	}
	out, err := mp.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != (StrideMap{Height: 1, Width: 2}) {
		t.Fatalf("Maxpool map = %v; want {1 2}", out.Map)
	}
	got := make([]float64, 2)
	out.ReadTimeStep(0, got)
	// Window covers y in [0,3), x in [0,3): timesteps 0,1,2,6,7,8,12,13,14.
	if got[0] != 14 {
		t.Errorf("feature 0 = %v; want 14", got[0])
	}
	if got[1] != float64(in.Map.Len()) {
		t.Errorf("feature 1 = %v; want %v", got[1], float64(in.Map.Len()))
	}
}

// Height 36 at Mp3,3 gives 12 rows; a width of 100 gives 33 columns and the
// last column of input is discarded.
func TestMaxpoolDropsPartialWindows(t *testing.T) {
	in := NewTensor(StrideMap{Height: 36, Width: 100}, 1)
	out, err := NewMaxpool("Maxpool", 1, 3, 3).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != (StrideMap{Height: 12, Width: 33}) {
		t.Errorf("Maxpool map = %v; want {12 33}", out.Map)
	}
}

func TestReconfigStacksTheBlock(t *testing.T) {
	in := NewTensor(StrideMap{Height: 2, Width: 2}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt) + 1})
	}
	rc := NewReconfig("Reconfig", 1, 2, 2)
	if rc.NumOutputs() != 4 {
		t.Fatalf("Reconfig.NumOutputs() = %d; want 4", rc.NumOutputs())
	}
	out, err := rc.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 4)
	out.ReadTimeStep(0, got)
	// Feature offset is (x*y_scale + y)*ni, source is T(y, x)+1.
	want := []float64{1, 3, 2, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Reconfig output = %v; want %v", got, want)
		}
	}
}
