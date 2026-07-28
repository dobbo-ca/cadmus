package nn

import "testing"

// stamp is a test layer that writes its own id into every output feature, so
// composition order is observable.
type stamp struct {
	id       float64
	features int
}

func (s *stamp) Name() string    { return "stamp" }
func (s *stamp) NumOutputs() int { return s.features }
func (s *stamp) Forward(in *Tensor) (*Tensor, error) {
	out := NewTensor(in.Map, s.features)
	row := make([]float64, s.features)
	for i := range row {
		row[i] = s.id
	}
	for t := range in.Map.Len() {
		out.WriteTimeStep(t, row)
	}
	return out, nil
}

func TestInputIsIdentity(t *testing.T) {
	in := NewTensor(StrideMap{Height: 2, Width: 2}, 1)
	in.WriteTimeStep(3, []float64{7})
	out, err := NewInput("Input", 1).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 1)
	out.ReadTimeStep(3, got)
	if got[0] != 7 {
		t.Errorf("Input.Forward changed the data: %v", got[0])
	}
}

func TestSeriesRunsInOrder(t *testing.T) {
	s, err := NewSeries("Series", []Layer{&stamp{id: 1, features: 2}, &stamp{id: 2, features: 3}})
	if err != nil {
		t.Fatalf("NewSeries() error = %v", err)
	}
	out, err := s.Forward(NewTensor(StrideMap{Height: 1, Width: 1}, 1))
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Features != 3 {
		t.Fatalf("Series output features = %d; want 3 (the last layer's)", out.Features)
	}
	got := make([]float64, 3)
	out.ReadTimeStep(0, got)
	for _, v := range got {
		if v != 2 {
			t.Fatalf("Series output = %v; want all 2 (the last layer ran last)", got)
		}
	}
}

func TestParallelConcatenatesFeatures(t *testing.T) {
	p, err := NewParallel("Parallel", []Layer{&stamp{id: 1, features: 2}, &stamp{id: 2, features: 3}})
	if err != nil {
		t.Fatalf("NewParallel() error = %v", err)
	}
	if p.NumOutputs() != 5 {
		t.Fatalf("Parallel.NumOutputs() = %d; want 5", p.NumOutputs())
	}
	out, err := p.Forward(NewTensor(StrideMap{Height: 1, Width: 1}, 1))
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 5)
	out.ReadTimeStep(0, got)
	want := []float64{1, 1, 2, 2, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Parallel output = %v; want %v", got, want)
		}
	}
}

// A Reversed wrapping an identity child must be a no-op overall, because the
// transform is applied to the input and again to the output.
func TestReversedIsANoOpAroundIdentity(t *testing.T) {
	for _, kind := range []ReversalKind{ReverseX, ReverseY, TransposeXY} {
		in := NewTensor(StrideMap{Height: 3, Width: 4}, 1)
		for tt := range in.Map.Len() {
			in.WriteTimeStep(tt, []float64{float64(tt)})
		}
		out, err := NewReversed("rev", kind, NewInput("id", 1)).Forward(in)
		if err != nil {
			t.Fatalf("kind %d: Forward() error = %v", kind, err)
		}
		if out.Map != in.Map {
			t.Fatalf("kind %d: output map = %v; want %v", kind, out.Map, in.Map)
		}
		got := make([]float64, 1)
		for tt := range in.Map.Len() {
			out.ReadTimeStep(tt, got)
			if got[0] != float64(tt) {
				t.Fatalf("kind %d: t=%d = %v; want %v", kind, tt, got[0], float64(tt))
			}
		}
	}
}

// The transform itself must be the documented one: RTLReversed mirrors x
// within each row, XYTranspose swaps the axes.
func TestReverseDataTransforms(t *testing.T) {
	src := NewTensor(StrideMap{Height: 2, Width: 3}, 1)
	for tt := range src.Map.Len() {
		src.WriteTimeStep(tt, []float64{float64(tt)})
	}
	got := make([]float64, 1)

	x := reverseData(src, ReverseX)
	x.ReadTimeStep(x.Map.T(0, 0), got)
	if got[0] != 2 {
		t.Errorf("ReverseX dst(0,0) = %v; want 2 (src(0,2))", got[0])
	}

	tr := reverseData(src, TransposeXY)
	if tr.Map != (StrideMap{Height: 3, Width: 2}) {
		t.Fatalf("TransposeXY map = %v; want {3 2}", tr.Map)
	}
	tr.ReadTimeStep(tr.Map.T(2, 1), got)
	if got[0] != float64(src.Map.T(1, 2)) {
		t.Errorf("TransposeXY dst(2,1) = %v; want src(1,2) = %v", got[0], float64(src.Map.T(1, 2)))
	}
}
