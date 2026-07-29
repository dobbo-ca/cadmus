package nn

import "testing"

// The LCG and the >>33 extraction are load-bearing: Convolve's edge padding
// consumes them in a position-dependent order, so a wrong generator shows up
// as noise on the image border and nowhere else.
func TestRandMatchesTesseractLCG(t *testing.T) {
	// Reproduce TRand by hand: seed_ = 5, then two iterations.
	const mul, inc = 6364136223846793005, 1442695040888963407
	seed := uint64(5)
	seed = seed*mul + inc
	first := int32(seed >> 33)
	seed = seed*mul + inc
	second := int32(seed >> 33)

	// NewRand performs the discarded IntRand that SetRandomSeed does, so the
	// first value it hands out is the *second* iterate.
	r := NewRand(5)
	if got := r.IntRand(); got != second {
		t.Fatalf("IntRand() = %d; want %d (NewRand must discard one iterate, first was %d)", got, second, first)
	}
	if got := r.IntRand() < 0; got {
		t.Fatal("IntRand() returned a negative value; seed_>>33 must fit in 31 bits")
	}
}

// The range is CLOSED at both ends. IntRand() returns seed_>>33, whose maximum
// is exactly INT32_MAX, so range*2*INT32_MAX/INT32_MAX - range == +range
// exactly. Tesseract's own comment says "in the range [-range, range]". A
// half-open assertion passes with probability 1 - 2^-31 per draw, which makes it
// a time bomb rather than a test.
func TestSignedRandRange(t *testing.T) {
	r := NewRand(814136 * 0x10000001)
	for range 1000 {
		v := r.SignedRand(1.0)
		if v < -1 || v > 1 {
			t.Fatalf("SignedRand(1.0) = %v; want [-1, 1]", v)
		}
	}
}

// A 1x1 map with ni=1 means every one of the 9 taps except the centre is
// off-image, so 8 of the 9 output features are random draws and only the
// centre carries the input.
func TestConvolveGathersAndRandomizesEdges(t *testing.T) {
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	in.WriteTimeStep(0, []float64{0.5})

	c := NewConvolve("Convolve", 1, 1, 1, NewRand(1))
	if c.NumOutputs() != 9 {
		t.Fatalf("NumOutputs() = %d; want 9", c.NumOutputs())
	}
	out, err := c.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != in.Map {
		t.Fatalf("Convolve changed the map to %v; stride is 1, it must not", out.Map)
	}
	got := make([]float64, 9)
	out.ReadTimeStep(0, got)
	// Feature index of the (dx=0, dy=0) tap is ((0+1)*3 + (0+1))*1 = 4.
	if got[4] != 0.5 {
		t.Errorf("centre tap (feature 4) = %v; want 0.5", got[4])
	}
	for i, v := range got {
		if i == 4 {
			continue
		}
		if v == 0 {
			t.Errorf("feature %d = 0; off-image taps must be randomized, not zeroed", i)
		}
	}
}

// Interior taps must gather the correct neighbours, x-major then y.
func TestConvolveFeatureLayout(t *testing.T) {
	in := NewTensor(StrideMap{Height: 3, Width: 3}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt) + 1})
	}
	out, err := NewConvolve("Convolve", 1, 1, 1, NewRand(1)).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 9)
	out.ReadTimeStep(in.Map.T(1, 1), got) // the fully interior position
	// f = ((dx+1)*3 + (dy+1))*1, source value = T(1+dy, 1+dx) + 1
	for _, tc := range []struct{ dx, dy int }{
		{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1},
	} {
		f := ((tc.dx+1)*3 + (tc.dy + 1))
		want := float64(in.Map.T(1+tc.dy, 1+tc.dx)) + 1
		if got[f] != want {
			t.Errorf("tap (dx=%d,dy=%d) at feature %d = %v; want %v", tc.dx, tc.dy, f, got[f], want)
		}
	}
}
