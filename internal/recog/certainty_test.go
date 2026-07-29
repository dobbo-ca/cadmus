package recog

import (
	"math"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

func TestProbToCertaintyFloor(t *testing.T) {
	if got := ProbToCertainty(1.0); got != 0 {
		t.Errorf("ProbToCertainty(1) = %v; want 0", got)
	}
	if got, want := ProbToCertainty(0.5), math.Log(0.5); got != want {
		t.Errorf("ProbToCertainty(0.5) = %v; want %v", got, want)
	}
	if got := ProbToCertainty(0); got != MinCertainty {
		t.Errorf("ProbToCertainty(0) = %v; want %v", got, MinCertainty)
	}
	if got := ProbToCertainty(math.Exp(MinCertainty) / 2); got != MinCertainty {
		t.Errorf("ProbToCertainty below kMinProb = %v; want the floor %v", got, MinCertainty)
	}
}

// probs builds a synthetic output tensor from an explicit per-timestep
// probability for the winning code, so the certainty arithmetic is checkable
// by hand.
func probs(codes []int, p []float64, n int) *nn.Tensor {
	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: len(codes)}, n)
	row := make([]float64, n)
	for t, c := range codes {
		rest := (1 - p[t]) / float64(n-1)
		for i := range row {
			row[i] = rest
		}
		row[c] = p[t]
		out.WriteTimeStep(t, row)
	}
	return out
}

// The blanks before a character are charged to that character, and its
// certainty is the minimum across the whole span.
//
// Every expectation narrows its probability through float32 first. probs()
// writes via nn.Tensor.WriteTimeStep, which stores float32, and ScoreSymbols
// reads float64(row[...]) back — so the value that reaches ProbToCertainty is
// float64(float32(0.9)) = 0.899999976…, whose log differs from log(0.9) by
// ~2.6e-8, four orders of magnitude above any sane tolerance. Task 9's
// equivalent test already does this; the pattern is not optional.
func TestScoreSymbolsChargesPrecedingBlanks(t *testing.T) {
	const null = 9
	// t0 blank p=0.5, t1 'A' p=0.9, t2 'A' p=0.8, t3 blank p=0.99
	out := probs([]int{null, 1, 1, null}, []float64{0.5, 0.9, 0.8, 0.99}, 10)
	syms := []Symbol{{UnicharID: 1, Text: "A", Start: 1, End: 3}}
	scored := ScoreSymbols(out, syms, 1.0)
	if len(scored) != 1 {
		t.Fatalf("got %d scored symbols; want 1", len(scored))
	}
	cert := func(p float64) float64 { return ProbToCertainty(float64(float32(p))) + CertOffset }
	// Span is t0..t3: the leading blank, both 'A' steps, and the trailing blank
	// folded back onto the last character.
	wantCert := math.Min(math.Min(cert(0.5), cert(0.9)), math.Min(cert(0.8), cert(0.99)))
	if math.Abs(scored[0].Certainty-wantCert) > 1e-12 {
		t.Errorf("Certainty = %v; want %v (the minimum over the whole span)", scored[0].Certainty, wantCert)
	}
	wantRating := -(cert(0.5) + cert(0.9) + cert(0.8) + cert(0.99))
	if math.Abs(scored[0].Rating-wantRating) > 1e-12 {
		t.Errorf("Rating = %v; want %v (minus the sum over the span)", scored[0].Rating, wantRating)
	}
}

// The dict ratio multiplies the certainty of every non-dictionary node, and it
// survives into the reported number.
func TestScoreSymbolsAppliesDictRatio(t *testing.T) {
	out := probs([]int{1}, []float64{0.9}, 10)
	plain := ScoreSymbols(out, []Symbol{{UnicharID: 1, Start: 0, End: 1}}, 1.0)
	scaled := ScoreSymbols(out, []Symbol{{UnicharID: 1, Start: 0, End: 1}}, DictRatio)
	if math.Abs(scaled[0].Certainty-plain[0].Certainty*DictRatio) > 1e-12 {
		t.Errorf("scaled certainty = %v; want %v", scaled[0].Certainty, plain[0].Certainty*DictRatio)
	}
}

// n characters give n+1 boundaries; interior boundaries are the floor of the
// midpoint of the blank gap.
func TestCharBoundaries(t *testing.T) {
	syms := []Scored{
		{Symbol: Symbol{Start: 1, End: 3}},
		{Symbol: Symbol{Start: 6, End: 7}},
	}
	got := CharBoundaries(syms, 10)
	want := []int{0, 4, 10} // 3 + (6-3)/2 = 4
	if len(got) != len(want) {
		t.Fatalf("CharBoundaries() = %v; want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("CharBoundaries() = %v; want %v", got, want)
		}
	}
}

// The edge case the research flagged as reasoned-through but never executed:
// a path with no characters at all, and a path that starts and ends on blanks.
func TestCharBoundariesEdgeCases(t *testing.T) {
	if got := CharBoundaries(nil, 10); len(got) != 1 || got[0] != 10 {
		t.Errorf("CharBoundaries(nil, 10) = %v; want [10]", got)
	}
	syms := []Scored{{Symbol: Symbol{Start: 4, End: 5}}}
	got := CharBoundaries(syms, 10)
	if len(got) != 2 || got[0] != 0 || got[1] != 10 {
		t.Errorf("single symbol boundaries = %v; want [0 10]", got)
	}
}
