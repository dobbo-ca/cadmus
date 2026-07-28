package nn

import (
	"math"
	"testing"
)

func TestTablesAreComplete(t *testing.T) {
	if len(tanhTable) != tableSize || len(logisticTable) != tableSize {
		t.Fatalf("table sizes = %d, %d; want %d each", len(tanhTable), len(logisticTable), tableSize)
	}
	if tanhTable[0] != 0 {
		t.Errorf("tanhTable[0] = %v; want 0", tanhTable[0])
	}
	if logisticTable[0] != 0.5 {
		t.Errorf("logisticTable[0] = %v; want 0.5", logisticTable[0])
	}
	for i := 1; i < tableSize; i++ {
		if tanhTable[i] <= tanhTable[i-1] {
			t.Fatalf("tanhTable is not strictly increasing at %d", i)
		}
		if logisticTable[i] <= logisticTable[i-1] {
			t.Fatalf("logisticTable is not strictly increasing at %d", i)
		}
	}
}

// Exactly on a table stop, the interpolation weight is zero, so the result must
// be the table entry itself.
func TestTanhHitsTableStopsExactly(t *testing.T) {
	for _, i := range []int{1, 7, 255, 1000, 4094} {
		x := float64(i) / tableScale
		if got := Tanh(x); got != tanhTable[i] {
			t.Errorf("Tanh(%v) = %v; want tanhTable[%d] = %v", x, got, i, tanhTable[i])
		}
		if got := Logistic(x); got != logisticTable[i] {
			t.Errorf("Logistic(%v) = %v; want logisticTable[%d] = %v", x, got, i, logisticTable[i])
		}
	}
}

func TestActivationSymmetryAndSaturation(t *testing.T) {
	if Tanh(0) != 0 {
		t.Errorf("Tanh(0) = %v; want 0", Tanh(0))
	}
	if Logistic(0) != 0.5 {
		t.Errorf("Logistic(0) = %v; want 0.5", Logistic(0))
	}
	// index >= tableSize-1 returns exactly 1; the negative branch mirrors.
	const sat = 4095.0 / tableScale
	if Tanh(sat) != 1 {
		t.Errorf("Tanh(%v) = %v; want exactly 1", sat, Tanh(sat))
	}
	if Tanh(-sat) != -1 {
		t.Errorf("Tanh(%v) = %v; want exactly -1", -sat, Tanh(-sat))
	}
	if Logistic(sat) != 1 {
		t.Errorf("Logistic(%v) = %v; want exactly 1", sat, Logistic(sat))
	}
	if Logistic(-sat) != 0 {
		t.Errorf("Logistic(%v) = %v; want exactly 0", -sat, Logistic(-sat))
	}
	if got := Logistic(-1.5); got != 1-Logistic(1.5) {
		t.Errorf("Logistic(-1.5) = %v; want 1-Logistic(1.5) = %v", got, 1-Logistic(1.5))
	}
}

// The whole point of transcribing the tables: our Tanh is measurably NOT
// math.Tanh. If this test starts passing with a tiny bound, someone
// regenerated the tables with Go's libm and the port silently diverged from
// Tesseract's arithmetic.
func TestTanhIsTheInterpolatedTableNotLibm(t *testing.T) {
	var maxDiff float64
	for i := 0; i < 200000; i++ {
		x := float64(i) * 8.0 / 200000
		d := math.Abs(Tanh(x) - math.Tanh(x))
		if d > maxDiff {
			maxDiff = d
		}
	}
	// Piecewise-linear error for tanh at h=1/256 is (h^2/8)*max|f''| ~ 1.5e-6.
	if maxDiff < 1e-7 {
		t.Fatalf("max |Tanh - math.Tanh| = %g; too small — the tables look like they were regenerated with Go's libm instead of transcribed from functions.cpp", maxDiff)
	}
	if maxDiff > 2e-6 {
		t.Fatalf("max |Tanh - math.Tanh| = %g; larger than the 1.5e-6 interpolation bound — the table or the interpolation is wrong", maxDiff)
	}
	t.Logf("max |Tanh - math.Tanh| = %g (expected ~1.5e-6)", maxDiff)
}

func TestSoftmaxInPlace(t *testing.T) {
	v := []float64{1, 2, 3}
	SoftmaxInPlace(v)
	var sum float64
	for _, p := range v {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("softmax sums to %v; want 1", sum)
	}
	if !(v[0] < v[1] && v[1] < v[2]) {
		t.Errorf("softmax did not preserve ordering: %v", v)
	}
	// Max subtraction must make a large constant offset a no-op.
	w := []float64{1001, 1002, 1003}
	SoftmaxInPlace(w)
	for i := range v {
		if math.Abs(v[i]-w[i]) > 1e-15 {
			t.Errorf("softmax is not shift invariant: %v vs %v", v, w)
		}
	}
	// Anything more than kMaxSoftmaxActivation below the max clips to exp(-86),
	// so it is small but never exactly zero.
	z := []float64{0, -1000}
	SoftmaxInPlace(z)
	if z[1] <= 0 {
		t.Errorf("softmax produced a zero probability %v; the -86 clip is missing", z[1])
	}
}
