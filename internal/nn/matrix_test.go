package nn

import (
	"math"
	"testing"
)

func TestMatrixDotVectorAddsTheBiasColumn(t *testing.T) {
	// Two outputs, three inputs, so each row is 3 weights plus a bias.
	m, err := NewMatrix(2, 3, []float64{
		1, 2, 3, 10,
		0, 0, 1, -5,
	})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	v := make([]float64, 2)
	m.DotVector([]float64{1, 1, 1}, v)
	if v[0] != 16 || v[1] != -4 {
		t.Fatalf("DotVector() = %v; want [16 -4]", v)
	}
}

func TestNewMatrixRejectsWrongWeightCount(t *testing.T) {
	if _, err := NewMatrix(2, 3, make([]float64, 7)); err == nil {
		t.Fatal("NewMatrix with 7 weights for a 2x4 matrix: want error, got nil")
	}
}

// The Go spec allows the compiler to contract a*b+c into a single FMA, across
// statements. That would change the low bits of every dot product in the
// network relative to a scalar C++ build. DotVector must round each product
// before accumulating; this input distinguishes the two.
//
//	term 0: 1.0 * -(1+2^-26)              exact, total = -(1+2^-26)
//	term 1: (1+2^-27) * (1+2^-27)         exact value 1+2^-26+2^-54
//
// Rounded first, term 1 is 1+2^-26 and the total is exactly 0.
// Fused, the exact product is added and the total is 2^-54.
func TestMatrixDotVectorDoesNotFuseMultiplyAdd(t *testing.T) {
	e27 := math.Ldexp(1, -27)
	e26 := math.Ldexp(1, -26)
	m, err := NewMatrix(1, 2, []float64{1, 1 + e27, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	v := make([]float64, 1)
	m.DotVector([]float64{-(1 + e26), 1 + e27}, v)
	if v[0] != 0 {
		t.Fatalf("DotVector() = %g; want exactly 0. The compiler fused the multiply-add; wrap each product in float64(...)", v[0])
	}
}
