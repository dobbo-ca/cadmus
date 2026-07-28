// This file is a Go translation of src/lstm/functions.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "math"

const (
	// tableSize and tableScale are kTableSize and kScaleFactor. Table index i
	// corresponds to x = i/256; indices 0..4094 are usable because the
	// interpolation reads index+1.
	tableSize  = 4096
	tableScale = 256.0

	// maxSoftmaxActivation is kMaxSoftmaxActivation: the limit on the negative
	// range of the exp input, which guarantees a non-zero probability.
	maxSoftmaxActivation = 86.0
)

// Tanh is Tesseract's tabulated hyperbolic tangent: linear interpolation
// between 4096 samples at 1/256 spacing, mirrored for negative inputs and
// saturating to exactly 1 at |x| >= 4095/256.
//
// The comparison against tableSize-1 happens before the truncation to an index
// so that a large x cannot overflow the conversion; C++ relies on the
// implementation-defined behaviour of static_cast<unsigned> instead.
func Tanh(x float64) float64 {
	if x < 0 {
		return -Tanh(-x)
	}
	x *= tableScale
	if x >= tableSize-1 {
		return 1
	}
	i := int(x)
	t0, t1 := tanhTable[i], tanhTable[i+1]
	return t0 + (t1-t0)*(x-float64(i))
}

// Logistic is Tesseract's tabulated logistic sigmoid. The negative branch is
// the complement, 1 - Logistic(-x), so it returns exactly 0 at
// x <= -4095/256.
func Logistic(x float64) float64 {
	if x < 0 {
		return 1 - Logistic(-x)
	}
	x *= tableScale
	if x >= tableSize-1 {
		return 1
	}
	i := int(x)
	l0, l1 := logisticTable[i], logisticTable[i+1]
	return l0 + (l1-l0)*(x-float64(i))
}

// TanhInPlace is FuncInplace<GFunc>.
func TanhInPlace(v []float64) {
	for i, x := range v {
		v[i] = Tanh(x)
	}
}

// LogisticInPlace is FuncInplace<FFunc>.
func LogisticInPlace(v []float64) {
	for i, x := range v {
		v[i] = Logistic(x)
	}
}

// SoftmaxInPlace is SoftmaxInPlace from src/lstm/functions.h. Unlike the gate
// activations it is not tabulated: it subtracts the maximum, clips to
// [-86, 0], and calls exp directly. Go's math.Exp and the system libm may
// differ by under one ulp; that is the one divergence from Tesseract this
// package accepts and cannot remove without shipping an exp implementation.
func SoftmaxInPlace(v []float64) {
	if len(v) == 0 {
		return
	}
	maxOut := v[0]
	for _, x := range v[1:] {
		if x > maxOut {
			maxOut = x
		}
	}
	var total float64
	for i, x := range v {
		p := x - maxOut
		if p < -maxSoftmaxActivation {
			p = -maxSoftmaxActivation
		} else if p > 0 {
			p = 0
		}
		e := math.Exp(p)
		total += e
		v[i] = e
	}
	if total > 0 {
		for i := range v {
			v[i] /= total
		}
	}
}
