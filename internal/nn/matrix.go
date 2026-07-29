// This file is a Go translation of src/lstm/weightmatrix.cpp and
// src/ccstruct/matrix.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Matrix is one of Tesseract's WeightMatrix objects on the float path: an
// Outputs x (Inputs+1) row-major array of float64 weights whose last column is
// the bias, applied against an implicit 1.0.
//
// GENERIC_2D_ARRAY's header comment claims column-major storage, but its
// address math is index(column, row) = column*dim2 + row, so w[i][j] is
// array_[i*dim2 + j] — contiguous rows of length dim2, with i the output and j
// the input.
type Matrix struct {
	Outputs int
	Inputs  int
	W       []float64
}

func NewMatrix(outputs, inputs int, w []float64) (*Matrix, error) {
	if outputs <= 0 || inputs < 0 {
		return nil, fmt.Errorf("nn: invalid matrix shape %dx%d", outputs, inputs+1)
	}
	if want := outputs * (inputs + 1); len(w) != want {
		return nil, fmt.Errorf("nn: matrix %dx%d needs %d weights, got %d", outputs, inputs+1, want, len(w))
	}
	return &Matrix{Outputs: outputs, Inputs: inputs, W: w}, nil
}

// DotVector is WeightMatrix::MatrixDotVector: v[i] = sum_j W[i][j]*u[j] plus
// the row's bias, added after the whole dot product exactly as
// MatrixDotVectorInternal does.
//
// Each product is wrapped in an explicit float64 conversion. The Go spec
// permits the compiler to contract a multiply and an add into a single fused
// operation "possibly across statements"; an explicit conversion is the
// documented way to force the intermediate rounding, and without it every dot
// product in the network drifts from a scalar C++ build.
func (m *Matrix) DotVector(u, v []float64) {
	stride := m.Inputs + 1
	for i := range m.Outputs {
		row := m.W[i*stride : (i+1)*stride]
		var total float64
		for j := range m.Inputs {
			total += float64(row[j] * u[j])
		}
		v[i] = total + row[m.Inputs]
	}
}
