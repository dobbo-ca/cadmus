// This file is a Go translation of src/lstm/stridemap.cpp and
// src/lstm/stridemap.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// StrideMap maps a 2-D (y, x) position in a feature map to the flat timestep
// index used by Tensor, and back.
//
// Tesseract's StrideMap carries a batch dimension (FD_BATCH) so that several
// differently-sized images can share one NetworkIO, which is why its code is
// full of per-batch valid widths, padding, and ZeroInvalidElements. Cadmus
// recognizes exactly one line image at a time, so the batch dimension is always
// 1 and is elided here: there is one (Height, Width) pair, every position is
// valid, and there is nothing to zero. Raster order is preserved — y is the
// outer dimension and x the inner, exactly as StrideMap::Index::Increment
// carries from FD_DIMSIZE-1 down to 0.
type StrideMap struct {
	Height, Width int
}

// Len is the number of timesteps in the map.
func (s StrideMap) Len() int { return s.Height * s.Width }

// T returns the timestep index of (y, x).
func (s StrideMap) T(y, x int) int { return y*s.Width + x }

// YX inverts T.
func (s StrideMap) YX(t int) (y, x int) { return t / s.Width, t % s.Width }

// Offset returns the timestep dy rows and dx columns from t, and reports
// whether that position is inside the map. It is StrideMap::Index::AddOffset
// applied to FD_HEIGHT and FD_WIDTH in turn.
func (s StrideMap) Offset(t, dy, dx int) (int, bool) {
	y, x := s.YX(t)
	y += dy
	x += dx
	if y < 0 || y >= s.Height || x < 0 || x >= s.Width {
		return 0, false
	}
	return s.T(y, x), true
}

// ScaleXY is StrideMap::ScaleXY. Both dimensions are integer-divided, so a
// partial trailing pooling window is dropped rather than padded.
func (s StrideMap) ScaleXY(xFactor, yFactor int) StrideMap {
	return StrideMap{Height: s.Height / yFactor, Width: s.Width / xFactor}
}

// TransposeXY is StrideMap::TransposeXY.
func (s StrideMap) TransposeXY() StrideMap {
	return StrideMap{Height: s.Width, Width: s.Height}
}

// ReduceWidthTo1 is StrideMap::ReduceWidthTo1, used by NT_LSTM_SUMMARY.
func (s StrideMap) ReduceWidthTo1() StrideMap {
	return StrideMap{Height: s.Height, Width: 1}
}

func (s StrideMap) String() string { return fmt.Sprintf("%dx%d", s.Height, s.Width) }
