// This file is a Go translation of src/lstm/networkio.cpp and
// src/lstm/networkio.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

// Tensor is Tesseract's NetworkIO on the float path: a Height x Width grid of
// Features-long activation vectors addressed by flat timestep.
//
// The backing store is float32 even though every arithmetic register in the
// network is float64. That is deliberate and load-bearing: NetworkIO stores
// GENERIC_2D_ARRAY<float>, WriteTimeStep narrows with static_cast<float>, and
// ReadTimeStep widens back. The LSTM's recurrent output passes through a
// NetworkIO (source_) on every timestep, so the narrowing sits inside the
// recurrence. Widening this to float64 diverges from Tesseract by ~1e-7 per
// timestep and compounds through four stacked LSTMs.
type Tensor struct {
	Map      StrideMap
	Features int
	data     []float32
}

func NewTensor(m StrideMap, features int) *Tensor {
	return &Tensor{Map: m, Features: features, data: make([]float32, m.Len()*features)}
}

// Row returns the activation vector at timestep t, aliasing the backing store.
func (x *Tensor) Row(t int) []float32 {
	return x.data[t*x.Features : (t+1)*x.Features]
}

// ReadTimeStep widens timestep t into dst, which must have length Features.
func (x *Tensor) ReadTimeStep(t int, dst []float64) {
	for i, v := range x.Row(t) {
		dst[i] = float64(v)
	}
}

// WriteTimeStep narrows the first Features values of src into timestep t.
func (x *Tensor) WriteTimeStep(t int, src []float64) {
	row := x.Row(t)
	for i := range row {
		row[i] = float32(src[i])
	}
}

// WriteTimeStepPart narrows the first n values of src into timestep t starting
// at feature index offset.
func (x *Tensor) WriteTimeStepPart(t, offset, n int, src []float64) {
	row := x.Row(t)[offset : offset+n]
	for i := range row {
		row[i] = float32(src[i])
	}
}

// CopyTimeStep copies a whole timestep from src. Feature counts must match.
func (x *Tensor) CopyTimeStep(t int, src *Tensor, srcT int) {
	copy(x.Row(t), src.Row(srcT))
}

// CopyTimeStepPart is NetworkIO::CopyTimeStepGeneral: it copies n features from
// src's timestep srcT starting at srcOffset into timestep t starting at offset.
func (x *Tensor) CopyTimeStepPart(t, offset, n int, src *Tensor, srcT, srcOffset int) {
	copy(x.Row(t)[offset:offset+n], src.Row(srcT)[srcOffset:srcOffset+n])
}

// MaxpoolTimeStep takes the elementwise maximum of timestep t and src's
// timestep srcT, in place. Tesseract also records which source timestep won,
// for the backward pass; L1b is inference only, so that is omitted.
func (x *Tensor) MaxpoolTimeStep(t int, src *Tensor, srcT int) {
	dst := x.Row(t)
	for i, v := range src.Row(srcT) {
		if dst[i] < v {
			dst[i] = v
		}
	}
}
