// This file is a Go translation of the TRand class in src/ccutil/helpers.h and
// LSTMRecognizer::SetRandomSeed in src/lstm/lstmrecognizer.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

// Rand is Tesseract's TRand: a linear congruential generator with Knuth's
// constants, used to fill the off-image taps of a convolution.
//
// This is not a general-purpose RNG and must not be replaced with math/rand.
// The draw sequence is part of the network's output: Convolve consumes it in a
// position-dependent order at every image border, so a different generator
// changes the activations along the entire edge of the feature map.
type Rand struct {
	seed uint64
}

// NewRand seeds the generator and discards one iterate, reproducing
// LSTMRecognizer::SetRandomSeed, which calls set_seed followed by IntRand.
// The caller passes sample_iteration * 0x10000001, from the LSTM trailer.
func NewRand(seed uint64) *Rand {
	r := &Rand{seed: seed}
	r.IntRand()
	return r
}

// IntRand steps the generator and returns a value in [0, math.MaxInt32].
func (r *Rand) IntRand() int32 {
	r.seed = r.seed*6364136223846793005 + 1442695040888963407
	return int32(r.seed >> 33)
}

// SignedRand returns a value in [-rng, rng] — closed at both ends, because
// IntRand()'s maximum is exactly INT32_MAX and the division is therefore exactly
// 1 at the top of the range. Tesseract's comment on TRand::SignedRand says the
// same; do not "tighten" this to a half-open interval.
func (r *Rand) SignedRand(rng float64) float64 {
	return rng*2.0*float64(r.IntRand())/2147483647.0 - rng
}
