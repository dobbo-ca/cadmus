// This file is a Go translation of NetworkIO::ProbToCertainty in
// src/lstm/networkio.cpp and RecodeBeamSearch::ExtractPathAsUnicharIds and
// ::calculateCharBoundaries in src/lstm/recodebeam.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"math"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

const (
	// MinCertainty is kMinCertainty from src/lstm/networkio.cpp.
	MinCertainty = -20.0
	// CertOffset is kCertOffset from src/lstm/lstmrecognizer.cpp, added to
	// every node's certainty.
	CertOffset = -0.085
	// DictRatio is kDictRatio. Tesseract multiplies the certainty of every
	// non-dictionary hypothesis by it, so a word the lexicon does not know is
	// reported 2.25 times worse than its raw log probability. Greedy decoding
	// has no lexicon, so every character takes it.
	DictRatio = 2.25
	// CertaintyScale is kCertaintyScale from src/ccmain/linerec.cpp, applied
	// once at export.
	CertaintyScale = 7.0
)

var minProb = math.Exp(MinCertainty)

// ProbToCertainty is NetworkIO::ProbToCertainty: the log probability, floored.
func ProbToCertainty(p float64) float64 {
	if p > minProb {
		return math.Log(p)
	}
	return MinCertainty
}

// Scored is a Symbol with its CTC certainty and rating attached.
type Scored struct {
	Symbol
	Certainty float64
	Rating    float64
}

// ScoreSymbols attaches per-character certainties and ratings.
//
// The span charged to a character runs from the end of the previous character
// through the end of this one, so the blanks *between* two characters are
// charged to the later one — that is what ExtractPathAsUnicharIds does, and it
// is why a character preceded by a long uncertain gap reports low confidence.
// Trailing blanks after the last character fold back onto it.
func ScoreSymbols(out *nn.Tensor, syms []Symbol, dictRatio float64) []Scored {
	width := out.Map.Len()
	cert := func(t int) float64 {
		row := out.Row(t)
		p := float64(row[argmax(row)])
		return (ProbToCertainty(p) + CertOffset) * dictRatio
	}

	scored := make([]Scored, 0, len(syms))
	prevEnd := 0
	for i, s := range syms {
		end := s.End
		if i == len(syms)-1 {
			// Fold the trailing blanks onto the last character.
			end = width
		}
		c := 0.0
		r := 0.0
		for t := prevEnd; t < end; t++ {
			v := cert(t)
			if v < c {
				c = v
			}
			r -= v
		}
		scored = append(scored, Scored{Symbol: s, Certainty: c, Rating: r})
		prevEnd = s.End
	}
	return scored
}

// CharBoundaries is calculateCharBoundaries: for n >= 1 characters it returns
// n+1 timestep boundaries, where an interior boundary is the floor of the
// midpoint of the blank gap between two characters, and character i occupies
// [bounds[i], bounds[i+1]).
//
// For n == 0 it returns the single element []int{width}, matching the C++'s
// push(0) / empty loop / pop_back / push(maxWidth) sequence. That is not the
// general rule with n substituted; it is a separate case.
func CharBoundaries(syms []Scored, width int) []int {
	bounds := make([]int, 0, len(syms)+1)
	bounds = append(bounds, 0)
	for i := 1; i < len(syms); i++ {
		end := syms[i-1].End
		bounds = append(bounds, end+(syms[i].Start-end)/2)
	}
	if len(syms) == 0 {
		return []int{width}
	}
	return append(bounds, width)
}
