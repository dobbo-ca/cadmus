// This file is a Go translation of RecodeBeamSearch::ExtractBestPathAsWords and
// ::InitializeWord in src/lstm/recodebeam.cpp, Tesseract::SearchWords in
// src/ccmain/linerec.cpp, and LTRResultIterator::Confidence in
// src/ccmain/ltrresultiterator.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"image"
	"math"
	"strings"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// Word is one recognized word with its geometry and confidence.
type Word struct {
	Text       string
	Bounds     image.Rectangle
	Confidence float64
}

// Line is one recognized line.
type Line struct {
	Text       string
	Words      []Word
	Bounds     image.Rectangle
	Confidence float64
}

// ConfidencePercent is LTRResultIterator::Confidence composed with
// Tesseract::SearchWords' kCertaintyScale: clip(100 + 5*7*certainty, 0, 100).
func ConfidencePercent(certainty float64) float64 {
	c := 100 + 5*CertaintyScale*certainty
	if c < 0 {
		return 0
	}
	if c > 100 {
		return 100
	}
	return c
}

// wordGroup is a run of symbols between spaces.
type wordGroup struct {
	start, end int
	spaceCert  float64
}

// splitWords cuts at UNICHAR_SPACE. Tesseract additionally cuts at a
// start_of_word node and, under TOP_CHOICE_PERM, at every character of a
// non-space-delimited script; the first only exists in the dictionary path
// (Task 23) and the second cannot occur while the recoder is restricted to
// Latin single-code models.
func splitWords(syms []Scored) []wordGroup {
	var groups []wordGroup
	i := 0
	prevSpaceCert := 0.0
	for i < len(syms) {
		if syms[i].UnicharID == tessdata.UnicharSpace {
			prevSpaceCert = syms[i].Certainty
			i++
			continue
		}
		start := i
		for i < len(syms) && syms[i].UnicharID != tessdata.UnicharSpace {
			i++
		}
		// The word's space certainty is the lesser of the spaces flanking it.
		spaceCert := prevSpaceCert
		if i < len(syms) && syms[i].Certainty < spaceCert {
			spaceCert = syms[i].Certainty
		}
		groups = append(groups, wordGroup{start: start, end: i, spaceCert: spaceCert})
	}
	return groups
}

// wordBounds converts a timestep span to page coordinates. Y is the line
// strip's own extent: InitializeWord gives every word the full line height,
// and nothing in the network output constrains it further.
func wordBounds(bounds []int, start, end int, scale float64, lineBox image.Rectangle) image.Rectangle {
	left := lineBox.Min.X + int(math.Floor(float64(bounds[start])*scale))
	right := lineBox.Min.X + int(math.Ceil(float64(bounds[end])*scale))
	if right <= left {
		right = left + 1
	}
	return image.Rect(left, lineBox.Min.Y, right, lineBox.Max.Y)
}

// Recognize transcribes one cropped line image.
func (r *Recognizer) Recognize(img image.Image) (Line, error) {
	out, norm, err := r.Forward(img)
	if err != nil {
		return Line{}, err
	}
	// BeamDecode already carries a certainty and a rating per character: the
	// beam's node scores are what the dictionary weighting acts on, so scoring
	// the timesteps a second time here would throw that weighting away.
	scored, err := r.BeamDecode(out)
	if err != nil {
		return Line{}, err
	}
	bounds := CharBoundaries(scored, out.Map.Len())

	lineBox := img.Bounds()
	line := Line{Bounds: lineBox, Confidence: 100}

	var text strings.Builder
	for _, s := range scored {
		text.WriteString(s.Text)
	}
	line.Text = text.String()

	for _, g := range splitWords(scored) {
		cert := 0.0
		var wt strings.Builder
		for i := g.start; i < g.end; i++ {
			wt.WriteString(scored[i].Text)
			if scored[i].Certainty < cert {
				cert = scored[i].Certainty
			}
		}
		if g.spaceCert < cert {
			cert = g.spaceCert
		}
		line.Words = append(line.Words, Word{
			Text:       wt.String(),
			Bounds:     wordBounds(bounds, g.start, g.end, norm.ScaleFactor, lineBox),
			Confidence: ConfidencePercent(cert),
		})
		if c := ConfidencePercent(cert); c < line.Confidence {
			line.Confidence = c
		}
	}
	if len(line.Words) == 0 {
		line.Confidence = 0
	}
	return line, nil
}
