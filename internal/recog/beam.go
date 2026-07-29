// This file is a Go translation of src/lstm/recodebeam.cpp and
// src/lstm/recodebeam.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"
	"slices"

	"github.com/dobbo-ca/cadmus/internal/dict"
	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// Transcription notes — what survives of RecodeBeamSearch here, and what the
// single-code recoder restriction deletes.
//
// kNumLengths is RecodedCharID::kMaxCodeLen + 1. NewRecognizer rejects any
// model whose recoder has a code longer than one, so every code is complete on
// its first output and the length dimension collapses to a single value, 0.
// Concretely, in this file:
//
//   - BeamIndex(is_dawg, cont, length) = (is_dawg*NC_COUNT + cont)*kNumLengths
//     + length degenerates to is_dawg*ncCount + cont, so there are 6 heaps per
//     timestep rather than 60.
//   - UnicharCompress::SetupDecoder (src/ccutil/unicharcompress.cpp:396) only
//     ever reaches its `while (len > 0)` loop for codes of length two or more,
//     so next_codes_ is empty and ContinueContext's whole GetNextCodes block is
//     dead. Only GetFinalCodes(empty prefix) is live, and that is the list of
//     every code(0) in unichar-id order with duplicates dropped — finalCodes
//     below is built the same way, in the same order, because the order is the
//     tie-break when two codes reach a full heap with equal scores.
//   - ContinueContext's prefix scan (`for p = length-1; p >= 0; --p`) has an
//     empty range, and its "allow nulls within multi code sequences" push is
//     gated on `length > 0`, so both vanish.
//
// Points that are easy to get wrong and are transcribed deliberately:
//
//   - ComputeTopN forces topNFlags[nullChar] = TN_TOP2 unconditionally, after
//     the heap sweep, so the blank is always available to every sweep tier.
//   - DecodeStep sweeps TN_TOP2, then TN_TOPN, then TN_ALSO_RAN, and stops as
//     soon as any NC_ANYTHING beam of the new step is non-empty.
//   - The min-certainty floor is skipped for nullChar, in both ContinueContext
//     and PushDupOrNoDawgIfBetter.
//   - ContinueDawg only runs when cert > worstDictCert (-25/7).
//   - ExtractBestPaths refuses NC_ONLY_DUP as a line ending, and a dawg
//     hypothesis may only end the line if its last real node has endOfWord set
//     or is a space.
//   - ContinueUnichar stamps NO_PERM on exactly one push — the
//     PushInitialDawgIfBetter for UNICHAR_SPACE — and does NOT scale that
//     node's certainty by dictRatio. Every other non-dawg push is
//     TOP_CHOICE_PERM at cert*dictRatio.
//   - Both UNICHAR_SPACE special cases of ExtractPathAsUnicharIds live here,
//     with opposite gates: the one before the run loop moves a space's
//     accumulated nulls onto the previous character and needs permuter !=
//     NO_PERM; the one inside the run loop makes the space forget those nulls
//     and needs permuter == NO_PERM.
//
// One correction to the plan, from the source: the permuter is NOT threaded
// across ContinueDawg calls. src/lstm/recodebeam.cpp:1101 constructs
// `DawgArgs dawg_args(&initial_dawgs, updated_dawgs, NO_PERM)` afresh on every
// call, and DawgArgs' third constructor parameter is the in/out permuter
// (src/dict/dict.h:84). So each dictionary probe starts from NoPerm and
// def_letter_is_okay's carry rule is inert on the LSTM path. dict.LetterIsOkay
// takes the carried permuter as a parameter, so the threading is available; the
// beam simply does not use it.

const (
	// beamWidth is kBeamWidths[0]: the heap size for completed characters.
	// Tesseract's wider entries apply only to partial multi-code sequences,
	// which L1b's single-code recoder restriction rules out.
	beamWidth = 5
	// worstDictCert is kWorstDictCertainty / kCertaintyScale = -25/7.
	worstDictCert = -25.0 / CertaintyScale
)

// nodeContinuation constrains what may follow a beam node, so that the score
// merges below cannot double-count a timestep's probability.
type nodeContinuation int

const (
	// ncAnything: the node used only its own score.
	ncAnything nodeContinuation = iota
	// ncOnlyDup: the node absorbed a neighbour's probability without a
	// preceding stand-alone, so it must be followed by a duplicate of itself.
	ncOnlyDup
	// ncNoDup: the node combined scores after a stand-alone, so it may not be
	// followed by a duplicate of itself.
	ncNoDup
	ncCount
)

// topNState classifies an output index at one timestep.
type topNState int

const (
	tnTop2 topNState = iota
	tnTopN
	tnAlsoRan
	tnCount
)

// node is RecodeNode. dawgs is the lexicon frontier carried by this hypothesis;
// nil means the hypothesis is not inside a dictionary word, which is what the
// C++'s null DawgPositionVector pointer means.
type node struct {
	code        int
	unicharID   int // -1 for the blank, INVALID_UNICHAR_ID
	permuter    dict.PermuterType
	startOfDawg bool
	startOfWord bool
	endOfWord   bool
	duplicate   bool
	certainty   float64
	score       float64
	prev        *node
	dawgs       []dict.Position
	codeHash    uint64
}

// beamStep is RecodeBeam: one timestep's heaps plus the single best initial
// dawg node per continuation, which is pushed once the step is complete.
type beamStep struct {
	beams   [2 * int(ncCount)][]*node
	initial [ncCount]*node
}

// beamIndex is RecodeBeamSearch::BeamIndex with the length dimension elided.
func beamIndex(isDawg bool, cont nodeContinuation) int {
	if isDawg {
		return int(ncCount) + int(cont)
	}
	return int(cont)
}

// continuationOf is ContinuationFromBeamsIndex.
func continuationOf(index int) nodeContinuation { return nodeContinuation(index % int(ncCount)) }

// isDawgIndex is IsDawgFromBeamsIndex.
func isDawgIndex(index int) bool { return index >= int(ncCount) }

// beamSearch is RecodeBeamSearch. One instance decodes one line.
type beamSearch struct {
	r         *Recognizer
	dict      *dict.Dict
	nullChar  int
	codeRange int
	// finalCodes is GetFinalCodes(empty prefix); unichars maps a code to the
	// unichar it completes, or -1 for the blank.
	finalCodes []int
	unichars   []int
	// spaceDelimited is RecodeBeamSearch::space_delimited_.
	spaceDelimited bool

	steps []*beamStep

	topNFlags  []topNState
	topCode    int
	secondCode int
}

func newBeamSearch(r *Recognizer, numOutputs int) (*beamSearch, error) {
	b := &beamSearch{
		r:              r,
		dict:           r.Dict,
		nullChar:       r.Net.NullChar,
		codeRange:      r.Recoder.CodeRange(),
		unichars:       make([]int, numOutputs),
		topNFlags:      make([]topNState, numOutputs),
		spaceDelimited: true,
		topCode:        -1,
		secondCode:     -1,
	}
	if b.nullChar < 0 || b.nullChar >= numOutputs {
		return nil, fmt.Errorf("recog: null char %d is outside the %d network outputs", b.nullChar, numOutputs)
	}
	for i := range b.unichars {
		b.unichars[i] = -1
	}
	seen := make([]bool, numOutputs)
	for id := range r.Recoder.Size() {
		codes := r.Recoder.Encode(id)
		if len(codes) != 1 {
			return nil, fmt.Errorf("recog: unichar %d encodes to %d codes; the beam supports single-code recoders only", id, len(codes))
		}
		c := int(codes[0])
		if c < 0 || c >= numOutputs {
			return nil, fmt.Errorf("recog: unichar %d encodes to code %d, outside the %d network outputs", id, c, numOutputs)
		}
		if seen[c] {
			continue
		}
		seen[c] = true
		b.finalCodes = append(b.finalCodes, c)
	}
	for _, c := range b.finalCodes {
		if c == b.nullChar {
			continue
		}
		id, ok := r.Recoder.DecodeUnichar([]int32{int32(c)})
		if !ok || id < 0 || id >= r.Charset.Size() {
			return nil, fmt.Errorf("recog: code %d does not decode to a unichar id inside the charset (got %d, ok=%v)", c, id, ok)
		}
		b.unichars[c] = id
	}
	// Dict::IsSpaceDelimitedLang: false as soon as the unicharset carries a
	// script that does not write spaces between words. Only consulted when
	// there is a dictionary, exactly as in the C++ constructor.
	if b.dict != nil {
		for id := range r.Charset.Size() {
			if !r.Charset.IsSpaceDelimited(id) {
				b.spaceDelimited = false
				break
			}
		}
	}
	return b, nil
}

// computeTopN is ComputeTopN: the two highest outputs become tnTop2, the next
// three tnTopN, everything else tnAlsoRan — and the blank is always tnTop2.
func (b *beamSearch) computeTopN(row []float32) {
	for i := range b.topNFlags {
		b.topNFlags[i] = tnAlsoRan
	}
	b.topCode, b.secondCode = -1, -1

	// top holds the beamWidth largest outputs, best first. Tesseract's heap
	// admits a new entry only on a strict improvement over its worst, so an
	// earlier index wins a tie; the insertion point below does the same.
	top := make([]int, 0, beamWidth)
	for i := range row {
		v := row[i]
		pos := 0
		for pos < len(top) && v <= row[top[pos]] {
			pos++
		}
		if pos >= beamWidth {
			continue
		}
		if len(top) < beamWidth {
			top = append(top, 0)
		}
		copy(top[pos+1:], top[pos:])
		top[pos] = i
	}
	for i, c := range top {
		if i < 2 {
			b.topNFlags[c] = tnTop2
		} else {
			b.topNFlags[c] = tnTopN
		}
	}
	if len(top) > 0 {
		b.topCode = top[0]
	}
	if len(top) > 1 {
		b.secondCode = top[1]
	}
	b.topNFlags[b.nullChar] = tnTop2
}

// decodeStep is DecodeStep.
func (b *beamSearch) decodeStep(row []float32, t int) {
	step := b.steps[t]
	if t == 0 {
		// The first step can only use singles and initials.
		b.continueContext(nil, beamIndex(false, ncAnything), row, tnTop2, step)
		if b.dict != nil {
			b.continueContext(nil, beamIndex(true, ncAnything), row, tnTop2, step)
		}
		return
	}
	prev := b.steps[t-1]
	// Work through the scores by group (top-2, top-n, the rest) while the beam
	// is empty: extending with only the top-n may have an empty intersection
	// with the valid codes, in which case the next tier is tried.
	total := 0
	for tn := topNState(0); tn < tnCount && total == 0; tn++ {
		for index := range prev.beams {
			h := prev.beams[index]
			for i := len(h) - 1; i >= 0; i-- {
				b.continueContext(h[i], index, row, tn, step)
			}
		}
		for index := range step.beams {
			if continuationOf(index) == ncAnything {
				total += len(step.beams[index])
			}
		}
	}
	// Special case for the best initial dawg: there is only one per
	// continuation, so pushing it cannot blow up the beam.
	for c := nodeContinuation(0); c < ncCount; c++ {
		if n := step.initial[c]; n != nil {
			pushNode(&step.beams[beamIndex(true, c)], n)
		}
	}
}

// continueContext is ContinueContext, specialized to length 0.
func (b *beamSearch) continueContext(prev *node, index int, row []float32, flag topNState, step *beamStep) {
	useDawgs := isDawgIndex(index)
	prevCont := continuationOf(index)

	if prev != nil {
		if b.topNFlags[prev.code] == flag {
			if prevCont != ncNoDup {
				cert := ProbToCertainty(float64(row[prev.code])) + CertOffset
				b.pushDupOrNoDawg(true, prev.code, prev.unicharID, cert, useDawgs, ncAnything, prev, step)
			}
			if prevCont == ncAnything && flag == tnTop2 && prev.code != b.nullChar {
				cert := ProbToCertainty(float64(row[prev.code])+float64(row[b.nullChar])) + CertOffset
				b.pushDupOrNoDawg(true, prev.code, prev.unicharID, cert, useDawgs, ncNoDup, prev, step)
			}
		}
		if prevCont == ncOnlyDup {
			return
		}
		// The "allow nulls within multi code sequences" push is gated on
		// length > 0, which a single-code recoder never reaches.
	}
	for _, code := range b.finalCodes {
		if b.topNFlags[code] != flag {
			continue
		}
		if prev != nil && prev.code == code {
			continue
		}
		cert := ProbToCertainty(float64(row[code])) + CertOffset
		if cert < MinCertainty && code != b.nullChar {
			continue
		}
		id := b.unichars[code]
		b.continueUnichar(code, id, cert, useDawgs, ncAnything, prev, step)
		if flag == tnTop2 && code != b.nullChar {
			// NXX and NNX both decode to X, so at a class transition the code
			// may claim the blank's probability too — and in the top-2 X->Y
			// case the outgoing class's as well, since XXY, XYY and XNY all
			// decode to XY.
			prob := float64(row[code]) + float64(row[b.nullChar])
			if prev != nil && prevCont == ncAnything && prev.code != b.nullChar &&
				((prev.code == b.topCode && code == b.secondCode) ||
					(code == b.topCode && prev.code == b.secondCode)) {
				prob += float64(row[prev.code])
			}
			cert = ProbToCertainty(prob) + CertOffset
			b.continueUnichar(code, id, cert, useDawgs, ncOnlyDup, prev, step)
		}
	}
}

// continueUnichar is ContinueUnichar: the two-world split.
func (b *beamSearch) continueUnichar(code, unicharID int, cert float64, useDawgs bool, cont nodeContinuation, prev *node, step *beamStep) {
	if useDawgs {
		if cert > worstDictCert {
			b.continueDawg(code, unicharID, cert, cont, prev, step)
		}
		return
	}
	b.pushHeap(&step.beams[beamIndex(false, cont)], code, unicharID, dict.TopChoicePerm,
		false, false, false, false, cert*DictRatio, prev, nil)
	if b.dict == nil {
		return
	}
	if (unicharID == tessdata.UnicharSpace && cert > worstDictCert) || !b.r.Charset.IsSpaceDelimited(unicharID) {
		// Any top choice position that can start a new word — a space, or any
		// non-space-delimited character — is also offered to the dawg search.
		dawgCert := cert
		permuter := dict.TopChoicePerm
		// The space either side of a dictionary word feeds that word's
		// certainty, so a space arriving from a non-dict hypothesis must not be
		// degraded: it keeps its raw certainty and is flagged NO_PERM, which
		// tells ExtractPathAsUnicharIds not to charge it the preceding nulls
		// (those have already been multiplied by the dict ratio and no entry
		// can be inserted into a previous heap after the fact).
		if unicharID == tessdata.UnicharSpace {
			permuter = dict.NoPerm
		} else {
			dawgCert *= DictRatio
		}
		b.pushInitialDawg(code, unicharID, permuter, false, false, dawgCert, cont, prev, step)
	}
}

// continueDawg is ContinueDawg.
func (b *beamSearch) continueDawg(code, unicharID int, cert float64, cont nodeContinuation, prev *node, step *beamStep) {
	dawgHeap := &step.beams[beamIndex(true, cont)]
	noDawgHeap := &step.beams[beamIndex(false, cont)]
	if unicharID < 0 {
		b.pushHeap(dawgHeap, code, unicharID, dict.NoPerm, false, false, false, false, cert, prev, nil)
		return
	}
	// Avoid the dictionary probe if the score is a total loss.
	score := cert
	if prev != nil {
		score += prev.score
	}
	if len(*dawgHeap) >= beamWidth && score <= worstScore(*dawgHeap) &&
		len(*noDawgHeap) >= beamWidth && score <= worstScore(*noDawgHeap) {
		return
	}
	// prev may be the blank or a duplicate, so scan back to the last real one.
	uniPrev := prev
	for uniPrev != nil && (uniPrev.unicharID < 0 || uniPrev.duplicate) {
		uniPrev = uniPrev.prev
	}
	if unicharID == tessdata.UnicharSpace {
		if uniPrev != nil && uniPrev.endOfWord {
			// The space is good: start a fresh dawg on the dawg beam and put a
			// plain space on the top-choice beam.
			b.pushInitialDawg(code, unicharID, uniPrev.permuter, false, false, cert, cont, prev, step)
			b.pushHeap(noDawgHeap, code, unicharID, uniPrev.permuter, false, false, false, false, cert, prev, nil)
		}
		return
	}
	if uniPrev != nil && uniPrev.startOfDawg && uniPrev.unicharID != tessdata.UnicharSpace &&
		b.r.Charset.IsSpaceDelimited(uniPrev.unicharID) && b.r.Charset.IsSpaceDelimited(unicharID) {
		return // Can't break words between space delimited chars.
	}
	var active []dict.Position
	wordStart := false
	switch {
	case uniPrev == nil:
		// Starting from the beginning of the line.
		active = b.dict.Start()
		wordStart = true
	case uniPrev.dawgs != nil:
		// Continuing a previous dict word.
		active = uniPrev.dawgs
		wordStart = uniPrev.startOfDawg
	default:
		return // Can't continue if not a dict word.
	}
	permuter, updated, validEnd := b.dict.LetterIsOkay(dict.NoPerm, active, unicharID, false)
	if permuter == dict.NoPerm {
		return
	}
	b.pushHeap(dawgHeap, code, unicharID, permuter, false, wordStart, validEnd, false, cert, prev, updated)
	if validEnd && !b.spaceDelimited {
		// Another word can start right away, so seed a fresh dawg as well and
		// put the plain character on the top-choice beam, since a non-dict word
		// can start here too.
		b.pushInitialDawg(code, unicharID, permuter, wordStart, true, cert, cont, prev, step)
		b.pushHeap(noDawgHeap, code, unicharID, permuter, false, wordStart, true, false, cert, prev, nil)
	}
}

// pushInitialDawg is PushInitialDawgIfBetter.
func (b *beamSearch) pushInitialDawg(code, unicharID int, permuter dict.PermuterType, start, end bool, cert float64, cont nodeContinuation, prev *node, step *beamStep) {
	score := cert
	if prev != nil {
		score += prev.score
	}
	if best := step.initial[cont]; best != nil && score <= best.score {
		return
	}
	step.initial[cont] = &node{
		code:        code,
		unicharID:   unicharID,
		permuter:    permuter,
		startOfDawg: true,
		startOfWord: start,
		endOfWord:   end,
		certainty:   cert,
		score:       score,
		prev:        prev,
		dawgs:       b.dict.Start(),
		codeHash:    b.codeHashOf(code, false, prev),
	}
}

// pushDupOrNoDawg is PushDupOrNoDawgIfBetter, specialized to length 0.
func (b *beamSearch) pushDupOrNoDawg(dup bool, code, unicharID int, cert float64, useDawgs bool, cont nodeContinuation, prev *node, step *beamStep) {
	if useDawgs {
		if cert > worstDictCert {
			permuter := dict.NoPerm
			if prev != nil {
				permuter = prev.permuter
			}
			b.pushHeap(&step.beams[beamIndex(true, cont)], code, unicharID, permuter,
				false, false, false, dup, cert, prev, nil)
		}
		return
	}
	cert *= DictRatio
	if cert >= MinCertainty || code == b.nullChar {
		permuter := dict.TopChoicePerm
		if prev != nil {
			permuter = prev.permuter
		}
		b.pushHeap(&step.beams[beamIndex(false, cont)], code, unicharID, permuter,
			false, false, false, dup, cert, prev, nil)
	}
}

// pushHeap is PushHeapIfBetter.
func (b *beamSearch) pushHeap(heap *[]*node, code, unicharID int, permuter dict.PermuterType, dawgStart, wordStart, end, dup bool, cert float64, prev *node, dawgs []dict.Position) {
	score := cert
	if prev != nil {
		score += prev.score
	}
	if len(*heap) >= beamWidth && score <= worstScore(*heap) {
		return
	}
	pushNode(heap, &node{
		code:        code,
		unicharID:   unicharID,
		permuter:    permuter,
		startOfDawg: dawgStart,
		startOfWord: wordStart,
		endOfWord:   end,
		duplicate:   dup,
		certainty:   cert,
		score:       score,
		prev:        prev,
		dawgs:       dawgs,
		codeHash:    b.codeHashOf(code, dup, prev),
	})
}

// pushNode is the PushHeapIfBetter overload that takes a ready-made node: it
// dedups on (code, codeHash, permuter, startOfDawg) keeping the better score
// (UpdateHeapIfMatched), then evicts the worst entry once over beamWidth.
//
// Tesseract's GenericHeap keeps the WORST element on top so the overflow pop is
// O(log n); at beamWidth 5 a linear scan is cheaper, and UpdateHeapIfMatched is
// a linear scan in the C++ too.
func pushNode(heap *[]*node, n *node) {
	h := *heap
	if len(h) >= beamWidth && n.score <= worstScore(h) {
		return
	}
	for _, m := range h {
		if m.code == n.code && m.codeHash == n.codeHash &&
			m.permuter == n.permuter && m.startOfDawg == n.startOfDawg {
			if n.score > m.score {
				*m = *n
			}
			return
		}
	}
	h = append(h, n)
	if len(h) > beamWidth {
		worst := 0
		for i := 1; i < len(h); i++ {
			if h[i].score < h[worst].score {
				worst = i
			}
		}
		h[worst] = h[len(h)-1]
		h = h[:len(h)-1]
	}
	*heap = h
}

func worstScore(h []*node) float64 {
	w := h[0].score
	for _, n := range h[1:] {
		if n.score < w {
			w = n.score
		}
	}
	return w
}

// codeHashOf is ComputeCodeHash: a rolling base-codeRange hash that skips
// duplicates and blanks, so it identifies the CTC-collapsed label sequence.
func (b *beamSearch) codeHashOf(code int, dup bool, prev *node) uint64 {
	var hash uint64
	if prev != nil {
		hash = prev.codeHash
	}
	if !dup && code != b.nullChar {
		n := uint64(b.codeRange)
		carry := ((hash >> 32) * n) >> 32
		hash *= n
		hash += carry
		hash += uint64(code)
	}
	return hash
}

// extractBestPath is ExtractBestPaths plus ExtractPath, without the second-best
// path, which nothing in cadmus consumes.
func (b *beamSearch) extractBestPath() []*node {
	last := b.steps[len(b.steps)-1]
	var best *node
	for c := nodeContinuation(0); c < ncCount; c++ {
		if c == ncOnlyDup {
			// An NC_ONLY_DUP node still owes a stand-alone duplicate of itself,
			// so it can never be the last node of a line.
			continue
		}
		for _, isDawg := range []bool{false, true} {
			for _, n := range last.beams[beamIndex(isDawg, c)] {
				if isDawg {
					// The node may be a blank or a duplicate, so scan back to
					// the last real unichar; a dawg hypothesis may only finish
					// the line on a complete word or on a space.
					d := n
					for d != nil && (d.unicharID < 0 || d.duplicate) {
						d = d.prev
					}
					if d == nil || (!d.endOfWord && d.unicharID != tessdata.UnicharSpace) {
						continue
					}
				}
				if best == nil || n.score > best.score {
					best = n
				}
			}
		}
	}
	var path []*node
	for n := best; n != nil; n = n.prev {
		path = append(path, n)
	}
	slices.Reverse(path)
	return path
}

// pathAsScored is ExtractPathAsUnicharIds: it walks the node path once,
// charging the blanks between two characters to the later one.
func (r *Recognizer) pathAsScored(path []*node) []Scored {
	var out []Scored
	width := len(path)
	t := 0
	for t < width {
		certainty := 0.0
		rating := 0.0
		for t < width && path[t].unicharID < 0 {
			c := path[t].certainty
			t++
			if c < certainty {
				certainty = c
			}
			rating -= c
		}
		if t >= width {
			if len(out) > 0 {
				last := &out[len(out)-1]
				if certainty < last.Certainty {
					last.Certainty = certainty
				}
				last.Rating += rating
			}
			continue
		}
		id := path[t].unicharID
		if id == tessdata.UnicharSpace && len(out) > 0 && path[t].permuter != dict.NoPerm {
			// All the rating and certainty go on the previous character except
			// for the space itself.
			last := &out[len(out)-1]
			if certainty < last.Certainty {
				last.Certainty = certainty
			}
			last.Rating += rating
			certainty = 0
			rating = 0
		}
		start := t
		for {
			c := path[t].certainty
			t++
			// A NO_PERM space forgets the certainty of the preceding blanks.
			if c < certainty || (id == tessdata.UnicharSpace && path[t-1].permuter == dict.NoPerm) {
				certainty = c
			}
			rating -= c
			if t >= width || !path[t].duplicate {
				break
			}
		}
		out = append(out, Scored{
			Symbol:    Symbol{UnicharID: id, Text: r.Charset.Text(id), Start: start, End: t},
			Certainty: certainty,
			Rating:    rating,
		})
	}
	return out
}

// BeamDecode is RecodeBeamSearch::Decode followed by ExtractBestPaths and
// ExtractPathAsUnicharIds. It returns the same Scored slice the greedy path
// produces, so Task 20's word segmentation is reused unchanged.
func (r *Recognizer) BeamDecode(out *nn.Tensor) ([]Scored, error) {
	if out.Features != r.Net.NumOutputs {
		return nil, fmt.Errorf("recog: output has %d features, want %d", out.Features, r.Net.NumOutputs)
	}
	width := out.Map.Len()
	if width == 0 {
		return nil, nil
	}
	b, err := newBeamSearch(r, out.Features)
	if err != nil {
		return nil, err
	}
	b.steps = make([]*beamStep, width)
	for t := range width {
		b.steps[t] = &beamStep{}
		row := out.Row(t)
		b.computeTopN(row)
		b.decodeStep(row, t)
	}
	return r.pathAsScored(b.extractBestPath()), nil
}
