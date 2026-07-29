// This file is a Go translation of Dict::def_letter_is_okay,
// Dict::default_dawgs, Dict::GetStartingNode and Dict::char_for_dawg in
// src/dict/dict.cpp and src/dict/dict.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package dict

import "github.com/dobbo-ca/cadmus/internal/tessdata"

// DawgType is which lexicon a dawg is, which decides how its unichar ids are
// folded before lookup and which other dawgs may follow it. It lives here
// rather than in internal/tessdata because nothing in the loader needs it.
//
// DAWG_TYPE_PATTERN has no value here on purpose. def_letter_is_okay has a
// branch for it (ProcessPatternEdges), pattern dawgs come from `user-patterns`
// through a separate API, and no .traineddata LSTM component is one — so
// declaring the value without the branch would advertise support that does not
// exist. New() takes exactly three dawgs and assigns these types itself, so an
// unhandled type is unreachable by construction.
type DawgType int

const (
	DawgPunctuation DawgType = iota
	DawgWord
	DawgNumber
)

// PermuterType is Tesseract's PermuterType, restricted to the values the LSTM
// path can produce. Larger is better; def_letter_is_okay takes the maximum over
// the surviving positions and then applies the carry rule in LetterIsOkay.
type PermuterType int

const (
	NoPerm         PermuterType = 0
	PuncPerm       PermuterType = 1
	NumberPerm     PermuterType = 6
	SystemDawgPerm PermuterType = 8
	// TopChoicePerm is what the beam stamps on a non-dictionary hypothesis.
	TopChoicePerm PermuterType = 2
)

// Position is DawgPosition: a point in the lexicon frontier. DawgIndex or
// PuncIndex of -1 means "not in that dawg", and a ref of NoEdge means "at the
// start of it".
type Position struct {
	DawgIndex  int
	DawgRef    int64
	PuncIndex  int
	PuncRef    int64
	BackToPunc bool
}

// Dict is the LSTM lexicon: the punctuation dawg wrapping the word and number
// dawgs, exactly as Dict::LoadLSTM assembles it.
type Dict struct {
	charset *tessdata.Unicharset
	dawgs   []*tessdata.Dawg // index 0 punctuation, 1 word, 2 number
	types   []DawgType
	perms   []PermuterType
}

const (
	puncIdx   = 0
	wordIdx   = 1
	numberIdx = 2
)

func New(charset *tessdata.Unicharset, punc, word, number *tessdata.Dawg) *Dict {
	return &Dict{
		charset: charset,
		dawgs:   []*tessdata.Dawg{punc, word, number},
		types:   []DawgType{DawgPunctuation, DawgWord, DawgNumber},
		perms:   []PermuterType{PuncPerm, SystemDawgPerm, NumberPerm},
	}
}

// Start is Dict::default_dawgs. The punctuation dawg seeds a position on its
// own; the word and number dawgs are subsumed by it whenever the punctuation
// dawg has a kPatternUnicharID edge out of its root, which every stock eng
// model does.
func (d *Dict) Start() []Position {
	punc := d.dawgs[puncIdx]
	puncAvailable := punc != nil && punc.EdgeChar(0, tessdata.PatternUnicharID, true) != tessdata.NoEdge

	var out []Position
	for i, dw := range d.dawgs {
		if dw == nil {
			continue
		}
		if i == puncIdx {
			out = append(out, Position{DawgIndex: -1, DawgRef: tessdata.NoEdge, PuncIndex: i, PuncRef: tessdata.NoEdge})
			continue
		}
		// Word and number are successors of punctuation, so they are only
		// seeded independently when punctuation cannot reach them.
		if !puncAvailable {
			out = append(out, Position{DawgIndex: i, DawgRef: tessdata.NoEdge, PuncIndex: -1, PuncRef: tessdata.NoEdge})
		}
	}
	return out
}

// LetterIsOkay is Dict::def_letter_is_okay.
//
// prev is the caller's carried permuter — def_letter_is_okay's in/out
// `dawg_args->permuter`. Pass NoPerm for the first letter of a word and the
// previous return value thereafter.
func (d *Dict) LetterIsOkay(prev PermuterType, active []Position, unicharID int, wordEnd bool) (PermuterType, []Position, bool) {
	// A word may never contain the pattern wildcard; accepting it would let a
	// pattern dawg match arbitrary text. This is before the carry block in the
	// C++ too, and it sets dawg_args->permuter = NO_PERM outright.
	if unicharID == tessdata.PatternUnicharID || unicharID < 0 || unicharID >= d.charset.Size() {
		return NoPerm, nil, false
	}

	curr := NoPerm
	var updated []Position
	validEnd := false

	add := func(p Position) {
		for _, q := range updated {
			if q == p {
				return
			}
		}
		updated = append(updated, p)
	}

	for _, pos := range active {
		var punc, dw *tessdata.Dawg
		if pos.PuncIndex >= 0 {
			punc = d.dawgs[pos.PuncIndex]
		}
		if pos.DawgIndex >= 0 {
			dw = d.dawgs[pos.DawgIndex]
		}
		if punc == nil && dw == nil {
			continue
		}

		if dw == nil {
			// In the punctuation dawg with no core dawg chosen yet.
			puncNode := startingNode(punc, pos.PuncRef)
			if trans := punc.EdgeChar(puncNode, tessdata.PatternUnicharID, wordEnd); trans != tessdata.NoEdge {
				for _, si := range d.successors(pos.PuncIndex) {
					sd := d.dawgs[si]
					ch := d.charForDawg(unicharID, si)
					if e := sd.EdgeChar(0, ch, wordEnd); e != tessdata.NoEdge {
						add(Position{DawgIndex: si, DawgRef: e, PuncIndex: pos.PuncIndex, PuncRef: trans})
						if d.perms[si] > curr {
							curr = d.perms[si]
						}
						if sd.EndOfWord(e) && punc.EndOfWord(trans) {
							validEnd = true
						}
					}
				}
			}
			if e := punc.EdgeChar(puncNode, unicharID, wordEnd); e != tessdata.NoEdge {
				add(Position{DawgIndex: -1, DawgRef: tessdata.NoEdge, PuncIndex: pos.PuncIndex, PuncRef: e})
				if PuncPerm > curr {
					curr = PuncPerm
				}
				if punc.EndOfWord(e) {
					validEnd = true
				}
			}
			continue
		}

		if punc != nil && pos.DawgRef != tessdata.NoEdge && dw.EndOfWord(pos.DawgRef) {
			// The core word can end here; see whether punctuation continues.
			puncNode := startingNode(punc, pos.PuncRef)
			if e := punc.EdgeChar(puncNode, unicharID, wordEnd); e != tessdata.NoEdge {
				add(Position{DawgIndex: pos.DawgIndex, DawgRef: pos.DawgRef, PuncIndex: pos.PuncIndex, PuncRef: e, BackToPunc: true})
				if d.perms[pos.DawgIndex] > curr {
					curr = d.perms[pos.DawgIndex]
				}
				if punc.EndOfWord(e) {
					validEnd = true
				}
			}
		}

		if pos.BackToPunc {
			continue
		}

		// DAWG_TYPE_PATTERN would be handled here, before the edge lookup.
		// L1b declares no such type; see Step 1 branch 7.

		node := startingNode(dw, pos.DawgRef)
		e := tessdata.NoEdge
		if node != tessdata.NoEdge {
			e = dw.EdgeChar(node, d.charForDawg(unicharID, pos.DawgIndex), wordEnd)
		}
		if e == tessdata.NoEdge {
			continue
		}
		if wordEnd && punc != nil && pos.PuncRef != tessdata.NoEdge && !punc.EndOfWord(pos.PuncRef) {
			// The punctuation constraint is not satisfied at the end of a word.
			continue
		}
		if d.perms[pos.DawgIndex] > curr {
			curr = d.perms[pos.DawgIndex]
		}
		if dw.EndOfWord(e) && (punc == nil || pos.PuncRef == tessdata.NoEdge || punc.EndOfWord(pos.PuncRef)) {
			validEnd = true
		}
		add(Position{DawgIndex: pos.DawgIndex, DawgRef: e, PuncIndex: pos.PuncIndex, PuncRef: pos.PuncRef})
	}

	// The permuter carry, verbatim from the tail of def_letter_is_okay:
	//
	//	if (dawg_args->permuter == NO_PERM || curr_perm == NO_PERM ||
	//	    (curr_perm != PUNC_PERM && dawg_args->permuter != COMPOUND_PERM)) {
	//	  dawg_args->permuter = curr_perm;
	//	}
	//	return dawg_args->permuter;
	//
	// COMPOUND_PERM is unreachable on the LSTM path — nothing assigns it — so
	// that conjunct is always true and is omitted. What remains is the case that
	// matters: when this letter's own best permuter is PUNC_PERM and a core word
	// was already established, the OLD permuter is preserved. That is why
	// "(the)" reports SystemDawgPerm at the closing paren rather than PuncPerm,
	// and Task 23 branches on exactly that value.
	out := prev
	if prev == NoPerm || curr == NoPerm || curr != PuncPerm {
		out = curr
	}
	if out == NoPerm {
		return NoPerm, nil, false
	}
	return out, updated, validEnd
}

// successors is kDawgSuccessors: punctuation may be followed by the word and
// number dawgs, and those two only by punctuation.
func (d *Dict) successors(index int) []int {
	if index == puncIdx {
		var out []int
		for _, i := range []int{wordIdx, numberIdx} {
			if d.dawgs[i] != nil {
				out = append(out, i)
			}
		}
		return out
	}
	return nil
}

// charForDawg is Dict::char_for_dawg: inside the number dawg every digit folds
// to the pattern wildcard, so "2024" and "1999" share one path. The type comes
// from this package's own table, keyed on the dawg's index, because
// tessdata.Dawg carries no type tag.
func (d *Dict) charForDawg(unicharID, dawgIndex int) int {
	if dawgIndex < 0 || dawgIndex >= len(d.types) || d.types[dawgIndex] != DawgNumber {
		return unicharID
	}
	ch, ok := d.charset.Char(unicharID)
	if ok && ch.Properties&tessdata.PropDigit != 0 {
		return tessdata.PatternUnicharID
	}
	return unicharID
}

// startingNode is Dict::GetStartingNode (src/dict/dict.h:395):
//
//	if (edge_ref == NO_EDGE) return 0;      // beginning to explore the dawg
//	NODE_REF node = dawg->next_node(edge_ref);
//	if (node == 0) node = NO_EDGE;          // end of word
//	return node;
//
// The next_node == 0 remapping is load-bearing and is NOT a nil-guard: a
// squished dawg writes next_node 0 to mean "every word through this edge
// terminates here", never "return to the root" (Trie::add_word_to_dawg).
// Returning 0 instead would re-enter the root and let the lexicon accept
// arbitrary continuations of any complete word. It is also what makes the
// caller's `node != NoEdge` test reachable.
func startingNode(dw *tessdata.Dawg, ref int64) int64 {
	if ref == tessdata.NoEdge {
		return 0
	}
	node := dw.NextNode(ref)
	if node == 0 {
		return tessdata.NoEdge
	}
	return node
}
