// This file is a Go translation of src/dict/dawg.cpp and src/dict/dawg.h from
// Tesseract OCR (https://github.com/tesseract-ocr/tesseract), licensed under
// the Apache License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"fmt"
	"math"
	"math/bits"
)

// dawgMagic is kDawgMagicNumber, src/dict/dawg.h:113.
const dawgMagic int16 = 42

// Edge-record flags, all shifted left by flagStartBit. src/dict/dawg.h:77.
const (
	dawgMarkerFlag    = 1 // last edge of this node's run
	dawgDirectionFlag = 2 // set for a backward edge
	dawgWordEndFlag   = 4
	dawgNumFlagBits   = 3
)

// maxDawgEdges is read_squished_dawg's resource-exhaustion guard.
const maxDawgEdges = 50_000_000

// Dawg is a SquishedDawg: a directed acyclic word graph, as stored in the
// lstm-punc-dawg, lstm-word-dawg and lstm-number-dawg components.
//
// A node is identified by the index of its first edge; its edges are the
// consecutive run ending at the edge whose MARKER flag is set. Node 0 is the
// root, so a next_node of 0 means "no successor" rather than "back to the
// root" — see Trie::add_word_to_dawg.
type Dawg struct {
	UnicharsetSize int

	edges            []uint64
	letterMask       uint64
	flagStartBit     uint
	nextNodeStartBit uint
}

// ParseDawg deserializes one lstm-*-dawg component. swap comes from
// Container.Swapped.
func ParseDawg(data []byte, swap bool) (*Dawg, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	magic, err := r.Int16()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg magic: %w", err)
	}
	if magic != dawgMagic {
		return nil, fmt.Errorf("tessdata: bad dawg magic %d, want %d", magic, dawgMagic)
	}
	size, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg unicharset size: %w", err)
	}
	if size == 0 || size > math.MaxInt32 {
		return nil, fmt.Errorf("tessdata: bad dawg unicharset size %d", size)
	}
	numEdges, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg edge count: %w", err)
	}
	if numEdges == 0 {
		return nil, fmt.Errorf("tessdata: dawg has 0 edges")
	}
	if numEdges > maxDawgEdges {
		return nil, fmt.Errorf("tessdata: dawg edge count %d exceeds the %d hard limit", numEdges, maxDawgEdges)
	}
	if int(numEdges) > r.Remaining()/8 {
		return nil, fmt.Errorf("tessdata: dawg declares %d edges but only %d bytes remain", numEdges, r.Remaining())
	}

	d := &Dawg{UnicharsetSize: int(size), edges: make([]uint64, numEdges)}
	for i := range d.edges {
		if d.edges[i], err = r.Uint64(); err != nil {
			return nil, fmt.Errorf("tessdata: reading dawg edge %d: %w", i, err)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the dawg", r.Remaining())
	}
	d.initMasks()
	if err := d.validate(); err != nil {
		return nil, err
	}
	return d, nil
}

// initMasks mirrors Dawg::init (src/dict/dawg.cpp:188).
//
// flagStartBit is the BIT LENGTH of unicharset_size. Tesseract's CeilLog2 is
// misnamed: it counts bits, so CeilLog2(64) == 7 while ceil(log2(64)) == 6.
// This is NOT the same function as ceilLog2 in network.go, which is
// src/lstm/lstm.cpp's ceil_log2. Confusing them shifts every mask by one bit.
func (d *Dawg) initMasks() {
	d.flagStartBit = uint(bits.Len(uint(d.UnicharsetSize)))
	d.nextNodeStartBit = d.flagStartBit + dawgNumFlagBits
	d.letterMask = 1<<d.flagStartBit - 1
}

// validate enforces the invariants SquishedDawg::write_squished_dawg
// guarantees: it emits forward edges only, and rewrites every next_node through
// build_node_map so it indexes the written array. Verified on all three
// lstm-*-dawg components of tessdata_best eng.
func (d *Dawg) validate() error {
	for i, e := range d.edges {
		if e&(dawgDirectionFlag<<d.flagStartBit) != 0 {
			return fmt.Errorf("tessdata: dawg edge %d is a backward edge; only written (forward-only) dawgs are supported", i)
		}
		if id := int(e & d.letterMask); id >= d.UnicharsetSize {
			return fmt.Errorf("tessdata: dawg edge %d has unichar id %d outside the %d-entry unicharset", i, id, d.UnicharsetSize)
		}
		if next := int(e >> d.nextNodeStartBit); next >= len(d.edges) {
			return fmt.Errorf("tessdata: dawg edge %d points at node %d beyond the %d edges in the file", i, next, len(d.edges))
		}
	}
	return nil
}

func (d *Dawg) NumEdges() int { return len(d.edges) }

func (d *Dawg) nextNode(edge int) int { return int(d.edges[edge] >> d.nextNodeStartBit) }

// edgeOf mirrors SquishedDawg::edge_char_of (src/dict/dawg.cpp:207).
//
// Tesseract binary-searches node 0 and linear-scans every other node. edgeOf
// linear-scans everywhere, which is a simplification, not an identity: the
// binary search's comparator (given_greater_than_edge_rec, src/dict/dawg.h:242)
// orders on (unichar_id, next_node, word_end), so the format permits node 0 to
// hold two edges with the same unichar id, and on such a node the two searches
// can pick different edges.
//
// What was measured: tessdata_best eng's lstm-word-dawg node 0 has 67 edges,
// sorted ascending, with no repeated unichar id. What enforces it for all three
// lexicons at test time: TestDawgNode0IsSortedAndDuplicateFree. If that test
// ever fails on a model, this function must grow the binary search for node 0
// rather than the test being relaxed.
func (d *Dawg) edgeOf(node, id int, wordEnd bool) int {
	for e := node; e < len(d.edges); e++ {
		rec := d.edges[e]
		if int(rec&d.letterMask) == id &&
			(!wordEnd || rec&(dawgWordEndFlag<<d.flagStartBit) != 0) {
			return e
		}
		if rec&(dawgMarkerFlag<<d.flagStartBit) != 0 {
			break
		}
	}
	return -1
}

// Contains reports whether the dawg holds the complete word spelled by ids.
// It is Dawg::word_in_dawg, i.e. prefixInDawg with requiresComplete = true.
//
// WORD DAWGS ONLY. Every unichar id on an edge is matched literally, which is
// correct for lstm-word-dawg (DAWG_TYPE_WORD) and WRONG for lstm-punc-dawg and
// lstm-number-dawg: in those, Dawg::kPatternUnicharID (== 0, src/dict/dawg.h:117)
// on an edge is a WILDCARD meaning "any dictionary word" / "any digit run", not
// the space character. Matching it literally silently mis-answers both of those
// lexicons. L1a has no consumer that needs pattern semantics — cadmusdump prints
// only edge counts — so the wildcard machinery (DawgPosition,
// Dict::LetterIsOkay) is deferred to L2b's internal/dict along with the
// DawgType tag that would make these methods safe to call on all three.
func (d *Dawg) Contains(ids []int) bool { return d.prefixInDawg(ids, true) }

// HasPrefix reports whether ids is a prefix of at least one word in the dawg.
// It is Dawg::prefix_in_dawg with requires_complete = false.
//
// WORD DAWGS ONLY, for the same reason as Contains.
func (d *Dawg) HasPrefix(ids []int) bool { return d.prefixInDawg(ids, false) }

// prefixInDawg mirrors Dawg::prefix_in_dawg (src/dict/dawg.cpp:42).
func (d *Dawg) prefixInDawg(ids []int, requiresComplete bool) bool {
	if len(ids) == 0 {
		return !requiresComplete
	}
	node := 0
	for _, id := range ids[:len(ids)-1] {
		e := d.edgeOf(node, id, false)
		if e < 0 {
			return false
		}
		// next_node == 0 means every word through this edge terminates here;
		// there are no longer words. See Trie::add_word_to_dawg.
		if node = d.nextNode(e); node == 0 {
			return false
		}
	}
	return d.edgeOf(node, ids[len(ids)-1], requiresComplete) >= 0
}

// NoEdge is Tesseract's NO_EDGE.
const NoEdge = int64(-1)

// PatternUnicharID is Dawg::kPatternUnicharID: inside a punctuation or number
// dawg, unichar id 0 on an edge is a wildcard, not the space character. The
// collision with UnicharSpace is deliberate in Tesseract.
const PatternUnicharID = 0

// UnicharID is unichar_id_from_edge_rec: the character this edge consumes.
func (d *Dawg) UnicharID(edge int64) int { return int(d.edges[edge] & d.letterMask) }

// NextNode is next_node_from_edge_rec. No mask is needed: nextNodeStartBit is
// flagStartBit + 3, so the shift already drops the id and the three flags.
func (d *Dawg) NextNode(edge int64) int64 {
	return int64(d.edges[edge] >> d.nextNodeStartBit)
}

// EndOfWord is end_of_word_from_edge_rec: this edge terminates a word.
func (d *Dawg) EndOfWord(edge int64) bool {
	return d.edges[edge]&(dawgWordEndFlag<<d.flagStartBit) != 0
}

// lastEdge is marker_flag_from_edge_rec: this edge closes its node's run.
func (d *Dawg) lastEdge(edge int64) bool {
	return d.edges[edge]&(dawgMarkerFlag<<d.flagStartBit) != 0
}

// emptySlot is edge_occupied's sentinel: an edge record equal to
// next_node_mask_ is a hole left by the squishing pass, not an edge.
func (d *Dawg) emptySlot() uint64 { return ^uint64(0) << d.nextNodeStartBit }

// EdgeChar is SquishedDawg::edge_char_of: the edge leaving node on unicharID,
// or NoEdge.
//
// Two divergences from the C++, both deliberate and both bounded by
// TestRealDawgsSatisfyTheLinearScanPreconditions:
//
//   - Node 0 is scanned linearly rather than binary-searched. The two agree
//     whenever the root's unichar ids are distinct, which the test asserts on
//     every shipped dawg; eng's roots are 8, 67 and 40 edges.
//   - The direction bit is not tested, because edge_char_of does not test it
//     either — read_squished_dawg's validation loop does, at load time, and a
//     written file contains only forward edges.
//
// The empty-slot guard IS reproduced, because edge_char_of has it
// (`if (edge != NO_EDGE && edge_occupied(edge))`) and an empty slot's unichar id
// is 0, which is exactly the wildcard a punctuation-dawg probe asks for.
func (d *Dawg) EdgeChar(node int64, unicharID int, wordEnd bool) int64 {
	if node < 0 || node >= int64(len(d.edges)) {
		return NoEdge
	}
	if d.edges[node] == d.emptySlot() {
		return NoEdge
	}
	for e := node; e < int64(len(d.edges)); e++ {
		if d.UnicharID(e) == unicharID && (!wordEnd || d.EndOfWord(e)) {
			return e
		}
		if d.lastEdge(e) {
			break
		}
	}
	return NoEdge
}
