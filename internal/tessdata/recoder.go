// This file is a Go translation of src/ccutil/unicharcompress.cpp and
// src/ccutil/unicharcompress.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import "fmt"

// maxCodeLen is RecodedCharID::kMaxCodeLen, src/ccutil/unicharcompress.h:37.
const maxCodeLen = 9

// codeKey is a comparable form of a code sequence, usable as a map key.
type codeKey struct {
	n    int
	code [maxCodeLen]int32
}

func makeCodeKey(code []int32) (codeKey, bool) {
	if len(code) == 0 || len(code) > maxCodeLen {
		return codeKey{}, false
	}
	k := codeKey{n: len(code)}
	copy(k.code[:], code)
	return k, true
}

// Recoder maps unichar ids to the network's output code sequences and back.
//
// For eng every code has length 1 and the mapping is a near-permutation: the
// only collapse is unichar ids 1 ("Joined") and 2 ("|Broken|0|1") both encoding
// to 110, which is also the CTC blank. CJK and Indic models use genuinely
// multi-code sequences.
type Recoder struct {
	codes      [][]int32
	codeRange  int
	maxLen     int
	decoder    map[codeKey]int
	validStart []bool
}

// ParseRecoder deserializes a .traineddata lstm-recoder component. swap comes
// from Container.Swapped.
func ParseRecoder(data []byte, swap bool) (*Recoder, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	n, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading recoder entry count: %w", err)
	}
	// TFile::DeSerialize(std::vector<T>&) rejects counts above 50,000,000.
	if n > maxStringLen {
		return nil, fmt.Errorf("tessdata: implausible recoder entry count %d", n)
	}
	// That bound alone still lets a 4-byte header ask for ~1.2 GB before a
	// single entry is read. The smallest possible entry is 5 bytes (int8
	// self_normalized + uint32 length, zero codes), so the buffer itself is
	// the tighter bound. ParseDawg has the same guard.
	if int(n) > r.Remaining()/5 {
		return nil, fmt.Errorf("tessdata: recoder declares %d entries but only %d bytes remain", n, r.Remaining())
	}
	rc := &Recoder{codes: make([][]int32, n), decoder: make(map[codeKey]int, n)}
	for i := range rc.codes {
		// self_normalized_ is int8 and is therefore never byte-swapped
		// (FReadEndian only reverses items wider than one byte). It is set to
		// 1 by RecodedCharID's constructor and assigned nowhere else in
		// Tesseract, so it carries no information. Read and discard.
		if _, err := r.Int8(); err != nil {
			return nil, fmt.Errorf("tessdata: recoder entry %d self_normalized: %w", i, err)
		}
		length, err := r.Uint32()
		if err != nil {
			return nil, fmt.Errorf("tessdata: recoder entry %d length: %w", i, err)
		}
		if length > maxCodeLen {
			return nil, fmt.Errorf("tessdata: recoder entry %d length %d exceeds kMaxCodeLen %d", i, length, maxCodeLen)
		}
		code := make([]int32, length)
		for j := range code {
			if code[j], err = r.Int32(); err != nil {
				return nil, fmt.Errorf("tessdata: recoder entry %d code %d: %w", i, j, err)
			}
			if code[j] < 0 {
				return nil, fmt.Errorf("tessdata: recoder entry %d code %d is negative (%d)", i, j, code[j])
			}
		}
		rc.codes[i] = code
		if int(length) > rc.maxLen {
			rc.maxLen = int(length)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the recoder", r.Remaining())
	}

	rc.computeCodeRange()
	if err := rc.setupDecoder(); err != nil {
		return nil, err
	}
	return rc, nil
}

// computeCodeRange mirrors UnicharCompress::ComputeCodeRange: max code + 1.
func (rc *Recoder) computeCodeRange() {
	max := -1
	for _, code := range rc.codes {
		for _, v := range code {
			if int(v) > max {
				max = int(v)
			}
		}
	}
	rc.codeRange = max + 1
}

// setupDecoder mirrors UnicharCompress::SetupDecoder. It iterates unichar ids
// in ascending order and overwrites, so when two ids share a code the HIGHER id
// wins: in eng, ids 1 and 2 both encode to [110], and [110] decodes to 2.
//
// next_codes_ and final_codes_ are deliberately not built. They are consumed
// only by RecodeBeamSearch, which is L2b.
//
// Deliberate divergence for a zero-length entry: Tesseract registers it in
// decoder_ and sets is_valid_start_[code(0)] BEFORE its `if (len == 0) continue`
// (src/ccutil/unicharcompress.cpp:400-408), where code(0) of an empty
// RecodedCharID is 0. We skip it entirely — an empty code sequence cannot be
// decoded, and makeCodeKey refuses it. No shipped model has one.
//
// No range check on code[0] is needed: computeCodeRange has just set codeRange
// to max(all codes)+1 over this same array, so every code is inside [0, range)
// by construction, and negatives were rejected at read time.
func (rc *Recoder) setupDecoder() error {
	rc.validStart = make([]bool, rc.codeRange)
	for id, code := range rc.codes {
		if len(code) == 0 {
			// FReadEndian(..., 0) succeeds, so a zero-length entry is legal.
			continue
		}
		k, ok := makeCodeKey(code)
		if !ok {
			return fmt.Errorf("tessdata: recoder entry %d has an unusable code length %d", id, len(code))
		}
		rc.decoder[k] = id
		rc.validStart[code[0]] = true
	}
	// LSTMRecognizer::LoadRecoder's own check (src/lstm/lstmrecognizer.cpp:198):
	// "Space was garbled in recoding!!"
	if len(rc.codes) <= UnicharSpace || len(rc.codes[UnicharSpace]) == 0 ||
		rc.codes[UnicharSpace][0] != UnicharSpace {
		return fmt.Errorf("tessdata: space was garbled in recoding: unichar %d encodes to %v", UnicharSpace, rc.Encode(UnicharSpace))
	}
	return nil
}

// Size is the number of entries, which is one per unichar id.
func (rc *Recoder) Size() int { return len(rc.codes) }

// CodeRange is the number of distinct code values, and therefore the network's
// output count. It is the authority: NetworkBuilder::ParseOutput overrides the
// spec string's output count with it.
func (rc *Recoder) CodeRange() int { return rc.codeRange }

// MaxCodeLen is the longest code sequence in the recoder. 1 means the mapping
// is flat and a decoder can skip the multi-code lookahead entirely.
func (rc *Recoder) MaxCodeLen() int { return rc.maxLen }

// Encode returns the code sequence for a unichar id, or nil if out of range.
// The slice aliases the Recoder; do not modify it.
func (rc *Recoder) Encode(unicharID int) []int32 {
	if unicharID < 0 || unicharID >= len(rc.codes) {
		return nil
	}
	return rc.codes[unicharID]
}

// DecodeUnichar returns the unichar id for a complete code sequence.
func (rc *Recoder) DecodeUnichar(code []int32) (int, bool) {
	k, ok := makeCodeKey(code)
	if !ok {
		return 0, false
	}
	id, ok := rc.decoder[k]
	return id, ok
}

// IsValidFirstCode reports whether a code value can begin a sequence.
func (rc *Recoder) IsValidFirstCode(code int32) bool {
	if code < 0 || int(code) >= len(rc.validStart) {
		return false
	}
	return rc.validStart[code]
}
