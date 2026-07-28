// This file is a Go translation of src/ccutil/serialis.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"encoding/binary"
	"fmt"
	"math"
)

// maxStringLen mirrors Tesseract's arbitrary corruption guard in
// TFile::DeSerialize(std::string&).
const maxStringLen = 50_000_000

// Reader reads Tesseract's serialization primitives from an in-memory buffer.
// Tesseract writes in host byte order and detects a foreign-endian file at the
// container header, then byte-swaps every subsequent multi-byte read; SetSwap
// reproduces that.
type Reader struct {
	data []byte
	pos  int
	swap bool
}

func NewReader(data []byte) *Reader { return &Reader{data: data} }

func (r *Reader) SetSwap(swap bool) { r.swap = swap }

func (r *Reader) Remaining() int { return len(r.data) - r.pos }

func (r *Reader) Bytes(n int) ([]byte, error) {
	if n < 0 {
		return nil, fmt.Errorf("tessdata: negative read length %d", n)
	}
	if r.Remaining() < n {
		return nil, fmt.Errorf("tessdata: short read at offset %d: want %d bytes, have %d", r.pos, n, r.Remaining())
	}
	b := r.data[r.pos : r.pos+n]
	r.pos += n
	return b, nil
}

func (r *Reader) Uint8() (uint8, error) {
	b, err := r.Bytes(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *Reader) Int8() (int8, error) {
	v, err := r.Uint8()
	return int8(v), err
}

func (r *Reader) Uint32() (uint32, error) {
	b, err := r.Bytes(4)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(b)
	if r.swap {
		v = bits32Reverse(v)
	}
	return v, nil
}

func (r *Reader) Int32() (int32, error) {
	v, err := r.Uint32()
	return int32(v), err
}

// Int16 reads a 2-byte signed integer. Only the DAWG header uses one
// (kDawgMagicNumber, src/dict/dawg.cpp).
func (r *Reader) Int16() (int16, error) {
	b, err := r.Bytes(2)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint16(b)
	if r.swap {
		v = v>>8 | v<<8
	}
	return int16(v), nil
}

// Float32 reads a 4-byte IEEE-754 value. Tesseract writes `float` rather than
// `TFloat` for the recognizer's adam_beta_, learning_rate_ and momentum_
// (src/lstm/lstmrecognizer.h) and for Plumbing::learning_rates_.
func (r *Reader) Float32() (float32, error) {
	b, err := r.Bytes(4)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(b)
	if r.swap {
		v = bits32Reverse(v)
	}
	return math.Float32frombits(v), nil
}

// Uint64 reads an 8-byte unsigned integer. DAWG edge records are uint64.
func (r *Reader) Uint64() (uint64, error) {
	b, err := r.Bytes(8)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(b)
	if r.swap {
		v = bits64Reverse(v)
	}
	return v, nil
}

func (r *Reader) Int64() (int64, error) {
	v, err := r.Uint64()
	return int64(v), err
}

func (r *Reader) Float64() (float64, error) {
	b, err := r.Bytes(8)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(b)
	if r.swap {
		v = bits64Reverse(v)
	}
	return math.Float64frombits(v), nil
}

func (r *Reader) String() (string, error) {
	n, err := r.Uint32()
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("tessdata: implausible string length %d at offset %d", n, r.pos)
	}
	if n == 0 {
		return "", nil
	}
	b, err := r.Bytes(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Float64Slice reads a uint32 count followed by that many float64 values.
func (r *Reader) Float64Slice() ([]float64, error) {
	n, err := r.Uint32()
	if err != nil {
		return nil, err
	}
	if int(n) > r.Remaining()/8 {
		return nil, fmt.Errorf("tessdata: float64 count %d exceeds %d remaining bytes", n, r.Remaining())
	}
	out := make([]float64, n)
	for i := range out {
		if out[i], err = r.Float64(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func bits32Reverse(v uint32) uint32 {
	return v>>24 | (v>>8)&0x0000ff00 | (v<<8)&0x00ff0000 | v<<24
}

func bits64Reverse(v uint64) uint64 {
	return v>>56 | (v>>40)&0x000000000000ff00 | (v>>24)&0x0000000000ff0000 |
		(v>>8)&0x00000000ff000000 | (v<<8)&0x000000ff00000000 |
		(v<<24)&0x0000ff0000000000 | (v<<40)&0x00ff000000000000 | v<<56
}
