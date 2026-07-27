// This file is a Go translation of src/ccutil/tessdatamanager.cpp from
// Tesseract OCR (https://github.com/tesseract-ocr/tesseract), licensed under
// the Apache License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"fmt"
	"strings"
)

// maxNumEntries mirrors kMaxNumTessdataEntries. A count above it in the
// file's own byte order means the file was written on the other endianness.
const maxNumEntries = 1000

type Type int

const (
	TypeLangConfig     Type = 0
	TypeUnicharset     Type = 1
	TypeAmbigs         Type = 2
	TypeIntTemp        Type = 3  // legacy engine
	TypePFFMTable      Type = 4  // legacy engine
	TypeNormProto      Type = 5  // legacy engine
	TypePuncDawg       Type = 6
	TypeSystemDawg     Type = 7
	TypeNumberDawg     Type = 8
	TypeFreqDawg       Type = 9
	TypeShapeTable     Type = 13 // legacy engine
	TypeBigramDawg     Type = 14
	TypeUnambigDawg    Type = 15
	TypeParamsModel    Type = 16
	TypeLSTM           Type = 17
	TypeLSTMPuncDawg   Type = 18
	TypeLSTMSystemDawg Type = 19
	TypeLSTMNumberDawg Type = 20
	TypeLSTMUnicharset Type = 21
	TypeLSTMRecoder    Type = 22
	TypeVersion        Type = 23

	numTypes Type = 24
)

var typeNames = map[Type]string{
	TypeLangConfig: "lang_config", TypeUnicharset: "unicharset", TypeAmbigs: "ambigs",
	TypeIntTemp: "inttemp", TypePFFMTable: "pffmtable", TypeNormProto: "normproto",
	TypePuncDawg: "punc_dawg", TypeSystemDawg: "system_dawg", TypeNumberDawg: "number_dawg",
	TypeFreqDawg: "freq_dawg", TypeShapeTable: "shapetable", TypeBigramDawg: "bigram_dawg",
	TypeUnambigDawg: "unambig_dawg", TypeParamsModel: "params_model", TypeLSTM: "lstm",
	TypeLSTMPuncDawg: "lstm-punc-dawg", TypeLSTMSystemDawg: "lstm-word-dawg",
	TypeLSTMNumberDawg: "lstm-number-dawg", TypeLSTMUnicharset: "lstm-unicharset",
	TypeLSTMRecoder: "lstm-recoder", TypeVersion: "version",
}

func (t Type) String() string {
	if n, ok := typeNames[t]; ok {
		return n
	}
	return fmt.Sprintf("type(%d)", int(t))
}

// Container is a parsed .traineddata file. Entry slices alias the input buffer.
type Container struct {
	entries map[Type][]byte
	swap    bool
}

// ParseContainer parses Tesseract's native .traineddata layout.
//
// Note: Tesseract built with libarchive also accepts zip/tar archives of
// component files. The models published in tessdata, tessdata_best, and
// tessdata_fast all use the native layout parsed here.
func ParseContainer(data []byte) (*Container, error) {
	r := NewReader(data)
	n, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading entry count: %w", err)
	}
	swap := n > maxNumEntries
	if swap {
		n = bits32Reverse(n)
		r.SetSwap(true)
	}
	if n > maxNumEntries {
		return nil, fmt.Errorf("tessdata: entry count %d exceeds maximum %d in both byte orders", n, maxNumEntries)
	}

	offsets := make([]int64, n)
	for i := range offsets {
		if offsets[i], err = r.Int64(); err != nil {
			return nil, fmt.Errorf("tessdata: reading offset %d: %w", i, err)
		}
	}

	size := int64(len(data))
	c := &Container{entries: make(map[Type][]byte), swap: swap}
	for i := range offsets {
		if offsets[i] < 0 {
			continue
		}
		if offsets[i] > size {
			return nil, fmt.Errorf("tessdata: entry %d offset %d past end of %d-byte file", i, offsets[i], size)
		}
		end := size
		for j := i + 1; j < len(offsets); j++ {
			if offsets[j] >= 0 {
				if offsets[j] > size {
					return nil, fmt.Errorf("tessdata: entry %d offset %d past end of %d-byte file", j, offsets[j], size)
				}
				end = offsets[j]
				break
			}
		}
		if end < offsets[i] {
			return nil, fmt.Errorf("tessdata: entry %d has negative length", i)
		}
		if Type(i) < numTypes {
			c.entries[Type(i)] = data[offsets[i]:end]
		}
	}
	return c, nil
}

// Swapped reports whether the source file was foreign-endian. Readers created
// for its entries must be told.
func (c *Container) Swapped() bool { return c.swap }

func (c *Container) Entry(t Type) ([]byte, bool) {
	b, ok := c.entries[t]
	return b, ok
}

// Present lists the component types in the file, in type order.
func (c *Container) Present() []Type {
	var out []Type
	for t := Type(0); t < numTypes; t++ {
		if _, ok := c.entries[t]; ok {
			out = append(out, t)
		}
	}
	return out
}

// Version returns the model's version string, or "" if the file predates it.
func (c *Container) Version() string {
	b, ok := c.entries[TypeVersion]
	if !ok {
		return ""
	}
	return strings.TrimRight(string(b), "\x00\n")
}
