// This file is a Go translation of src/ccutil/unicharset.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Reserved unichar ids. kSpecialUnicharCodes, src/ccutil/unicharset.cpp:79,
// is {" ", "Joined", "|Broken|0|1"} and every unicharset with special codes
// starts with exactly those three.
const (
	UnicharSpace  = 0
	UnicharJoined = 1
	UnicharBroken = 2
)

// Character property bits. Stored HEXADECIMAL in the file.
// src/ccutil/unicharset.cpp:41.
const (
	PropAlpha uint32 = 0x1
	PropLower uint32 = 0x2
	PropUpper uint32 = 0x4
	PropDigit uint32 = 0x8
	PropPunct uint32 = 0x10
)

// maxUnicharsetSize is a corruption guard, not a Tesseract constant. The
// largest shipped unicharset (chi_sim) is a few thousand entries.
const maxUnicharsetSize = 1_000_000

// Unichar is one unicharset entry, reduced to the columns the LSTM
// recognition path reads.
//
// Deliberately not retained: the comma-separated metrics tuple (top/bottom,
// width, bearing and advance statistics), other_case, direction and mirror.
// Every accessor for those is reachable only from the legacy Tesseract-3
// classifier, the wordrec language model, or x-height fixing — none of which
// cadmus ports.
type Unichar struct {
	// Text is the raw file token, the UTF-8 that id_to_unichar returns and
	// that the `tesseract` binary emits.
	Text string
	// Normed is get_normed_unichar's result: LSTMRecognizer::DecodeLabel uses
	// it, so it differs from Text for 6 eng entries (curly quotes, em dash,
	// trade mark). Defaults to Text when the column is absent.
	Normed string
	// Script drives IsSpaceDelimited, i.e. word splitting.
	Script     string
	Properties uint32
}

// Unicharset is a parsed lstm-unicharset component.
type Unicharset struct {
	chars []Unichar
}

// ParseUnicharset parses the text format read by UNICHARSET::load_via_fgets.
func ParseUnicharset(data []byte) (*Unicharset, error) {
	sc := bufio.NewScanner(bytes.NewReader(data))
	if !sc.Scan() {
		return nil, errors.New("tessdata: unicharset is empty")
	}
	n, err := strconv.Atoi(strings.TrimSpace(sc.Text()))
	if err != nil {
		return nil, fmt.Errorf("tessdata: unicharset entry count %q: %w", sc.Text(), err)
	}
	// Deliberate divergence: load_via_fgets accepts a count of 0 and returns
	// true. An empty LSTM unicharset cannot yield a usable model, so reject it
	// here where the error can say why.
	if n <= 0 || n > maxUnicharsetSize {
		return nil, fmt.Errorf("tessdata: implausible unicharset size %d", n)
	}
	u := &Unicharset{chars: make([]Unichar, 0, n)}
	for id := range n {
		if !sc.Scan() {
			return nil, fmt.Errorf("tessdata: unicharset declares %d entries but ends after %d", n, id)
		}
		c, err := parseUnicharLine(sc.Text())
		if err != nil {
			return nil, fmt.Errorf("tessdata: unicharset entry %d: %w", id, err)
		}
		u.chars = append(u.chars, c)
	}
	// Anything after the declared count is ignored, matching load_via_fgets.
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("tessdata: reading unicharset: %w", err)
	}
	return u, nil
}

// parseUnicharLine parses one entry line. The C++ loader's six progressive
// extractions accept: unichar, HEX properties, an optional comma-separated
// metrics tuple, a script name, then up to four further fields
// (other_case, direction, mirror, normed).
func parseUnicharLine(line string) (Unichar, error) {
	// Everything from the first tab is a human-readable debug comment that
	// load_via_fgets never reaches, because its istream extraction stops at
	// the field count it wants.
	if i := strings.IndexByte(line, '\t'); i >= 0 {
		line = line[:i]
	}
	f := strings.Fields(line)
	if len(f) < 3 {
		return Unichar{}, fmt.Errorf("want at least 3 fields, got %d in %q", len(f), line)
	}
	c := Unichar{Text: f[0]}
	// The literal token "NULL" in column 1 spells the space character.
	if c.Text == "NULL" {
		c.Text = " "
	}
	props, err := strconv.ParseUint(f[1], 16, 32)
	if err != nil {
		return Unichar{}, fmt.Errorf("properties %q is not hexadecimal: %w", f[1], err)
	}
	c.Properties = uint32(props)

	// Field 3 is the metrics tuple when it contains commas, else the script.
	si := 2
	if strings.ContainsRune(f[2], ',') {
		si = 3
	}
	if si >= len(f) {
		return Unichar{}, fmt.Errorf("no script field in %q", line)
	}
	c.Script = f[si]

	// After the script come other_case, direction, mirror, normed, in that
	// order. normed is present only when all four are.
	if rest := f[si+1:]; len(rest) == 4 {
		c.Normed = rest[3]
	} else {
		// set_normed(id, normed[0] != '\0' ? normed : unichar). Because Text
		// has already been rewritten from "NULL" to " ", this also subsumes
		// get_normed_unichar's UNICHAR_SPACE special case.
		c.Normed = c.Text
	}
	return c, nil
}

func (u *Unicharset) Size() int { return len(u.chars) }

func (u *Unicharset) Char(id int) (Unichar, bool) {
	if id < 0 || id >= len(u.chars) {
		return Unichar{}, false
	}
	return u.chars[id], true
}

// Text is id_to_unichar: the raw file representation.
func (u *Unicharset) Text(id int) string {
	c, _ := u.Char(id)
	return c.Text
}

// Normed is get_normed_unichar.
func (u *Unicharset) Normed(id int) string {
	c, _ := u.Char(id)
	return c.Normed
}

// IsSpaceDelimited mirrors UNICHARSET::IsSpaceDelimited
// (src/ccutil/unicharset.h:668): every script except Han, Thai, Hangul,
// Hiragana and Katakana writes words with spaces between them.
//
// Deliberate divergence: Tesseract compares resolved script *ids*, and
// get_script_id_from_name returns 0 (the null script) for a name it has never
// seen — so in a unicharset with no Han entry, han_sid_ collides with
// null_sid_ and a null-script character is reported as NOT space delimited.
// Comparing names directly avoids that. eng has no null-script entry, so the
// two agree on the model at hand.
func (u *Unicharset) IsSpaceDelimited(id int) bool {
	c, ok := u.Char(id)
	if !ok {
		return true // INVALID_UNICHAR_ID
	}
	switch c.Script {
	case "Han", "Thai", "Hangul", "Hiragana", "Katakana":
		return false
	}
	return true
}
