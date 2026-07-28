package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadRealUnicharset(t *testing.T) *Unicharset {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, ok := c.Entry(TypeLSTMUnicharset)
	if !ok {
		t.Fatal("eng.traineddata has no lstm-unicharset component")
	}
	u, err := ParseUnicharset(b)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	return u
}

func TestParseUnicharsetRealModel(t *testing.T) {
	u := loadRealUnicharset(t)

	if u.Size() != 112 {
		t.Fatalf("Size() = %d; want 112", u.Size())
	}

	// The three mandatory SpecialUnicharCodes, src/ccutil/unicharset.cpp:79.
	// Note id 0 is written as the literal token "NULL".
	for _, tc := range []struct {
		id           int
		text, normed string
	}{
		{UnicharSpace, " ", " "},
		{UnicharJoined, "Joined", "Joined"},
		{UnicharBroken, "|Broken|0|1", "|Broken|0|1"},
		{3, "C", "C"},
		{55, "’", "'"},  // right single quote -> ASCII apostrophe
		{59, "™", "TM"}, // trade mark sign
		{70, "—", "-"},  // em dash
		{111, "é", "é"},
	} {
		if got := u.Text(tc.id); got != tc.text {
			t.Errorf("Text(%d) = %q; want %q", tc.id, got, tc.text)
		}
		if got := u.Normed(tc.id); got != tc.normed {
			t.Errorf("Normed(%d) = %q; want %q", tc.id, got, tc.normed)
		}
	}

	// Properties are HEXADECIMAL. |Broken|0|1 carries "f"; a decimal parse
	// would fail outright, and every value below 0xa would parse the same
	// either way, so this entry is the one that proves the base.
	if got := u.mustChar(t, UnicharBroken).Properties; got != 0xf {
		t.Errorf("Properties(|Broken|0|1) = %#x; want 0xf", got)
	}
	if got := u.mustChar(t, 3).Properties; got != PropAlpha|PropUpper {
		t.Errorf("Properties(C) = %#x; want %#x", got, PropAlpha|PropUpper)
	}

	// eng uses only Common and Latin, and every entry is space delimited.
	for id := range u.Size() {
		c, _ := u.Char(id)
		if c.Script != "Common" && c.Script != "Latin" {
			t.Errorf("Char(%d).Script = %q; want Common or Latin", id, c.Script)
		}
		if !u.IsSpaceDelimited(id) {
			t.Errorf("IsSpaceDelimited(%d) = false; want true for a Latin script", id)
		}
	}
}

// Tesseract reads unicharset lines through a char[256] buffer via TFile::FGets
// (src/ccutil/serialis.cpp:213), which copies at most 255 bytes and stops early
// at '\n'. It does NOT discard the rest of an over-long line: the remainder
// stays in the stream and is consumed as the NEXT entry line, so every
// subsequent unichar id shifts and the whole file desynchronises. Our parser
// reads whole lines and does not. eng's longest line is 82 bytes; if a model
// ever exceeds the cap the two parses disagree about every id past that point,
// and that must be visible, not silent.
func TestUnicharsetLinesFitTesseractsBuffer(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, _ := c.Entry(TypeLSTMUnicharset)
	longest := 0
	for _, line := range strings.Split(string(b), "\n") {
		if len(line) > longest {
			longest = len(line)
		}
	}
	t.Logf("longest lstm-unicharset line: %d bytes", longest)
	if longest > 255 {
		t.Errorf("longest line is %d bytes; Tesseract's FGets stops at 255 and reads the remainder as the next entry, shifting every subsequent unichar id", longest)
	}
}

func TestParseUnicharsetShapes(t *testing.T) {
	// Shape A (space only, 4 fields) and shape B (full, 8 fields).
	in := "2\n" +
		"NULL 0 Common 0\n" +
		"é 3 0,255,0,255,0,0,0,0,0,0 Latin 111 0 111 e\t# é [e9 ]a\n"
	u, err := ParseUnicharset([]byte(in))
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	if u.Size() != 2 {
		t.Fatalf("Size() = %d; want 2", u.Size())
	}
	if u.Text(0) != " " || u.Normed(0) != " " {
		t.Errorf("id 0 = %q/%q; want \" \"/\" \"", u.Text(0), u.Normed(0))
	}
	if u.Text(1) != "é" || u.Normed(1) != "e" {
		t.Errorf("id 1 = %q/%q; want \"é\"/\"e\"", u.Text(1), u.Normed(1))
	}
	if got := u.mustChar(t, 1).Properties; got != PropAlpha|PropLower {
		t.Errorf("Properties(id 1) = %#x; want %#x", got, PropAlpha|PropLower)
	}
	if u.mustChar(t, 1).Script != "Latin" {
		t.Errorf("Script(id 1) = %q; want Latin", u.mustChar(t, 1).Script)
	}
}

func TestParseUnicharsetErrors(t *testing.T) {
	for name, in := range map[string]string{
		"empty":            "",
		"bad count":        "abc\nNULL 0 Common 0\n",
		"count too high":   "3\nNULL 0 Common 0\n",
		"non-hex property": "1\nA zz Latin 0\n",
		"too few fields":   "1\nA 3\n",
		"zero count":       "0\n",
	} {
		if _, err := ParseUnicharset([]byte(in)); err == nil {
			t.Errorf("ParseUnicharset(%s): want error, got nil", name)
		}
	}
}

// Trailing bytes after the declared entry count are ignored, matching
// load_via_fgets, which reads exactly unicharset_size lines.
func TestParseUnicharsetIgnoresTrailingLines(t *testing.T) {
	u, err := ParseUnicharset([]byte("1\nNULL 0 Common 0\nJoined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n"))
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	if u.Size() != 1 {
		t.Errorf("Size() = %d; want 1", u.Size())
	}
}

func (u *Unicharset) mustChar(t *testing.T, id int) Unichar {
	t.Helper()
	c, ok := u.Char(id)
	if !ok {
		t.Fatalf("Char(%d) not present", id)
	}
	return c
}
