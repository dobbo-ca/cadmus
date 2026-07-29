package dict

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func loadDict(t *testing.T) (*Dict, *tessdata.Unicharset) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	csEntry, _ := c.Entry(tessdata.TypeLSTMUnicharset)
	cs, err := tessdata.ParseUnicharset(csEntry)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	// tessdata.ParseDawg takes no type tag (L1a Task 7); Dict assigns the types.
	load := func(typ tessdata.Type) *tessdata.Dawg {
		e, ok := c.Entry(typ)
		if !ok {
			t.Fatalf("no %v component", typ)
		}
		d, err := tessdata.ParseDawg(e, c.Swapped())
		if err != nil {
			t.Fatalf("ParseDawg(%v) error = %v", typ, err)
		}
		return d
	}
	return New(cs,
		load(tessdata.TypeLSTMPuncDawg),
		load(tessdata.TypeLSTMSystemDawg),
		load(tessdata.TypeLSTMNumberDawg)), cs
}

func idsOf(t *testing.T, cs *tessdata.Unicharset, s string) []int {
	t.Helper()
	out := make([]int, 0, len(s))
	for _, r := range s {
		found := -1
		for id := range cs.Size() {
			if cs.Text(id) == string(r) {
				found = id
				break
			}
		}
		if found < 0 {
			t.Fatalf("no unichar for %q", r)
		}
		out = append(out, found)
	}
	return out
}

// walk feeds a whole string through LetterIsOkay and returns the final
// permuter and whether the last position was a valid word end.
//
// The permuter is threaded from letter to letter — that is what makes it an
// in/out parameter in the C++, and it is the whole point of the carry rule.
// Seeding with NoPerm reproduces DawgArgs' initial state.
func walk(t *testing.T, d *Dict, cs *tessdata.Unicharset, s string) (PermuterType, bool) {
	t.Helper()
	active := d.Start()
	ids := idsOf(t, cs, s)
	perm := NoPerm
	var validEnd bool
	for i, id := range ids {
		perm, active, validEnd = d.LetterIsOkay(perm, active, id, i == len(ids)-1)
		if perm == NoPerm {
			return NoPerm, false
		}
	}
	return perm, validEnd
}

func TestLetterIsOkayAcceptsDictionaryWords(t *testing.T) {
	d, cs := loadDict(t)
	for _, w := range []string{"the", "and", "zebra", "hello"} {
		perm, validEnd := walk(t, d, cs, w)
		if perm == NoPerm {
			t.Errorf("%q was rejected outright", w)
			continue
		}
		if !validEnd {
			t.Errorf("%q did not end on a valid word boundary", w)
		}
		if perm != SystemDawgPerm {
			t.Errorf("%q permuter = %d; want SystemDawgPerm (%d)", w, perm, SystemDawgPerm)
		}
	}
}

func TestLetterIsOkayRejectsNonWords(t *testing.T) {
	d, cs := loadDict(t)
	for _, w := range []string{"qqqqq", "zzzxq"} {
		if perm, _ := walk(t, d, cs, w); perm != NoPerm {
			t.Errorf("%q was accepted with permuter %d; want NoPerm", w, perm)
		}
	}
}

// A dictionary word wrapped in punctuation must reach the word dawg through the
// punctuation dawg's kPatternUnicharID transition.
func TestLetterIsOkayHandlesPunctuationWrapping(t *testing.T) {
	d, cs := loadDict(t)
	perm, validEnd := walk(t, d, cs, "(the)")
	if perm == NoPerm {
		t.Fatal("\"(the)\" was rejected; the punctuation dawg does not wrap the word dawg")
	}
	if !validEnd {
		t.Error("\"(the)\" did not end on a valid word boundary")
	}
	// The carry rule is what keeps this SystemDawgPerm rather than PuncPerm:
	// the closing paren's own curr_perm is PUNC_PERM, and the final block
	// therefore preserves the permuter the core word established.
	if perm != SystemDawgPerm {
		t.Errorf("\"(the)\" permuter = %d; want SystemDawgPerm (%d) — the permuter carry was dropped", perm, SystemDawgPerm)
	}
}

// A prefix of a real word is extendable but is not a valid end.
func TestLetterIsOkayPrefixIsNotAValidEnd(t *testing.T) {
	d, cs := loadDict(t)
	active := d.Start()
	ids := idsOf(t, cs, "zebr")
	perm := NoPerm
	var validEnd bool
	for _, id := range ids {
		perm, active, validEnd = d.LetterIsOkay(perm, active, id, false)
		if perm == NoPerm {
			t.Fatal("\"zebr\" was rejected; it is a prefix of \"zebra\"")
		}
	}
	if validEnd {
		t.Error("\"zebr\" reported a valid word end")
	}
	if len(active) == 0 {
		t.Error("\"zebr\" left no active positions to extend")
	}
}

// Digits are folded to kPatternUnicharID inside the number dawg.
func TestLetterIsOkayAcceptsNumbers(t *testing.T) {
	d, cs := loadDict(t)
	if perm, _ := walk(t, d, cs, "2024"); perm == NoPerm {
		t.Error("\"2024\" was rejected; digits must fold to the pattern id in the number dawg")
	}
}

// The pattern id must never be accepted as a literal letter.
func TestLetterIsOkayRejectsThePatternID(t *testing.T) {
	d, _ := loadDict(t)
	if perm, _, _ := d.LetterIsOkay(NoPerm, d.Start(), tessdata.PatternUnicharID, false); perm != NoPerm {
		t.Errorf("kPatternUnicharID was accepted with permuter %d; want NoPerm", perm)
	}
	// And a non-NoPerm carry must not rescue it: the wildcard rejection is
	// unconditional and happens before the carry block.
	if perm, _, _ := d.LetterIsOkay(SystemDawgPerm, d.Start(), tessdata.PatternUnicharID, false); perm != NoPerm {
		t.Errorf("kPatternUnicharID with a carried permuter = %d; want NoPerm", perm)
	}
}
