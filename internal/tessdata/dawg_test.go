package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func loadRealDawgs(t *testing.T) (punc, word, number *Dawg, u *Unicharset) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	get := func(typ Type) *Dawg {
		b, ok := c.Entry(typ)
		if !ok {
			t.Fatalf("eng.traineddata has no %v component", typ)
		}
		d, err := ParseDawg(b, c.Swapped())
		if err != nil {
			t.Fatalf("ParseDawg(%v) error = %v", typ, err)
		}
		return d
	}
	ub, _ := c.Entry(TypeLSTMUnicharset)
	u, err = ParseUnicharset(ub)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	return get(TypeLSTMPuncDawg), get(TypeLSTMSystemDawg), get(TypeLSTMNumberDawg), u
}

func TestParseDawgRealModel(t *testing.T) {
	punc, word, number, _ := loadRealDawgs(t)
	for _, tc := range []struct {
		name  string
		d     *Dawg
		edges int
	}{
		{"punc", punc, 539},
		{"word", word, 461848},
		{"number", number, 591},
	} {
		if tc.d.NumEdges() != tc.edges {
			t.Errorf("%s dawg NumEdges() = %d; want %d", tc.name, tc.d.NumEdges(), tc.edges)
		}
		if tc.d.UnicharsetSize != 112 {
			t.Errorf("%s dawg UnicharsetSize = %d; want 112", tc.name, tc.d.UnicharsetSize)
		}
	}
}

// Words verified against the real lstm-word-dawg during planning by walking the
// edge array directly. If one of these fails, suspect the unichar-id mapping in
// the test helper before suspecting the walk.
func TestDawgContainsRealWords(t *testing.T) {
	_, word, _, u := loadRealDawgs(t)

	byText := map[string]int{}
	for id := range u.Size() {
		byText[u.Text(id)] = id
	}
	ids := func(s string) []int {
		out := make([]int, 0, len(s))
		for _, r := range s {
			id, ok := byText[string(r)]
			if !ok {
				t.Fatalf("%q: no unichar id for %q", s, string(r))
			}
			out = append(out, id)
		}
		return out
	}

	for _, w := range []string{"the", "and", "zebra", "hello", "Cadmus", "a", "I", "of"} {
		if !word.Contains(ids(w)) {
			t.Errorf("Contains(%q) = false; want true", w)
		}
	}
	for _, w := range []string{"qqqqq", "tesseract"} {
		if word.Contains(ids(w)) {
			t.Errorf("Contains(%q) = true; want false", w)
		}
	}
	if word.Contains(nil) {
		t.Error("Contains(nil) = true; want false")
	}
	// A prefix of a word that is not itself a word, which is what separates
	// HasPrefix from Contains. "zebr" leads to zebra, zebrafish and zebras.
	//
	// The plan named "th" here, but "th" IS a word of this lstm-word-dawg:
	// Tesseract's own dawg2wordlist dumps 338,080 words from these exact bytes
	// and "th" is one of them, while "zebr" is not. The assertion was wrong
	// about the model, not about the walk — Contains agrees with that dump on
	// every one of the 338,080 words.
	if !word.HasPrefix(ids("zebr")) {
		t.Error("HasPrefix(\"zebr\") = false; want true")
	}
	if word.Contains(ids("zebr")) {
		t.Error("Contains(\"zebr\") = true; want false")
	}
}

// Dawg::init uses CeilLog2, a BIT LENGTH. network.go's ceilLog2 is
// src/lstm/lstm.cpp's ceil_log2, a genuine ceil(log2). They differ at exact
// powers of two and confusing them shifts every mask by one bit.
func TestDawgFlagStartBitIsBitLength(t *testing.T) {
	for _, tc := range []struct{ size, want int }{
		{1, 1}, {63, 6}, {64, 7}, {112, 7}, {128, 8},
	} {
		d := &Dawg{UnicharsetSize: tc.size}
		d.initMasks()
		if int(d.flagStartBit) != tc.want {
			t.Errorf("flagStartBit for unicharset_size %d = %d; want %d", tc.size, d.flagStartBit, tc.want)
		}
	}
	if got := ceilLog2(64); got != 6 {
		t.Errorf("ceilLog2(64) = %d; want 6 — the LSTM one must stay ceil(log2)", got)
	}
}

// buildDawg emits the component header plus raw edge records.
func buildDawg(unicharsetSize int, edges []uint64) []byte {
	b := binary.LittleEndian.AppendUint16(nil, 42)
	b = binary.LittleEndian.AppendUint32(b, uint32(unicharsetSize))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(edges)))
	for _, e := range edges {
		b = binary.LittleEndian.AppendUint64(b, e)
	}
	return b
}

// makeEdge packs one record for a 112-entry unicharset (flag_start_bit 7).
func makeEdge(id int, last, eow bool, next int) uint64 {
	e := uint64(id)
	if last {
		e |= 1 << 7
	}
	if eow {
		e |= 4 << 7
	}
	return e | uint64(next)<<10
}

func TestParseDawgTinyGraph(t *testing.T) {
	// Node 0: 'a'(id 1) -> node 2, last. Node 1 unused filler.
	// Node 2: 'b'(id 2), eow, last.
	edges := []uint64{
		makeEdge(1, true, false, 2),
		makeEdge(9, true, false, 0),
		makeEdge(2, true, true, 0),
	}
	d, err := ParseDawg(buildDawg(112, edges), false)
	if err != nil {
		t.Fatalf("ParseDawg() error = %v", err)
	}
	if !d.Contains([]int{1, 2}) {
		t.Error("Contains([1 2]) = false; want true")
	}
	if d.Contains([]int{1}) {
		t.Error("Contains([1]) = true; want false — 'a' alone is not word-end")
	}
	if !d.HasPrefix([]int{1}) {
		t.Error("HasPrefix([1]) = false; want true")
	}
	if d.Contains([]int{1, 3}) {
		t.Error("Contains([1 3]) = true; want false")
	}
}

func TestParseDawgErrors(t *testing.T) {
	good := []uint64{makeEdge(1, true, true, 0)}

	bad := buildDawg(112, good)
	bad[0] = 43 // wrong magic
	if _, err := ParseDawg(bad, false); err == nil {
		t.Error("ParseDawg() with magic 43: want error, got nil")
	}

	if _, err := ParseDawg(buildDawg(112, nil), false); err == nil {
		t.Error("ParseDawg() with 0 edges: want error, got nil")
	}

	if _, err := ParseDawg(buildDawg(0, good), false); err == nil {
		t.Error("ParseDawg() with unicharset_size 0: want error, got nil")
	}

	if _, err := ParseDawg(append(buildDawg(112, good), 0xde), false); err == nil {
		t.Error("ParseDawg() with a trailing byte: want error, got nil")
	}

	truncated := buildDawg(112, good)
	if _, err := ParseDawg(truncated[:len(truncated)-2], false); err == nil {
		t.Error("ParseDawg() with a truncated edge: want error, got nil")
	}

	// A backward edge cannot appear in a written dawg.
	backward := []uint64{makeEdge(1, true, true, 0) | 2<<7}
	if _, err := ParseDawg(buildDawg(112, backward), false); err == nil {
		t.Error("ParseDawg() with a backward edge: want error, got nil")
	}

	// next_node beyond the edge array.
	if _, err := ParseDawg(buildDawg(112, []uint64{makeEdge(1, true, true, 99)}), false); err == nil {
		t.Error("ParseDawg() with an out-of-range next_node: want error, got nil")
	}

	// unichar_id at or beyond unicharset_size. Size 1 (not 4) is deliberate:
	// makeEdge hardcodes flag_start_bit = 7, so with size 4 (bit length 3,
	// letter mask 0x7) the id reads back as 1 < 4 and it is the NEXT_NODE check
	// that fires, not the id check. With size 1 the mask is 0x1, the id reads
	// back as 1 >= 1, and the assertion tests what it claims to test.
	if _, err := ParseDawg(buildDawg(1, []uint64{makeEdge(1, true, true, 0)}), false); err == nil {
		t.Error("ParseDawg() with unichar_id >= unicharset_size: want error, got nil")
	}
}

// Node 0 is the ONLY node Tesseract binary-searches (SquishedDawg::edge_char_of,
// src/dict/dawg.cpp:207); every other node it linear-scans. edgeOf linear-scans
// everywhere, which is equivalent only while node 0's edges carry no repeated
// unichar id — the binary search's comparator orders on
// (unichar_id, next_node, word_end), so the format permits two node-0 edges with
// the same id (one with WERD_END clear, one set), and on such a node the two
// searches can return different edges and therefore different next_nodes.
//
// Only the word dawg's node 0 was measured during planning (67 edges, sorted,
// no duplicates). Punctuation dawgs are exactly where duplicate ids are most
// likely, so this test measures all three. If it ever fails, do NOT relax it:
// implement the binary search for node 0 in edgeOf and fix its comment.
//
// node0UnicharIDs lives here rather than in dawg.go because nothing outside
// this test uses it; the test file is in package tessdata and so reaches the
// unexported fields directly.
func (d *Dawg) node0UnicharIDs() []int {
	var ids []int
	for e := 0; e < len(d.edges); e++ {
		ids = append(ids, int(d.edges[e]&d.letterMask))
		if d.edges[e]&(dawgMarkerFlag<<d.flagStartBit) != 0 {
			break
		}
	}
	return ids
}

func TestDawgNode0IsSortedAndDuplicateFree(t *testing.T) {
	punc, word, number, _ := loadRealDawgs(t)
	for _, tc := range []struct {
		name string
		d    *Dawg
	}{
		{"punc", punc},
		{"word", word},
		{"number", number},
	} {
		ids := tc.d.node0UnicharIDs()
		t.Logf("%s dawg node 0: %d edges", tc.name, len(ids))
		if len(ids) == 0 {
			t.Errorf("%s dawg node 0 has no edges", tc.name)
			continue
		}
		for i := 1; i < len(ids); i++ {
			if ids[i] < ids[i-1] {
				t.Errorf("%s dawg node 0 edge %d id %d < edge %d id %d; node 0 must be sorted ascending for the binary search edgeOf replaces",
					tc.name, i, ids[i], i-1, ids[i-1])
			}
			if ids[i] == ids[i-1] {
				t.Errorf("%s dawg node 0 has duplicate unichar id %d at edges %d and %d; a linear scan is NOT equivalent to Tesseract's binary search here",
					tc.name, ids[i], i-1, i)
			}
		}
	}
	if got := len(word.node0UnicharIDs()); got != 67 {
		t.Errorf("word dawg node 0 has %d edges; want 67 (measured during planning)", got)
	}
}
