package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func loadRealRecoder(t *testing.T) *Recoder {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, ok := c.Entry(TypeLSTMRecoder)
	if !ok {
		t.Fatal("eng.traineddata has no lstm-recoder component")
	}
	rc, err := ParseRecoder(b, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	return rc
}

func TestParseRecoderRealModel(t *testing.T) {
	rc := loadRealRecoder(t)

	if rc.Size() != 112 {
		t.Fatalf("Size() = %d; want 112 (one entry per unichar id)", rc.Size())
	}
	if rc.CodeRange() != 111 {
		t.Fatalf("CodeRange() = %d; want 111", rc.CodeRange())
	}
	// eng's recoder is a near-permutation: every code is a single value, which
	// is why greedy CTC decoding is viable for Latin at all. A model where this
	// is false needs the multi-code lookahead in LSTMRecognizer::DecodeLabel.
	if rc.MaxCodeLen() != 1 {
		t.Errorf("MaxCodeLen() = %d; want 1 for eng", rc.MaxCodeLen())
	}

	// LSTMRecognizer::LoadRecoder's own check.
	if got := rc.Encode(UnicharSpace); len(got) != 1 || got[0] != 0 {
		t.Errorf("Encode(UnicharSpace) = %v; want [0]", got)
	}

	for _, tc := range []struct {
		code []int32
		id   int
	}{
		{[]int32{0}, UnicharSpace},
		{[]int32{1}, 3},     // 'C'
		{[]int32{2}, 4},     // 'H'
		{[]int32{109}, 111}, // 'e' with acute
		// Ids 1 (Joined) and 2 (|Broken|0|1) BOTH encode to 110, and
		// SetupDecoder iterates ids ascending and overwrites, so the higher id
		// wins. 110 is also null_char_.
		{[]int32{110}, UnicharBroken},
	} {
		got, ok := rc.DecodeUnichar(tc.code)
		if !ok || got != tc.id {
			t.Errorf("DecodeUnichar(%v) = %d, %v; want %d, true", tc.code, got, ok, tc.id)
		}
	}
	if _, ok := rc.DecodeUnichar([]int32{999}); ok {
		t.Error("DecodeUnichar([999]) reported a hit; want miss")
	}
	if _, ok := rc.DecodeUnichar(nil); ok {
		t.Error("DecodeUnichar(nil) reported a hit; want miss")
	}

	// Every code value in [0, 111) is a valid first code, because all codes
	// have length 1 and every value is used.
	for c := range int32(111) {
		if !rc.IsValidFirstCode(c) {
			t.Errorf("IsValidFirstCode(%d) = false; want true", c)
		}
	}
	if rc.IsValidFirstCode(111) {
		t.Error("IsValidFirstCode(111) = true; want false (outside the code range)")
	}

	// Both eng models ship a byte-identical recoder; the encoder must be a
	// total function over the unicharset.
	for id := range rc.Size() {
		if len(rc.Encode(id)) == 0 {
			t.Errorf("Encode(%d) is empty", id)
		}
	}
}

// buildRecoder emits the component format: uint32 count, then per entry
// int8 self_normalized, uint32 length, int32 code[length].
func buildRecoder(codes [][]int32) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(len(codes)))
	for _, code := range codes {
		b = append(b, 1) // self_normalized
		b = binary.LittleEndian.AppendUint32(b, uint32(len(code)))
		for _, v := range code {
			b = binary.LittleEndian.AppendUint32(b, uint32(v))
		}
	}
	return b
}

func TestParseRecoderHigherIDWinsOnCollision(t *testing.T) {
	// ids 0,1,2 -> codes 0,5,5. SetupDecoder overwrites in ascending id order.
	rc, err := ParseRecoder(buildRecoder([][]int32{{0}, {5}, {5}}), false)
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	got, ok := rc.DecodeUnichar([]int32{5})
	if !ok || got != 2 {
		t.Errorf("DecodeUnichar([5]) = %d, %v; want 2, true", got, ok)
	}
	if rc.CodeRange() != 6 {
		t.Errorf("CodeRange() = %d; want 6", rc.CodeRange())
	}
}

func TestParseRecoderMultiCode(t *testing.T) {
	// Untested against a real CJK model — this exercises the multi-code layout
	// the format allows but eng never uses.
	rc, err := ParseRecoder(buildRecoder([][]int32{{0}, {1, 2, 3}}), false)
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	if rc.MaxCodeLen() != 3 {
		t.Errorf("MaxCodeLen() = %d; want 3", rc.MaxCodeLen())
	}
	if got, ok := rc.DecodeUnichar([]int32{1, 2, 3}); !ok || got != 1 {
		t.Errorf("DecodeUnichar([1 2 3]) = %d, %v; want 1, true", got, ok)
	}
	if _, ok := rc.DecodeUnichar([]int32{1, 2}); ok {
		t.Error("DecodeUnichar([1 2]) reported a hit on a proper prefix; want miss")
	}
	if !rc.IsValidFirstCode(1) || rc.IsValidFirstCode(2) {
		t.Error("IsValidFirstCode must be true only for code[0] values")
	}
}

func TestParseRecoderErrors(t *testing.T) {
	tooLong := binary.LittleEndian.AppendUint32(nil, 1)
	tooLong = append(tooLong, 1)
	tooLong = binary.LittleEndian.AppendUint32(tooLong, 10) // > kMaxCodeLen 9
	if _, err := ParseRecoder(tooLong, false); err == nil {
		t.Error("ParseRecoder() with length 10: want error, got nil")
	}

	trailing := append(buildRecoder([][]int32{{0}}), 0xde, 0xad)
	if _, err := ParseRecoder(trailing, false); err == nil {
		t.Error("ParseRecoder() with trailing bytes: want error, got nil")
	}

	// Space must encode onto itself, or the whole decode is misaligned.
	if _, err := ParseRecoder(buildRecoder([][]int32{{7}, {0}}), false); err == nil {
		t.Error("ParseRecoder() with a garbled space: want error, got nil")
	}

	if _, err := ParseRecoder(buildRecoder([][]int32{{0}, {-1}}), false); err == nil {
		t.Error("ParseRecoder() with a negative code: want error, got nil")
	}

	if _, err := ParseRecoder([]byte{0x01}, false); err == nil {
		t.Error("ParseRecoder() on a truncated header: want error, got nil")
	}
}
