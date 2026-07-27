package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// buildContainer produces a synthetic .traineddata in the native format.
// offsets of -1 mark absent entries.
func buildContainer(t *testing.T, entries map[Type][]byte) []byte {
	t.Helper()
	const n = int(numTypes)
	header := 4 + 8*n
	offsets := make([]int64, n)
	var body []byte
	for i := range offsets {
		payload, ok := entries[Type(i)]
		if !ok {
			offsets[i] = -1
			continue
		}
		offsets[i] = int64(header + len(body))
		body = append(body, payload...)
	}
	out := make([]byte, 0, header+len(body))
	out = binary.LittleEndian.AppendUint32(out, uint32(n))
	for _, off := range offsets {
		out = binary.LittleEndian.AppendUint64(out, uint64(off))
	}
	return append(out, body...)
}

func TestParseContainerRoundTrip(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{
		TypeLSTM:           []byte("network-bytes"),
		TypeLSTMUnicharset: []byte("charset"),
		TypeVersion:        []byte("5.0.0-test"),
	})

	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}

	got, ok := c.Entry(TypeLSTM)
	if !ok || string(got) != "network-bytes" {
		t.Fatalf("Entry(TypeLSTM) = %q, %v; want \"network-bytes\", true", got, ok)
	}
	if got, ok := c.Entry(TypeLSTMUnicharset); !ok || string(got) != "charset" {
		t.Fatalf("Entry(TypeLSTMUnicharset) = %q, %v; want \"charset\", true", got, ok)
	}
	if c.Version() != "5.0.0-test" {
		t.Fatalf("Version() = %q; want \"5.0.0-test\"", c.Version())
	}
}

func TestParseContainerAbsentEntry(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{TypeLSTM: []byte("x")})
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	if _, ok := c.Entry(TypeAmbigs); ok {
		t.Fatal("Entry(TypeAmbigs) reported present; want absent")
	}
}

// The last present entry runs to end-of-file rather than to a following offset.
func TestParseContainerLastEntryRunsToEOF(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{TypeVersion: []byte("tail")})
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	if got, _ := c.Entry(TypeVersion); string(got) != "tail" {
		t.Fatalf("Entry(TypeVersion) = %q; want \"tail\"", got)
	}
}

func TestParseContainerRejectsTooManyEntries(t *testing.T) {
	// 2000 exceeds kMaxNumTessdataEntries in both byte orders, so it is
	// corruption rather than an endianness signal.
	raw := binary.LittleEndian.AppendUint32(nil, 2000)
	if _, err := ParseContainer(raw); err == nil {
		t.Fatal("ParseContainer() with 2000 entries: want error, got nil")
	}
}

// Real-model check. Skipped when the fixture has not been fetched, so the
// suite still passes on a machine without ./testdata/fetch.sh having been run.
func TestParseContainerRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer(eng.traineddata) error = %v", err)
	}
	for _, want := range []Type{TypeLSTM, TypeLSTMUnicharset, TypeLSTMRecoder} {
		if _, ok := c.Entry(want); !ok {
			t.Errorf("Entry(%v) absent from eng.traineddata; want present", want)
		}
	}
	if c.Version() == "" {
		t.Error("Version() is empty; want a version string")
	}
	t.Logf("eng.traineddata components: %v, version %q", c.Present(), c.Version())
}
