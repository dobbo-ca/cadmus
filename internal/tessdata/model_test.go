package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadRealModel(t *testing.T) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	m, err := LoadModel(raw)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	return m
}

// The L1a deliverable, asserted in one place.
func TestLoadModelRealModel(t *testing.T) {
	m := loadRealModel(t)

	if !strings.HasPrefix(m.Version, "4.00.00alpha:eng:") {
		t.Errorf("Version = %q; want a 4.00.00alpha:eng: prefix", m.Version)
	}
	if m.Swapped {
		t.Error("Swapped = true; the fixture is little-endian")
	}

	// The chain that binds the components together.
	if got := m.Unicharset.Size(); got != 112 {
		t.Errorf("Unicharset.Size() = %d; want 112", got)
	}
	if got := m.Recoder.Size(); got != m.Unicharset.Size() {
		t.Errorf("Recoder.Size() = %d; want %d (one entry per unichar id)", got, m.Unicharset.Size())
	}
	if got := m.Recoder.CodeRange(); got != 111 {
		t.Errorf("Recoder.CodeRange() = %d; want 111", got)
	}
	if got := m.NumOutputs(); got != m.Recoder.CodeRange() {
		t.Errorf("NumOutputs() = %d; want %d", got, m.Recoder.CodeRange())
	}
	if got := m.NullChar(); got != 110 {
		t.Errorf("NullChar() = %d; want 110", got)
	}

	sm := m.Softmax()
	if sm == nil {
		t.Fatal("Softmax() = nil")
	}
	if sm.Name != "Output" || sm.NumOutputs != 111 || sm.NumInputs != 512 {
		t.Errorf("Softmax = %q ni=%d no=%d; want \"Output\" ni=512 no=111", sm.Name, sm.NumInputs, sm.NumOutputs)
	}
	// Print SHAPES, never the matrices themselves: %v on sm.Matrices would dump
	// 56,943 float64s into the test log.
	if len(sm.Matrices) != 1 {
		t.Fatalf("Softmax has %d matrices; want one 111x513", len(sm.Matrices))
	}
	if sm.Matrices[0].Rows != 111 || sm.Matrices[0].Cols != 513 {
		t.Fatalf("Softmax matrix = %dx%d; want 111x513", sm.Matrices[0].Rows, sm.Matrices[0].Cols)
	}

	for _, tc := range []struct {
		name  string
		d     *Dawg
		edges int
	}{
		{"punc", m.PuncDawg, 539},
		{"word", m.WordDawg, 461848},
		{"number", m.NumberDawg, 591},
	} {
		if tc.d == nil {
			t.Errorf("%s dawg is nil; want loaded", tc.name)
			continue
		}
		if tc.d.NumEdges() != tc.edges {
			t.Errorf("%s dawg NumEdges() = %d; want %d", tc.name, tc.d.NumEdges(), tc.edges)
		}
	}

	// Every weight in the model is loaded. 1461007 is the root layer's own
	// num_weights, cross-checked recursively during parsing.
	total := 0
	var walk func(*Layer)
	walk = func(l *Layer) {
		for i := range l.Matrices {
			total += len(l.Matrices[i].Values)
		}
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(m.Recognizer.Network)
	if total != 1461007 {
		t.Errorf("loaded %d weight values; want 1461007", total)
	}
}

func TestLoadModelRejectsMissingComponents(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	// A container holding only the version string: every required component
	// is absent, so LoadModel must refuse rather than return a half-model.
	ver, _ := c.Entry(TypeVersion)
	only := buildContainer(t, map[Type][]byte{TypeVersion: ver})
	if _, err := LoadModel(only); err == nil {
		t.Fatal("LoadModel() without an lstm component: want error, got nil")
	}
}

func TestLoadModelRejectsGarbage(t *testing.T) {
	if _, err := LoadModel([]byte{0xde, 0xad, 0xbe, 0xef}); err == nil {
		t.Fatal("LoadModel() on garbage: want error, got nil")
	}
}
