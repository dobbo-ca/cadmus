package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDumpRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	var b strings.Builder
	if err := dump(path, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{"lstm", "lstm-unicharset", "Series"} {
		if !strings.Contains(out, want) {
			t.Errorf("dump output missing %q\n%s", want, out)
		}
	}
}

func TestDumpMissingFileErrors(t *testing.T) {
	var b strings.Builder
	if err := dump("does-not-exist.traineddata", &b); err == nil {
		t.Fatal("dump() on missing file: want error, got nil")
	}
}
