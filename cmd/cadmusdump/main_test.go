package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	return path
}

func TestDumpDefaultSections(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"lstm-unicharset",
		"lstm-recoder",
		"null_char      110",
		"sample_iter    814136",
		"[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]",
		`Series "Series" ni=36 no=111 weights=1461007`,
		`Softmax "Output" ni=512 no=111 weights=56943`,
		"111x513 min=-29.279424 max=35.155323",
		`LSTM "Lfx512" ni=96 no=512 na=608`,
		"[2] GF1 512x609",
		"half_x=1 half_y=1",
		"x_scale=3 y_scale=3",
		"shape batch=1 height=36 width=0 depth=1 loss=0",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("dump output missing %q\n---\n%s", want, out)
		}
	}
	// Opt-in sections must stay off by default.
	for _, notWant := range []string{"unicharset:", "recoder:", "dawgs:"} {
		if strings.Contains(out, notWant) {
			t.Errorf("dump output contains %q without the flag", notWant)
		}
	}
}

func TestDumpUnicharsetSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{unicharset: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"unicharset: 112 entries",
		`    0 " "`,
		`    2 "|Broken|0|1"`,
		"props=f",
		"normed=\"'\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("unicharset section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpRecoderSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{recoder: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"recoder: 112 entries, code range 111, max code length 1",
		// dumpRecoder prints the code SLICE with %v, so the rendering is
		// "-> code [110]", not "-> code 110".
		"-> code [110]",
		"BLANK",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("recoder section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpDawgSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{dawgs: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{"dawgs:", "word", "461848 edges"} {
		if !strings.Contains(out, want) {
			t.Errorf("dawg section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpMissingFileErrors(t *testing.T) {
	var b strings.Builder
	if err := dump("does-not-exist.traineddata", options{}, &b); err == nil {
		t.Fatal("dump() on missing file: want error, got nil")
	}
}
