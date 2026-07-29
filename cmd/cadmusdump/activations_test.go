package main

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestDumpActivations(t *testing.T) {
	model := filepath.Join("..", "..", "testdata", "eng.traineddata")
	line := filepath.Join("..", "..", "testdata", "lines", "h36", "0001.png")
	for _, p := range []string{model, line} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("fixture not present: %v", err)
		}
	}
	var b strings.Builder
	if err := dumpActivations(model, line, &b); err != nil {
		t.Fatalf("dumpActivations() error = %v", err)
	}
	out := b.String()
	// One block per layer, keyed on the serialized layer name, so the blocks
	// line up with Tesseract's own DEBUG_DETAIL output. "Normalized" is the
	// pre-network input; the Input LAYER is not tapped, because its Name() is
	// "Input" and two identical headers would break the diff protocol.
	for _, want := range []string{
		"Output:Normalized", "Output:Convolve", "Output:ConvNL", "Output:Maxpool",
		"Output:Lfys64", "Output:Lfx96", "Output:Lrx96", "Output:Lfx512",
		"Output:Output",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("activation dump missing block %q", want)
		}
	}
	if n := strings.Count(out, "Output:Input\n"); n != 0 {
		t.Errorf("dump contains %d \"Output:Input\" headers; want 0 — the Input layer must not be tapped", n)
	}
	// Every value line must parse as floats. Checking for stray letters cannot
	// work: %g never emits letters for a finite value, so the only thing such a
	// check could ever catch is NaN/Inf, which contain no lowercase letters
	// other than 'a' and 'n'. Parse instead.
	for i, ln := range strings.Split(out, "\n") {
		if ln == "" || strings.HasPrefix(ln, "Output:") {
			continue
		}
		for _, f := range strings.Fields(ln) {
			if _, err := strconv.ParseFloat(f, 64); err != nil {
				t.Fatalf("line %d: field %q is not a float: %v", i+1, f, err)
			}
		}
	}
}

// TestDumpActivationsBlockShapes pins the one property the differential
// protocol actually rests on: that each block carries exactly NumOutputs rows
// of exactly Len() timesteps, in graph order, under headers that are unique.
// Step 6 compares Cadmus and Tesseract block by block with `wc -l` and a
// column-wise max-abs-delta, and both of those silently produce nonsense if a
// block's shape is wrong or a header repeats.
//
// Unlike TestDumpActivations it needs no line corpus, so it runs today rather
// than after Task 17. The image is 36 rows tall, which makes Normalize's scale
// factor exactly 1 and the timestep counts below arithmetic rather than
// resampling-dependent: 36x60 input, x and y reduced by the Mp3,3 maxpool to
// 12x20, then Lfys64's width-to-1 summary over the transposed map leaving 20.
func TestDumpActivationsBlockShapes(t *testing.T) {
	model := fixture(t)
	line := filepath.Join(t.TempDir(), "line.png")
	writeTestLine(t, line, 60, 36)

	var b strings.Builder
	if err := dumpActivations(model, line, &b); err != nil {
		t.Fatalf("dumpActivations() error = %v", err)
	}

	want := []struct {
		name   string
		rows   int // the layer's NumOutputs
		fields int // the layer's timestep count
	}{
		{"Normalized", 1, 2160},
		{"Convolve", 9, 2160},
		{"ConvNL", 16, 2160},
		{"ConvSeries", 16, 2160},
		{"Maxpool", 16, 240},
		{"Lfys64", 64, 20},
		{"XYTransLSTM", 64, 20},
		{"Lfx96", 96, 20},
		{"Lrx96", 96, 20},
		{"RevLSTM", 96, 20},
		{"Lfx512", 512, 20},
		{"Output", 111, 20},
		{"Series", 111, 20},
	}

	blocks := parseBlocks(t, b.String())
	if len(blocks) != len(want) {
		t.Fatalf("dump has %d blocks; want %d\ngot %v", len(blocks), len(want), blockNames(blocks))
	}
	seen := map[string]bool{}
	for i, w := range want {
		got := blocks[i]
		if got.name != w.name {
			t.Errorf("block %d is %q; want %q", i, got.name, w.name)
			continue
		}
		if seen[got.name] {
			t.Errorf("block %d repeats header %q; block-keyed diffs need unique headers", i, got.name)
		}
		seen[got.name] = true
		if len(got.rows) != w.rows {
			t.Errorf("block %q has %d rows; want %d", w.name, len(got.rows), w.rows)
		}
		for r, n := range got.rows {
			if n != w.fields {
				t.Errorf("block %q row %d has %d timesteps; want %d", w.name, r, n, w.fields)
			}
		}
	}
}

type block struct {
	name string
	rows []int // field count of each value line
}

// parseBlocks splits the dump on its "Output:<name>" headers, which is exactly
// what the Step 6 sed range does.
func parseBlocks(t *testing.T, out string) []block {
	t.Helper()
	var blocks []block
	for _, ln := range strings.Split(strings.TrimSuffix(out, "\n"), "\n") {
		if name, ok := strings.CutPrefix(ln, "Output:"); ok {
			blocks = append(blocks, block{name: name})
			continue
		}
		if len(blocks) == 0 {
			t.Fatalf("value line %q precedes the first header", ln)
		}
		last := &blocks[len(blocks)-1]
		last.rows = append(last.rows, len(strings.Fields(ln)))
	}
	return blocks
}

func blockNames(blocks []block) []string {
	names := make([]string, len(blocks))
	for i, b := range blocks {
		names[i] = b.name
	}
	return names
}

// writeTestLine writes an 8bpp line image with real ink: a light background and
// dark vertical strokes, so ComputeBlackWhite finds genuine minima and maxima
// instead of falling back on its 0/255 defaults.
func writeTestLine(t *testing.T, path string, w, h int) {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.SetGray(x, y, color.Gray{Y: 235})
		}
	}
	for x0 := 2; x0+2 < w; x0 += 7 {
		for y := h / 4; y < 3*h/4; y++ {
			img.SetGray(x0, y, color.Gray{Y: 30})
			img.SetGray(x0+1, y, color.Gray{Y: 30})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("creating %s: %v", path, err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatalf("encoding %s: %v", path, err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("closing %s: %v", path, err)
	}
}
