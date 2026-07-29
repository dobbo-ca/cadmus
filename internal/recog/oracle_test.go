package recog

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// corpusLine is one committed acceptance case.
type corpusLine struct {
	Name   string
	Image  image.Image
	Truth  string // the text that was rendered
	Oracle string // what `tesseract --psm 13` produced
}

// loadCorpus reads one arm of testdata/lines. Arms are "h36" (already 36 px
// tall, so the scaler is an identity on both sides) and "native".
//
// The parameter is testing.TB, not *testing.T, so Task 24's benchmark can call
// this same helper instead of a near-identical copy.
func loadCorpus(t testing.TB, arm string) []corpusLine {
	t.Helper()
	dir := filepath.Join("..", "..", "testdata", "lines", arm)
	pngs, err := filepath.Glob(filepath.Join(dir, "*.png"))
	if err != nil || len(pngs) == 0 {
		t.Skipf("corpus arm %q not present (run ./testdata/lines/gen.sh): %v", arm, err)
	}
	var out []corpusLine
	for _, p := range pngs {
		base := strings.TrimSuffix(p, ".png")
		f, err := os.Open(p)
		if err != nil {
			t.Fatalf("opening %s: %v", p, err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("decoding %s: %v", p, err)
		}
		truth, err := os.ReadFile(base + ".gt.txt")
		if err != nil {
			t.Fatalf("reading ground truth for %s: %v", p, err)
		}
		oracle, err := os.ReadFile(base + ".psm13.txt")
		if err != nil {
			t.Fatalf("reading oracle for %s: %v", p, err)
		}
		out = append(out, corpusLine{
			Name:   filepath.Base(base),
			Image:  img,
			Truth:  strings.TrimRight(string(truth), "\n"),
			Oracle: strings.TrimRight(string(oracle), "\n"),
		})
	}
	return out
}

// cer is the Levenshtein character error rate of got against want.
func cer(got, want string) float64 {
	a, b := []rune(want), []rune(got)
	if len(a) == 0 {
		if len(b) == 0 {
			return 0
		}
		return 1
	}
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min(min(curr[j-1]+1, prev[j]+1), prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return float64(prev[len(b)]) / float64(len(a))
}

func TestCorpusIsPresentAndConsistent(t *testing.T) {
	for _, arm := range []string{"h36", "native"} {
		lines := loadCorpus(t, arm)
		if len(lines) < 10 {
			t.Errorf("arm %q has %d lines; want at least 10", arm, len(lines))
		}
		for _, l := range lines {
			if l.Oracle == "" {
				t.Errorf("%s/%s: oracle output is empty", arm, l.Name)
			}
			if arm == "h36" && l.Image.Bounds().Dy() != 36 {
				t.Errorf("h36/%s is %d px tall; want 36", l.Name, l.Image.Bounds().Dy())
			}
		}
	}
}
