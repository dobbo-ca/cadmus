package recog

import (
	"image"
	"math"
	"strings"
	"testing"
)

func TestConfidencePercent(t *testing.T) {
	// A dictionary-clean word with every timestep at p >= 0.99 has certainty
	// about -0.095 and reports about 96.7%.
	if got := ConfidencePercent(-0.095); math.Abs(got-96.675) > 0.01 {
		t.Errorf("ConfidencePercent(-0.095) = %v; want ~96.68", got)
	}
	// Certainty 0 is a perfect 100, and anything below -100/35 clips to 0.
	if got := ConfidencePercent(0); got != 100 {
		t.Errorf("ConfidencePercent(0) = %v; want 100", got)
	}
	if got := ConfidencePercent(-10); got != 0 {
		t.Errorf("ConfidencePercent(-10) = %v; want 0", got)
	}
}

func TestSplitWordsAtSpaces(t *testing.T) {
	syms := []Scored{
		{Symbol: Symbol{Text: "h", UnicharID: 3}, Certainty: -0.2},
		{Symbol: Symbol{Text: "i", UnicharID: 4}, Certainty: -0.1},
		{Symbol: Symbol{Text: " ", UnicharID: 0}, Certainty: -0.9},
		{Symbol: Symbol{Text: "t", UnicharID: 5}, Certainty: -0.3},
		{Symbol: Symbol{Text: "o", UnicharID: 6}, Certainty: -0.4},
	}
	groups := splitWords(syms)
	if len(groups) != 2 {
		t.Fatalf("splitWords returned %d groups; want 2", len(groups))
	}
	if groups[0].start != 0 || groups[0].end != 2 {
		t.Errorf("first group = [%d,%d); want [0,2)", groups[0].start, groups[0].end)
	}
	if groups[1].start != 3 || groups[1].end != 5 {
		t.Errorf("second group = [%d,%d); want [3,5)", groups[1].start, groups[1].end)
	}
	// The space's certainty bounds both neighbours.
	if math.Abs(groups[0].spaceCert-(-0.9)) > 1e-12 {
		t.Errorf("first group space certainty = %v; want -0.9", groups[0].spaceCert)
	}
}

func TestWordBoundsScaleFromTimesteps(t *testing.T) {
	// bounds [0, 4, 10] with scale 6 and a line box starting at x=100:
	// word 0 spans timesteps [0,4) -> [100, 100+ceil(24)] = [100, 124]
	got := wordBounds([]int{0, 4, 10}, 0, 1, 6.0, image.Rect(100, 50, 400, 90))
	want := image.Rect(100, 50, 124, 90)
	if got != want {
		t.Errorf("wordBounds = %v; want %v", got, want)
	}
}

func TestRecognizeProducesWordsWithGeometry(t *testing.T) {
	r := loadRecognizer(t)
	lines := loadCorpus(t, "h36")
	for _, l := range lines {
		line, err := r.Recognize(l.Image)
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		if line.Text != l.Oracle {
			continue // text accuracy is Task 18's assertion, not this one
		}
		if len(line.Words) == 0 && line.Text != "" {
			t.Errorf("%s: text %q produced no words", l.Name, line.Text)
		}
		joined := make([]string, len(line.Words))
		for i, w := range line.Words {
			joined[i] = w.Text
			if w.Bounds.Empty() {
				t.Errorf("%s: word %q has an empty bounding box", l.Name, w.Text)
			}
			if !w.Bounds.In(line.Bounds) {
				t.Errorf("%s: word %q bounds %v escape the line bounds %v", l.Name, w.Text, w.Bounds, line.Bounds)
			}
			if w.Confidence < 0 || w.Confidence > 100 {
				t.Errorf("%s: word %q confidence %v out of range", l.Name, w.Text, w.Confidence)
			}
		}
		if got := strings.Join(joined, " "); got != strings.Join(strings.Fields(line.Text), " ") {
			t.Errorf("%s: words %q do not reconstruct the line %q", l.Name, got, line.Text)
		}
		// Words must be left to right and non-overlapping.
		for i := 1; i < len(line.Words); i++ {
			if line.Words[i].Bounds.Min.X < line.Words[i-1].Bounds.Max.X {
				t.Errorf("%s: word %d overlaps word %d", l.Name, i, i-1)
			}
		}
	}
}
