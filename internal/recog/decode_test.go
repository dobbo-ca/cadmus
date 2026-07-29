package recog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// testing.TB rather than *testing.T, so Task 24's benchmark reuses this helper
// rather than a duplicate named loadRecognizerB.
func loadRecognizer(t testing.TB) *Recognizer {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	r, err := NewRecognizer(raw)
	if err != nil {
		t.Fatalf("NewRecognizer() error = %v", err)
	}
	return r
}

// oneHot builds a synthetic softmax output that puts all the mass on the given
// code at each timestep, so the collapse rule can be tested in isolation.
func oneHot(codes []int, n int) *nn.Tensor {
	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: len(codes)}, n)
	row := make([]float64, n)
	for t, c := range codes {
		for i := range row {
			row[i] = 0.001
		}
		row[c] = 0.9
		out.WriteTimeStep(t, row)
	}
	return out
}

func TestGreedyDecodeCollapsesRepeatsAndDropsBlanks(t *testing.T) {
	r := loadRecognizer(t)
	null := r.Net.NullChar
	// codes: A A blank A blank blank B  ->  A A B
	// (the blank between the two A runs is what keeps them separate)
	codeA := codeFor(t, r, "A")
	codeB := codeFor(t, r, "B")
	syms, err := r.GreedyDecode(oneHot([]int{codeA, codeA, null, codeA, null, null, codeB}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	var got strings.Builder
	for _, s := range syms {
		got.WriteString(s.Text)
	}
	if got.String() != "AAB" {
		t.Fatalf("decoded %q; want \"AAB\"", got.String())
	}
	// The run boundaries must be the argmax runs, not the emission points.
	if syms[0].Start != 0 || syms[0].End != 2 {
		t.Errorf("first symbol run = [%d,%d); want [0,2)", syms[0].Start, syms[0].End)
	}
	if syms[1].Start != 3 || syms[1].End != 4 {
		t.Errorf("second symbol run = [%d,%d); want [3,4)", syms[1].Start, syms[1].End)
	}
	if syms[2].Start != 6 || syms[2].End != 7 {
		t.Errorf("third symbol run = [%d,%d); want [6,7)", syms[2].Start, syms[2].End)
	}
}

func TestGreedyDecodeAllBlank(t *testing.T) {
	r := loadRecognizer(t)
	syms, err := r.GreedyDecode(oneHot([]int{r.Net.NullChar, r.Net.NullChar}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	if len(syms) != 0 {
		t.Errorf("all-blank decode produced %d symbols; want 0", len(syms))
	}
}

// The raw token, not the normalized form: `tesseract` emits U+2019, not "'".
func TestGreedyDecodeEmitsTheRawToken(t *testing.T) {
	r := loadRecognizer(t)
	id := unicharIDFor(t, r, "’")
	if r.Charset.Normed(id) == r.Charset.Text(id) {
		t.Skip("this model does not normalize the right single quote")
	}
	syms, err := r.GreedyDecode(oneHot([]int{codeFor(t, r, "’")}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	if syms[0].Text != "’" {
		t.Errorf("decoded %q; want the raw token U+2019, not the normed form", syms[0].Text)
	}
}

func unicharIDFor(t testing.TB, r *Recognizer, s string) int {
	t.Helper()
	for id := range r.Charset.Size() {
		if r.Charset.Text(id) == s {
			return id
		}
	}
	t.Fatalf("no unichar %q in the charset", s)
	return -1
}

// codeFor is the network output index for a character. Recoder.Encode is a
// method returning []int32 (L1a Task 6), and L1b has already asserted every
// entry is length 1, so index 0 is the whole code.
func codeFor(t testing.TB, r *Recognizer, s string) int {
	t.Helper()
	codes := r.Recoder.Encode(unicharIDFor(t, r, s))
	if len(codes) != 1 {
		t.Fatalf("unichar %q encodes to %d codes; want 1", s, len(codes))
	}
	return int(codes[0])
}

// The acceptance test. Cadmus's text must match the committed
// `tesseract --psm 13` output on the corpus arm where neither side resamples.
func TestGreedyDecodeMatchesOracleH36(t *testing.T) {
	r := loadRecognizer(t)
	lines := loadCorpus(t, "h36")

	var exact int
	var totalCER float64
	for _, l := range lines {
		got, err := r.RecognizeText(l.Image)
		if err != nil {
			t.Errorf("%s: RecognizeText() error = %v", l.Name, err)
			continue
		}
		if got == l.Oracle {
			exact++
		} else {
			t.Logf("%s\n  oracle: %q\n  cadmus: %q\n  cer:    %.4f", l.Name, l.Oracle, got, cer(got, l.Oracle))
		}
		totalCER += cer(got, l.Oracle)
	}
	meanCER := totalCER / float64(len(lines))
	t.Logf("h36 arm: %d/%d exact, mean CER vs oracle %.4f", exact, len(lines), meanCER)

	// Greedy decoding has no dictionary, so it will not match the oracle
	// everywhere; the bound is what a correct forward pass with no lexicon can
	// be expected to reach on clean rendered text. See Step 5 for what to do
	// when this fails.
	if meanCER > 0.02 {
		t.Errorf("mean CER vs oracle = %.4f; want <= 0.02", meanCER)
	}
}
