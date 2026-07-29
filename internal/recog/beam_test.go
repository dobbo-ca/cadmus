package recog

import (
	"strings"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// With no dictionary and unambiguous per-timestep distributions, the beam must
// agree with greedy exactly — that is the invariant that says the search
// machinery is not corrupting an already-correct path.
func TestBeamAgreesWithGreedyOnUnambiguousInput(t *testing.T) {
	r := loadRecognizer(t)
	r.Dict = nil
	for _, l := range loadCorpus(t, "h36") {
		out, _, err := r.Forward(l.Image)
		if err != nil {
			t.Fatalf("%s: Forward() error = %v", l.Name, err)
		}
		greedy, err := r.GreedyDecode(out)
		if err != nil {
			t.Fatalf("%s: GreedyDecode() error = %v", l.Name, err)
		}
		beam, err := r.BeamDecode(out)
		if err != nil {
			t.Fatalf("%s: BeamDecode() error = %v", l.Name, err)
		}
		var g, b strings.Builder
		for _, s := range greedy {
			g.WriteString(s.Text)
		}
		for _, s := range beam {
			b.WriteString(s.Text)
		}
		if g.String() != b.String() {
			t.Errorf("%s: dictionary-free beam %q differs from greedy %q", l.Name, b.String(), g.String())
		}
	}
}

// The score merge is the beam's main contribution to confidence: a class
// transition where the probability is split between a code and the blank must
// report a better certainty than greedy's per-timestep minimum.
//
// THREE TIMESTEPS. The merged push in ContinueContext's final_codes loop
// carries continuation NC_ONLY_DUP — "must be followed by a stand-alone
// duplicate of itself" — and ExtractBestPaths refuses NC_ONLY_DUP as a line
// ending. So a width-1 input cannot show the merge at all, and neither can a
// width-2 input whose second timestep is a blank: a blank is not a duplicate of
// 'A'. ContinueContext (recodebeam.cpp:906-927) gives an NC_ONLY_DUP
// predecessor exactly one continuation, a stand-alone duplicate of its own code
// scored ProbToCertainty(outputs[prev->code]), and then returns; with 'A' at
// 0.01/(n-1) that duplicate scores -21.1 after the dict ratio and
// PushDupOrNoDawgIfBetter (recodebeam.cpp:1174-1181) drops it for falling below
// kMinCertainty. The merged node would die with no successor and the unmerged
// NC_ANYTHING node would win, so a width-2 test measures the opposite of what it
// claims to.
//
// The layout below is Tesseract's own worked example (recodebeam.h:39-67): the
// split at one timestep is followed by a stand-alone at the next.
//
// The assertion is a BRACKET, not an equality. Which continuation survives to
// the end of the path depends on the heap ordering, and pinning an exact value
// here without measuring it first is how a plan-invented number gets "fixed" by
// loosening the tolerance. The bracket is tight enough to fail loudly if the
// merge is missing and loose enough not to lie.
func TestBeamMergesTransitionScores(t *testing.T) {
	r := loadRecognizer(t)
	r.Dict = nil
	n := r.Net.NumOutputs
	null := r.Net.NullChar
	code := codeFor(t, r, "A")

	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: 3}, n)
	row := make([]float64, n)

	// t=0: the mass is split 0.55 / 0.40 between 'A' and the blank.
	for i := range row {
		row[i] = 0.05 / float64(n-2)
	}
	row[code] = 0.55
	row[null] = 0.40
	out.WriteTimeStep(0, row)

	// t=1: a stand-alone 'A' — the duplicate the NC_ONLY_DUP node owes, and the
	// only thing that lets the merged hypothesis survive the step.
	for i := range row {
		row[i] = 0.01 / float64(n-2)
	}
	row[code] = 0.98
	row[null] = 0.01
	out.WriteTimeStep(1, row)

	// t=2: an unambiguous blank, so the path collapses to a single "A".
	for i := range row {
		row[i] = 0.02 / float64(n-1)
	}
	row[null] = 0.98
	out.WriteTimeStep(2, row)

	beam, err := r.BeamDecode(out)
	if err != nil {
		t.Fatalf("BeamDecode() error = %v", err)
	}
	if len(beam) != 1 || beam[0].Text != "A" {
		t.Fatalf("beam decoded %v; want a single \"A\"", beam)
	}

	// The tensor stores float32, so these are the probabilities the beam
	// actually reads. float32(0.55)+float32(0.40) sits 3e-8 above float32(0.95),
	// which a 1e-9 bracket cannot absorb — so the bound is written as the sum
	// that is really computed rather than as the literal 0.95.
	cert := func(p float64) float64 { return (ProbToCertainty(p) + CertOffset) * DictRatio }
	unmerged := cert(float64(float32(0.55)))                        // greedy's per-timestep minimum
	merged := cert(float64(float32(0.55)) + float64(float32(0.40))) // log(0.55 + 0.40)
	if beam[0].Certainty <= unmerged+1e-9 {
		t.Errorf("beam certainty %v is no better than the per-timestep minimum %v; the P(code)+P(blank) merge is missing",
			beam[0].Certainty, unmerged)
	}
	if beam[0].Certainty > merged+1e-9 {
		t.Errorf("beam certainty %v is better than log(0.55+0.40) = %v; the merge is double-counting a timestep",
			beam[0].Certainty, merged)
	}
	t.Logf("merged certainty %v, bracket (%v, %v]", beam[0].Certainty, unmerged, merged)
}

// With the dictionary on, a dictionary word must not be scaled by the dict
// ratio and so must report a better confidence than the same characters would
// as a non-word.
func TestBeamPrefersTheDictionaryPath(t *testing.T) {
	r := loadRecognizer(t)
	if r.Dict == nil {
		t.Skip("model has no dawg components")
	}
	lines := loadCorpus(t, "h36")
	var found bool
	for _, l := range lines {
		if !strings.Contains(l.Oracle, "the ") {
			continue
		}
		found = true
		withDict, err := r.Recognize(l.Image)
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		saved := r.Dict
		r.Dict = nil
		withoutDict, err := r.Recognize(l.Image)
		r.Dict = saved
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		if withDict.Confidence < withoutDict.Confidence {
			t.Errorf("%s: dictionary confidence %.2f is below the dictionary-free %.2f",
				l.Name, withDict.Confidence, withoutDict.Confidence)
		}
	}
	if !found {
		t.Skip("no corpus line contains a common dictionary word")
	}
}

// The acceptance test for Stage 2: beam plus dictionary must match the oracle
// at least as well as greedy did, on both arms.
func TestBeamMatchesOracle(t *testing.T) {
	r := loadRecognizer(t)
	for _, arm := range []string{"h36", "native"} {
		lines := loadCorpus(t, arm)
		var exact int
		var total float64
		for _, l := range lines {
			line, err := r.Recognize(l.Image)
			if err != nil {
				t.Errorf("%s/%s: Recognize() error = %v", arm, l.Name, err)
				continue
			}
			if line.Text == l.Oracle {
				exact++
			} else {
				t.Logf("%s/%s\n  oracle: %q\n  cadmus: %q\n  cer:    %.4f", arm, l.Name, l.Oracle, line.Text, cer(line.Text, l.Oracle))
			}
			total += cer(line.Text, l.Oracle)
		}
		mean := total / float64(len(lines))
		t.Logf("%s arm: %d/%d exact, mean CER vs oracle %.4f", arm, exact, len(lines), mean)
		if mean > 0.01 {
			t.Errorf("%s arm: mean CER vs oracle = %.4f; want <= 0.01", arm, mean)
		}
	}
}
