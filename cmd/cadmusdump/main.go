// Command cadmusdump prints the contents of a Tesseract .traineddata file: its
// component inventory, the recognizer's header fields, the layer tree with
// per-matrix weight statistics, and optionally the unicharset, the recoder
// mapping and the DAWG lexicons.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

type options struct {
	unicharset bool
	recoder    bool
	dawgs      bool
}

func dump(path string, opt options, w io.Writer) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		return fmt.Errorf("parsing container: %w", err)
	}
	_, _ = fmt.Fprintf(w, "%s\nversion: %s\nbyte-swapped: %v\n\ncomponents:\n", path, c.Version(), c.Swapped())
	for _, t := range c.Present() {
		b, _ := c.Entry(t)
		_, _ = fmt.Fprintf(w, "  %-20s %9d bytes\n", t, len(b))
	}

	// Preserved from the pre-L1a cadmusdump: a legacy (Tesseract-3, no LSTM)
	// .traineddata still has a printable component inventory, and dumping it is
	// useful. LoadModel requires the LSTM component, so bail before calling it
	// rather than turning `cadmusdump some-legacy.traineddata` into exit 1.
	if _, ok := c.Entry(tessdata.TypeLSTM); !ok {
		_, _ = fmt.Fprintln(w, "\nno lstm component (legacy-only model)")
		return nil
	}

	// LoadModel re-walks the container. That is deliberate: ParseContainer only
	// reads the offset table and re-slices the caller's buffer, so the second
	// walk costs nothing and copies nothing, and keeping LoadModel's signature
	// as ([]byte) -> (*Model, error) keeps the library free of a CLI-shaped
	// "also give me the container" return value.
	m, err := tessdata.LoadModel(raw)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}
	rec := m.Recognizer
	_, _ = fmt.Fprintf(w, `
recognizer:
  spec           %s
  training_flags %d
  iteration      %d
  sample_iter    %d
  null_char      %d
  adam_beta      %g
  learning_rate  %g
  momentum       %g

network:
`, rec.NetworkStr, rec.TrainingFlags, rec.TrainingIteration,
		rec.SampleIteration, rec.NullChar, rec.AdamBeta, rec.LearningRate, rec.Momentum)
	rec.Network.Tree(w)

	if opt.unicharset {
		dumpUnicharset(w, m)
	}
	if opt.recoder {
		dumpRecoder(w, m)
	}
	if opt.dawgs {
		dumpDawgs(w, m)
	}
	return nil
}

func dumpUnicharset(w io.Writer, m *tessdata.Model) {
	u := m.Unicharset
	_, _ = fmt.Fprintf(w, "\nunicharset: %d entries\n", u.Size())
	for id := range u.Size() {
		c, _ := u.Char(id)
		_, _ = fmt.Fprintf(w, "  %3d %-14q props=%x script=%-8s", id, c.Text, c.Properties, c.Script)
		if c.Normed != c.Text {
			_, _ = fmt.Fprintf(w, " normed=%q", c.Normed)
		}
		_, _ = fmt.Fprintln(w)
	}
}

func dumpRecoder(w io.Writer, m *tessdata.Model) {
	rc := m.Recoder
	_, _ = fmt.Fprintf(w, "\nrecoder: %d entries, code range %d, max code length %d\n",
		rc.Size(), rc.CodeRange(), rc.MaxCodeLen())
	for id := range rc.Size() {
		code := rc.Encode(id)
		_, _ = fmt.Fprintf(w, "  unichar %3d %-14q -> code %v", id, m.Unicharset.Text(id), code)
		if len(code) == 1 && int(code[0]) == m.NullChar() {
			_, _ = fmt.Fprint(w, "  BLANK")
		}
		_, _ = fmt.Fprintln(w)
	}
}

func dumpDawgs(w io.Writer, m *tessdata.Model) {
	_, _ = fmt.Fprintln(w, "\ndawgs:")
	for _, d := range []struct {
		name string
		d    *tessdata.Dawg
	}{
		{"punc", m.PuncDawg},
		{"word", m.WordDawg},
		{"number", m.NumberDawg},
	} {
		if d.d == nil {
			_, _ = fmt.Fprintf(w, "  %-8s absent\n", d.name)
			continue
		}
		_, _ = fmt.Fprintf(w, "  %-8s %9d edges, unicharset size %d\n", d.name, d.d.NumEdges(), d.d.UnicharsetSize)
	}
}

func main() {
	var opt options
	var activations bool
	flag.BoolVar(&opt.unicharset, "unicharset", false, "print the unicharset table")
	flag.BoolVar(&opt.recoder, "recoder", false, "print the unichar -> output-code mapping")
	flag.BoolVar(&opt.dawgs, "dawgs", false, "print the DAWG lexicon summaries")
	flag.BoolVar(&activations, "activations", false, "run a line image through the network and dump every layer's output")
	flag.Parse()

	if activations {
		if flag.NArg() != 2 {
			fmt.Fprintln(os.Stderr, "usage: cadmusdump -activations <model.traineddata> <line.png>")
			os.Exit(2)
		}
		if err := dumpActivations(flag.Arg(0), flag.Arg(1), os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, "cadmusdump:", err)
			os.Exit(1)
		}
		return
	}

	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: cadmusdump [-unicharset] [-recoder] [-dawgs] <model.traineddata>")
		os.Exit(2)
	}
	if err := dump(flag.Arg(0), opt, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cadmusdump:", err)
		os.Exit(1)
	}
}
