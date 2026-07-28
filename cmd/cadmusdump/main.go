// Command cadmusdump prints the contents of a Tesseract .traineddata file:
// its component inventory and, for the LSTM component, its layer tree.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func dump(path string, w io.Writer) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		return fmt.Errorf("parsing container: %w", err)
	}

	fmt.Fprintf(w, "%s\nversion: %s\nbyte-swapped: %v\n\ncomponents:\n", path, c.Version(), c.Swapped())
	for _, t := range c.Present() {
		b, _ := c.Entry(t)
		fmt.Fprintf(w, "  %-20s %9d bytes\n", t, len(b))
	}

	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		fmt.Fprintln(w, "\nno lstm component (legacy-only model)")
		return nil
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return fmt.Errorf("parsing lstm component: %w", err)
	}
	fmt.Fprintf(w, "\nrecognizer:\n  spec           %s\n  training_flags %d\n  iteration      %d\n  sample_iter    %d\n  null_char      %d\n  adam_beta      %g\n  learning_rate  %g\n  momentum       %g\n",
		rec.NetworkStr, rec.TrainingFlags, rec.TrainingIteration,
		rec.SampleIteration, rec.NullChar, rec.AdamBeta, rec.LearningRate, rec.Momentum)
	fmt.Fprintln(w, "\nnetwork:")
	rec.Network.Tree(w)
	return nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: cadmusdump <model.traineddata>")
		os.Exit(2)
	}
	if err := dump(os.Args[1], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cadmusdump:", err)
		os.Exit(1)
	}
}
