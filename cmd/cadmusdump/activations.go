package main

import (
	"fmt"
	"image"
	_ "image/png"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/recog"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// dumpActivations runs one line image through the network and writes every
// layer's output in the same shape as Tesseract's DEBUG_DETAIL dump: a header
// line "Output:<layer name>", then one line per feature holding that feature's
// value at every timestep, space separated.
//
// Matching that shape exactly is the point — it makes the two dumps diffable
// without a parser on either side.
func dumpActivations(modelPath, imagePath string, w io.Writer) error {
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("reading %s: %w", modelPath, err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		return fmt.Errorf("parsing container: %w", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		return fmt.Errorf("%s has no lstm component", modelPath)
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return fmt.Errorf("parsing recognizer: %w", err)
	}

	f, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("opening %s: %w", imagePath, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return fmt.Errorf("decoding %s: %w", imagePath, err)
	}

	var dumpErr error
	net, err := recog.BuildWithTap(rec, func(l nn.Layer, out *nn.Tensor) {
		if dumpErr != nil {
			return
		}
		dumpErr = writeBlock(w, l.Name(), out)
	})
	if err != nil {
		return fmt.Errorf("building network: %w", err)
	}

	// Normalize consumes the same randomizer Convolve does, and it must run
	// first, exactly as Copy2DImage runs before Convolve::Forward.
	norm, err := recog.Normalize(img, net.InputHeight, net.XScale, net.Rand)
	if err != nil {
		return fmt.Errorf("normalizing: %w", err)
	}
	// Keyed "Normalized", not "Input": the graph's Input layer is also named
	// "Input", and two blocks with the same header make Step 6's diff ambiguous.
	if err := writeBlock(w, "Normalized", norm.Input); err != nil {
		return err
	}
	if _, err := net.Root.Forward(norm.Input); err != nil {
		return fmt.Errorf("forward: %w", err)
	}
	return dumpErr
}

// writeBlock's " %.9g" is Tesseract's own format, and the precision is not
// cosmetic. NetworkIO::Print is `tprintf(" %g", f_[t][y])`, and C's %g gives 6
// significant digits, which quantizes any comparison at ~1e-6 relative — above
// the ~1e-7 float32-store divergence the delta profile exists to measure.
// debug-detail.patch raises Tesseract to %.9g; 9 digits round-trips a float
// exactly, so identical float32s print identically on both sides and the
// profile's floor is a real zero rather than a rendering artifact.
func writeBlock(w io.Writer, name string, x *nn.Tensor) error {
	if _, err := fmt.Fprintf(w, "Output:%s\n", name); err != nil {
		return err
	}
	for feat := range x.Features {
		for t := range x.Map.Len() {
			if _, err := fmt.Fprintf(w, " %.9g", x.Row(t)[feat]); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintln(w); err != nil {
			return err
		}
	}
	return nil
}
