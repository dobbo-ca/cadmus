package recog

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func loadModel(t *testing.T) *tessdata.Recognizer {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		t.Fatal("no lstm component")
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	return rec
}

func TestBuildRealModel(t *testing.T) {
	net, err := Build(loadModel(t))
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if net.InputHeight != 36 {
		t.Errorf("InputHeight = %d; want 36", net.InputHeight)
	}
	if net.NumOutputs != 111 {
		t.Errorf("NumOutputs = %d; want 111", net.NumOutputs)
	}
	if net.NullChar != 110 {
		t.Errorf("NullChar = %d; want 110 (NumOutputs-1)", net.NullChar)
	}
	if net.XScale != 3 {
		t.Errorf("XScale = %d; want 3 (Mp3,3)", net.XScale)
	}
	if net.Root.NumOutputs() != 111 {
		t.Errorf("root NumOutputs() = %d; want 111", net.Root.NumOutputs())
	}
}

// A forward pass over a synthetic input must produce one probability
// distribution per output timestep, of the right width, in the right map.
func TestBuildForwardShapes(t *testing.T) {
	net, err := Build(loadModel(t))
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	const width = 120
	in := nn.NewTensor(nn.StrideMap{Height: 36, Width: width}, 1)
	out, err := net.Root.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	// Maxpool 3,3 reduces 36x120 to 12x40; XYTranspose+SummLSTM collapses the
	// 12 rows to 1; so the output is a 1 x 40 sequence of 111-wide softmax rows.
	if out.Map != (nn.StrideMap{Height: 1, Width: width / 3}) {
		t.Fatalf("output map = %v; want {1 %d}", out.Map, width/3)
	}
	if out.Features != 111 {
		t.Fatalf("output features = %d; want 111", out.Features)
	}
	row := make([]float64, 111)
	for tt := range out.Map.Len() {
		out.ReadTimeStep(tt, row)
		var sum float64
		for _, p := range row {
			sum += p
		}
		if sum < 0.99 || sum > 1.01 {
			t.Fatalf("t=%d: softmax row sums to %v; want ~1", tt, sum)
		}
	}
}

func TestBuildRejectsAnInt8Model(t *testing.T) {
	rec := loadModel(t)
	rec.TrainingFlags |= 1 // TF_INT_MODE
	if _, err := Build(rec); err == nil {
		t.Fatal("Build with TF_INT_MODE set: want error, got nil")
	}
}
