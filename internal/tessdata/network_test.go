package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The whole point of the spike: parse the real model's graph.
func TestParseNetworkRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(TypeLSTM)
	if !ok {
		t.Fatal("eng.traineddata has no lstm component")
	}

	root, err := ParseNetwork(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseNetwork() error = %v", err)
	}

	if root.Type != LayerSeries {
		t.Errorf("root.Type = %v; want %v", root.Type, LayerSeries)
	}
	if len(root.Children) == 0 {
		t.Fatal("root has no children; the graph did not deserialize")
	}

	var b strings.Builder
	root.Tree(&b)
	t.Logf("network graph:\n%s", b.String())

	// The tessdata_best English model is a CRNN: it must contain at least one
	// convolution and at least one LSTM somewhere in the tree.
	var conv, lstmCount int
	var walk func(*Layer)
	walk = func(l *Layer) {
		switch l.Type {
		case LayerConvolve:
			conv++
		case LayerLSTM, LayerLSTMSummary, LayerLSTMSoftmax, LayerLSTMSoftmaxEncoded:
			lstmCount++
		}
		for _, ch := range l.Children {
			walk(ch)
		}
	}
	walk(root)
	if conv == 0 {
		t.Error("no convolution layers found; expected a CRNN")
	}
	if lstmCount == 0 {
		t.Error("no LSTM layers found; expected a CRNN")
	}
}

func TestParseNetworkRejectsAbsurdStackSize(t *testing.T) {
	// A Series header claiming 99999 children must be rejected, matching
	// Plumbing::DeSerialize's 10000 guard.
	var b []byte
	b = append(b, byte(LayerSeries), 0, 0)            // type, training, backprop
	b = append(b, 0, 0, 0, 0)                         // flags
	b = append(b, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) // ni, no, num_weights
	b = append(b, 0, 0, 0, 0)                         // empty name
	b = append(b, 0x9f, 0x86, 0x01, 0x00)             // stack size 99999
	if _, err := ParseNetwork(b, false); err == nil {
		t.Fatal("ParseNetwork() with 99999-child stack: want error, got nil")
	}
}
