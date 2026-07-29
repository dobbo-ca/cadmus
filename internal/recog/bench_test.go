package recog

import "testing"

func BenchmarkRecognizeLine(b *testing.B) {
	r := loadRecognizer(b)
	lines := loadCorpus(b, "h36")
	b.ResetTimer()
	for range b.N {
		for _, l := range lines {
			if _, err := r.Recognize(l.Image); err != nil {
				b.Fatalf("Recognize() error = %v", err)
			}
		}
	}
}
