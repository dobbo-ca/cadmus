package imaging

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// loadGolden reads a dump produced by testdata/golden/gen/gen.c: three int32
// header fields (width, height, depth) then packed 32-bit-word rows, exactly as
// Leptonica stores them.
//
// Leptonica packs pixels MSB-first within the *value* of each l_uint32 word,
// and gen.c fwrites those words raw, so the file carries them in the generating
// host's byte order — little-endian, since the committed goldens were produced
// on arm64. Decoding each word as little-endian reconstructs the logical word
// value that Leptonica's own accessors see, which is why the header and the
// body use the same byte order here.
func loadGolden(t *testing.T, name string) *Bitmap {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", name))
	if err != nil {
		t.Skipf("golden %s not present (run make goldens): %v", name, err)
	}
	if len(raw) < 12 {
		t.Fatalf("golden %s is truncated: %d bytes", name, len(raw))
	}
	w := int(int32(binary.LittleEndian.Uint32(raw[0:4])))
	h := int(int32(binary.LittleEndian.Uint32(raw[4:8])))
	d := int(int32(binary.LittleEndian.Uint32(raw[8:12])))

	b := NewBitmap(w, h, d)
	wpl := (w*d + 31) / 32
	body := raw[12:]
	for y := range h {
		for x := range w {
			word := binary.LittleEndian.Uint32(body[(y*wpl+(x*d)/32)*4:])
			switch d {
			case 1:
				b.Set(x, y, uint8((word>>(31-uint(x*d)%32))&1))
			case 8:
				b.Set(x, y, uint8((word>>(24-uint(x*d)%32))&0xff))
			default:
				t.Fatalf("unsupported golden depth %d", d)
			}
		}
	}
	return b
}

func TestOtsuMatchesLeptonica(t *testing.T) {
	gray := loadGolden(t, "gray.bin")
	want := loadGolden(t, "otsu.bin")

	got := Otsu(gray)

	if got.Width != want.Width || got.Height != want.Height {
		t.Fatalf("Otsu() size = %dx%d; want %dx%d", got.Width, got.Height, want.Width, want.Height)
	}
	// Leptonica's binary convention is 1 = foreground; ours must match exactly.
	var diff int
	for y := range got.Height {
		for x := range got.Width {
			if got.At(x, y) != want.At(x, y) {
				diff++
			}
		}
	}
	if diff != 0 {
		total := got.Width * got.Height
		t.Errorf("Otsu() differs from Leptonica in %d of %d pixels (%.4f%%)",
			diff, total, 100*float64(diff)/float64(total))
	}
}

func TestOtsuThresholdBimodal(t *testing.T) {
	// Half the pixels at 20, half at 200: the threshold must land between.
	b := NewBitmap(10, 10, 8)
	for y := range 10 {
		for x := range 10 {
			if y < 5 {
				b.Set(x, y, 20)
			} else {
				b.Set(x, y, 200)
			}
		}
	}
	got := OtsuThreshold(b)
	if got < 20 || got > 200 {
		t.Errorf("OtsuThreshold() = %d; want a value in [20,200]", got)
	}
}

func TestOtsuForegroundIsOne(t *testing.T) {
	// A dark blob on a light field must binarize to 1s on the blob.
	b := NewBitmap(10, 10, 8)
	for y := range 10 {
		for x := range 10 {
			b.Set(x, y, 240)
		}
	}
	for y := 2; y < 5; y++ {
		for x := 2; x < 5; x++ {
			b.Set(x, y, 10)
		}
	}
	got := Otsu(b)
	if got.At(3, 3) != 1 {
		t.Errorf("Otsu() at dark pixel = %d; want 1 (foreground)", got.At(3, 3))
	}
	if got.At(8, 8) != 0 {
		t.Errorf("Otsu() at light pixel = %d; want 0 (background)", got.At(8, 8))
	}
}
