package imaging

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// loadFloatGolden reads a scalar golden produced by dumpFloats in
// testdata/golden/gen/gen.c: an int32 count then that many float32, all
// little-endian.
func loadFloatGolden(t *testing.T, name string, want int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", name))
	if err != nil {
		t.Skipf("golden %s not present (run make goldens): %v", name, err)
	}
	if len(raw) < 4 {
		t.Fatalf("golden %s is truncated: %d bytes", name, len(raw))
	}
	n := int(int32(binary.LittleEndian.Uint32(raw[0:4])))
	if n != want || len(raw) != 4+4*n {
		t.Fatalf("golden %s holds %d floats in %d bytes; want %d floats in %d bytes",
			name, n, len(raw), want, 4+4*want)
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4+4*i:]))
	}
	return v
}

// The layout of deskew_meta.bin, in the order gen.c writes it.
const (
	metaRotRad      = 0 // the exact float32 angle the operand was rotated by
	metaSkewDeg     = 1 // pixFindSkew on the rotated operand, degrees
	metaStraightDeg = 3 // pixFindSkew on the unrotated Otsu image, degrees
)

func degrees(radians float64) float64 { return radians * 180 / math.Pi }

func TestRotateMatchesLeptonica(t *testing.T) {
	meta := loadFloatGolden(t, "deskew_meta.bin", 5)
	in := loadGolden(t, "otsu.bin")
	want := loadGolden(t, "deskew_rot.bin")

	// The same float32 angle Leptonica was handed, so the two implementations
	// cannot disagree about the constant.
	got := Rotate(in, float64(meta[metaRotRad]))

	if got.Width != want.Width || got.Height != want.Height {
		t.Fatalf("Rotate() size = %dx%d; want %dx%d",
			got.Width, got.Height, want.Width, want.Height)
	}
	if diff := countDiff(t, got, want); diff != 0 {
		t.Errorf("Rotate() differs from Leptonica in %d of %d pixels",
			diff, got.Width*got.Height)
	}
}

// pixFindSkew reports the angle required to deskew; SkewAngle reports the skew
// the image carries, which is its negative.
//
// The bound is 0.1 degrees rather than the 0.05 asserted against a known
// rotation below, because Leptonica is the coarser of the two measurements
// here: pixFindSkew sweeps a 4x-reduced image and then interval-halves down to
// a 0.01 degree step limit, which is why both golden values land on binary
// fractions (-2.5625 = -41/16 for an operand rotated by exactly 2.5, and
// -0.046875 = -3/64 for one that is straight). Its quantization alone is of the
// order of the tighter bound.
func TestSkewAngleMatchesLeptonica(t *testing.T) {
	meta := loadFloatGolden(t, "deskew_meta.bin", 5)
	rot := loadGolden(t, "deskew_rot.bin")

	got := degrees(SkewAngle(rot))

	if want := -float64(meta[metaSkewDeg]); math.Abs(got-want) > 0.1 {
		t.Errorf("SkewAngle() = %.4f deg; want %.4f deg (Leptonica), within 0.1", got, want)
	}
}

func TestSkewAngleOnAStraightPageMatchesLeptonica(t *testing.T) {
	meta := loadFloatGolden(t, "deskew_meta.bin", 5)
	straight := loadGolden(t, "otsu.bin")

	got := degrees(SkewAngle(straight))

	if want := -float64(meta[metaStraightDeg]); math.Abs(got-want) > 0.1 {
		t.Errorf("SkewAngle() = %.4f deg; want %.4f deg (Leptonica), within 0.1", got, want)
	}
	if math.Abs(got) > 0.05 {
		t.Errorf("SkewAngle() on an unrotated page = %.4f deg; want 0 within 0.05", got)
	}
}

// The measurement that matters: rotate a page whose skew is known exactly
// because we just introduced it, and recover the angle. This is independent of
// Leptonica's own estimator and of its quantization.
func TestSkewAngleRecoversAKnownRotation(t *testing.T) {
	straight := loadGolden(t, "otsu.bin")

	// Both signs, angles on and off the 0.1 degree sweep grid, and one near the
	// edge of the +-5 degree sweep range.
	for _, wantDeg := range []float64{-4.5, -2.5, -1.23, 0.7, 2.5, 3.87} {
		t.Run(fmt.Sprintf("%+.2fdeg", wantDeg), func(t *testing.T) {
			got := degrees(SkewAngle(Rotate(straight, wantDeg*math.Pi/180)))
			if math.Abs(got-wantDeg) > 0.05 {
				t.Errorf("SkewAngle() = %.4f deg; want %.4f deg within 0.05", got, wantDeg)
			}
		})
	}
}

// Rotation is clockwise for a positive angle, in a coordinate system with y
// increasing downward: a horizontal rule must come out lower on the right of
// the page than on the left. Getting this backwards deskews in the wrong
// direction, doubling the skew instead of removing it.
func TestRotateIsClockwiseForAPositiveAngle(t *testing.T) {
	const w, h = 200, 120
	b := NewBitmap(w, h, 1)
	fillRect(b, 0, 60, w, 1)

	got := Rotate(b, 5*math.Pi/180)

	left, okL := columnInkRow(got, 20)
	right, okR := columnInkRow(got, w-20)
	if !okL || !okR {
		t.Fatalf("Rotate() lost the rule: ink at x=20 %v, at x=%d %v", okL, w-20, okR)
	}
	if right <= left {
		t.Errorf("Rotate(+5deg) put the rule at row %d on the left and %d on the right; want the right lower",
			left, right)
	}
}

// columnInkRow returns the row of the first foreground pixel in column x.
func columnInkRow(b *Bitmap, x int) (int, bool) {
	for y := range b.Height {
		if b.At(x, y) != 0 {
			return y, true
		}
	}
	return 0, false
}

func TestRotateByZeroIsIdentity(t *testing.T) {
	b := NewBitmap(37, 29, 1)
	fillRect(b, 3, 5, 11, 7)
	b.Set(36, 28, 1)

	if diff := countDiff(t, Rotate(b, 0), b); diff != 0 {
		t.Errorf("Rotate(0) changed %d pixels; want 0", diff)
	}
}

// The result is the same size as the input, so a rotation cuts the corners off
// and brings background in behind them.
func TestRotateKeepsSizeAndBringsInBackground(t *testing.T) {
	const n = 101
	b := NewBitmap(n, n, 1)
	fillRect(b, 0, 0, n, n)

	got := Rotate(b, 20*math.Pi/180)

	if got.Width != n || got.Height != n {
		t.Fatalf("Rotate() size = %dx%d; want %dx%d", got.Width, got.Height, n, n)
	}
	for _, c := range [][2]int{{0, 0}, {n - 1, 0}, {0, n - 1}, {n - 1, n - 1}} {
		if v := got.At(c[0], c[1]); v != 0 {
			t.Errorf("Rotate() at corner (%d,%d) = %d; want 0 (background)", c[0], c[1], v)
		}
	}
	if v := got.At(n/2, n/2); v != 1 {
		t.Errorf("Rotate() at the centre = %d; want 1", v)
	}
}

func TestRotateDoesNotModifyItsInput(t *testing.T) {
	b := NewBitmap(40, 40, 1)
	fillRect(b, 5, 5, 20, 12)
	before := b.Clone()

	Rotate(b, 3*math.Pi/180)

	if diff := countDiff(t, b, before); diff != 0 {
		t.Errorf("Rotate changed %d input pixels; want 0", diff)
	}
}

func TestSkewAngleOfAnEmptyBitmapIsZero(t *testing.T) {
	if got := SkewAngle(NewBitmap(50, 50, 1)); got != 0 {
		t.Errorf("SkewAngle(empty) = %v; want 0", got)
	}
}

func TestDeskewRejectsBadInput(t *testing.T) {
	cases := map[string]func(){
		"Rotate depth":    func() { Rotate(NewBitmap(4, 4, 8), 0.1) },
		"SkewAngle depth": func() { SkewAngle(NewBitmap(4, 4, 8)) },
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s with a bad argument did not panic", name)
				}
			}()
			fn()
		})
	}
}
