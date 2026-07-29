package recog

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// testRand is the seed Normalize's callers use in these tests. The value is
// irrelevant to every assertion below except that it must be non-nil.
func testRand() *nn.Rand { return nn.NewRand(1) }

// ile is Tesseract's interpolated percentile. These expectations are computed
// by hand from src/ccstruct/statistc.cpp:172-197.
func TestILEInterpolatesAcrossBuckets(t *testing.T) {
	// Four samples: two at 10, two at 20. total=4.
	var s stats
	s.add(10, 2)
	s.add(20, 2)
	// frac 0.25 -> target = clip(1.0, 1, 4) = 1.0. The loop runs while
	// sum < target: index 10 consumes 2, sum=2, index=11, loop ends.
	// result = 0 + 11 - (2-1)/2 = 10.5
	if got := s.ile(0.25); math.Abs(got-10.5) > 1e-12 {
		t.Errorf("ile(0.25) = %v; want 10.5", got)
	}
	// frac 0.75 -> target = 3.0. index 10 gives sum=2 (<3), index 20 gives
	// sum=4, index=21. result = 0 + 21 - (4-3)/2 = 20.5
	if got := s.ile(0.75); math.Abs(got-20.5) > 1e-12 {
		t.Errorf("ile(0.75) = %v; want 20.5", got)
	}
	// An empty histogram returns rangemin.
	var empty stats
	if got := empty.ile(0.5); got != 0 {
		t.Errorf("empty ile(0.5) = %v; want 0", got)
	}
	// frac 0 clips the target up to 1, so it does not return rangemin.
	if got := s.ile(0.0); math.Abs(got-10.5) > 1e-12 {
		t.Errorf("ile(0.0) = %v; want 10.5 (target clips to 1)", got)
	}
}

// A flat image has no local minima or maxima at all, so the defaults kick in:
// a single sample at 0 in `mins` and a single sample at 255 in `maxes`.
//
// ile does NOT then return 0 and 255. With total_count_ == 1 the target clips to
// 1.0, the loop consumes the one populated bucket and leaves `index` one PAST
// it, and the result is `rangemin + index - (sum-target)/buckets[index-1]`:
//
//	mins:  index lands at 1   -> 0 + 1   - 0/1 = 1    -> black = 1
//	maxes: index lands at 256 -> 0 + 256 - 0/1 = 256  -> white = 256
//
// contrast is therefore (256-1)/2 = 127.5, and a mid-grey pixel maps to a
// slightly NEGATIVE value, not a positive one. This was checked by running the
// port of `ile` in Step 5 against these histograms.
func TestNormalizeFlatImageUsesDefaults(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 40, 36))
	for i := range img.Pix {
		img.Pix[i] = 128
	}
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Black != 1 || math.Abs(float64(n.Contrast)-127.5) > 1e-6 {
		t.Fatalf("black=%v contrast=%v; want 1 and 127.5", n.Black, n.Contrast)
	}
	// SetPixel is `(pixel - black) / contrast - 1.0f` with every operand a
	// float, so C rounds to float32 after the divide AND after the subtract.
	// Spelling this as one Go constant expression would round only once, at
	// the end, and land exactly one ulp away (-0.003921568859368563 rather
	// than -0.003921568393707275). Runtime float32 variables force the same
	// two rounding points Tesseract has.
	var pixel, black, contrast float32 = 128, 1, 127.5
	want := float64((pixel-black)/contrast - 1)
	if want >= 0 {
		t.Fatalf("expectation %v is not negative; the black point is not 1", want)
	}
	got := make([]float64, 1)
	n.Input.ReadTimeStep(0, got)
	if got[0] != want {
		t.Errorf("pixel value = %v; want %v", got[0], want)
	}
}

// ile's defaults, asserted directly, so a regression here is not mistaken for a
// Normalize bug.
func TestILEEmptyHistogramDefaults(t *testing.T) {
	var mins, maxes stats
	mins.add(0, 1)
	maxes.add(255, 1)
	if got := mins.ile(0.25); got != 1 {
		t.Errorf("mins.ile(0.25) with a single sample at 0 = %v; want 1", got)
	}
	if got := maxes.ile(0.75); got != 256 {
		t.Errorf("maxes.ile(0.75) with a single sample at 255 = %v; want 256", got)
	}
}

// Black ink is the most negative value and white paper the most positive; the
// polarity is not inverted.
func TestNormalizePolarity(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 40, 36))
	for y := range 36 {
		for x := range 40 {
			v := uint8(240)
			if x%4 == 0 {
				v = 10
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	dark := make([]float64, 1)
	light := make([]float64, 1)
	n.Input.ReadTimeStep(n.Input.Map.T(0, 0), dark)
	n.Input.ReadTimeStep(n.Input.Map.T(0, 1), light)
	if dark[0] >= light[0] {
		t.Errorf("dark pixel %v is not below light pixel %v; the polarity is inverted", dark[0], light[0])
	}
}

// A 36-pixel-tall image is not scaled, so the map width is the image width and
// one timestep is XScale page pixels.
func TestNormalizeAtNativeHeightDoesNotScale(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 97, 36))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Input.Map != (nn.StrideMap{Height: 36, Width: 97}) {
		t.Fatalf("map = %v; want 36x97", n.Input.Map)
	}
	if math.Abs(n.ScaleFactor-3) > 1e-9 {
		t.Errorf("ScaleFactor = %v; want 3 (XScale/1.0)", n.ScaleFactor)
	}
}

// A 72-pixel-tall image halves, so one timestep covers 6 page pixels.
func TestNormalizeScaleFactor(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 200, 72))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if math.Abs(n.ScaleFactor-6) > 1e-6 {
		t.Errorf("ScaleFactor = %v; want 6 (3 / (36/72))", n.ScaleFactor)
	}
}

// Tesseract does NOT require the scaled height to equal the network's input
// height. PreScale reports pixGetHeight(pix) as-is, PrepareLSTMInputs rejects
// only `width < min_width || height < min_width`, and Copy2DImage fills any
// missing rows with Randomize. A hard error here would reject real crops whose
// odd source height makes pixScale round to 35 or 37.
func TestNormalizeToleratesAnOffByOneScaledHeight(t *testing.T) {
	// 37 px tall: im_factor = 36/37, and Leptonica's rounding is free to land on
	// 35, 36 or 37. Whatever it lands on, the map must be 36 rows.
	img := image.NewGray(image.Rect(0, 0, 120, 37))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Input.Map.Height != 36 {
		t.Errorf("map height = %d; want 36 (the network's input height, not the scaled image's)", n.Input.Map.Height)
	}
}

// The rejection Tesseract does make: both dimensions must reach min_width,
// which RecognizeLine passes as network->XScaleFactor().
//
// The check is on the SCALED pix, not the crop: PreScale overwrites its
// *scaled_width/*scaled_height out-params with pixGetWidth(pix)/
// pixGetHeight(pix) after pixScale, and PrepareLSTMInputs tests those. So a
// 5-px-wide crop passes at source size and is rejected after halving.
//
// The height arm is unreachable at eng's geometry and cannot be exercised
// here: PreScale always scales towards target_height == 36, which is above
// min_width == 3 whatever the crop's height, so a 1-px-tall line scales UP to
// 4320x36 and Tesseract accepts it. Asserting an error there would contradict
// Input::PrepareLSTMInputs.
func TestNormalizeRejectsImagesBelowTheMinimum(t *testing.T) {
	if _, err := Normalize(image.NewGray(image.Rect(0, 0, 2, 36)), 36, 3, testRand()); err == nil {
		t.Error("a 2-px-wide line: want an error, got nil")
	}
	// 5x72 halves to 2x36: wide enough before scaling, too narrow after.
	if _, err := Normalize(image.NewGray(image.Rect(0, 0, 5, 72)), 36, 3, testRand()); err == nil {
		t.Error("a line that scales to under 3 px wide: want an error, got nil")
	}
}
