package imaging

import (
	"image"
	"testing"
)

// countDiff returns the number of pixels at which got and want disagree,
// failing the test outright if they are not the same size.
func countDiff(t *testing.T, got, want *Bitmap) int {
	t.Helper()
	if got.Width != want.Width || got.Height != want.Height {
		t.Fatalf("size = %dx%d; want %dx%d", got.Width, got.Height, want.Width, want.Height)
	}
	var diff int
	for y := range got.Height {
		for x := range got.Width {
			if got.At(x, y) != want.At(x, y) {
				diff++
			}
		}
	}
	return diff
}

// The rectangle and source origin baked into testdata/golden/gen/gen.c. Against
// the 320x400 operands they overhang the destination to the top and left and
// both images to the right and bottom, so every clipping adjustment in
// pixRasterop fires: the destination rect is pushed to the origin, the source
// origin is advanced to compensate, and the width and height are cut twice.
var (
	ropRect      = image.Rect(-40, -30, 360, 470)
	ropSrcOrigin = image.Pt(60, 80)
)

func TestRasterOpMatchesLeptonica(t *testing.T) {
	cases := []struct {
		name   string
		op     Op
		golden string
	}{
		// OpSet and OpClear ignore the source entirely, including for
		// clipping: Leptonica clips them to the destination alone, so this
		// rectangle covers the whole 320x400 image rather than the 220x290
		// region the two-image clip would leave.
		{"set", OpSet, "rop_set.bin"},
		{"clear", OpClear, "rop_clr.bin"},
		{"src", OpSrc, "rop_copy.bin"},
		{"notsrc", OpNotSrc, "rop_notsrc.bin"},
		{"or", OpSrcOrDst, "rop_or.bin"},
		{"and", OpSrcAndDst, "rop_and.bin"},
		{"xor", OpSrcXorDst, "rop_xor.bin"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dst := loadGolden(t, "rop_dst_in.bin")
			src := loadGolden(t, "rop_src_in.bin")
			want := loadGolden(t, tc.golden)

			RasterOp(dst, ropRect, tc.op, src, ropSrcOrigin)

			if diff := countDiff(t, dst, want); diff != 0 {
				t.Errorf("RasterOp(%s) differs from Leptonica in %d of %d pixels",
					tc.name, diff, dst.Width*dst.Height)
			}
		})
	}
}

// A negative source origin clips the other way round: the destination rect
// moves right and down instead of the source.
func TestRasterOpNegativeSourceOrigin(t *testing.T) {
	dst := loadGolden(t, "rop_dst_in.bin")
	src := loadGolden(t, "rop_src_in.bin")
	want := loadGolden(t, "rop_negsrc.bin")

	RasterOp(dst, image.Rect(200, 250, 400, 550), OpSrc, src, image.Pt(-60, -80))

	if diff := countDiff(t, dst, want); diff != 0 {
		t.Errorf("RasterOp with a negative source origin differs from Leptonica in %d of %d pixels",
			diff, dst.Width*dst.Height)
	}
}

func TestRasterOpDisjointRectIsNoOp(t *testing.T) {
	dst := NewBitmap(16, 16, 1)
	dst.Set(3, 3, 1)
	src := NewBitmap(16, 16, 1)
	for y := range 16 {
		for x := range 16 {
			src.Set(x, y, 1)
		}
	}
	before := dst.Clone()

	RasterOp(dst, image.Rect(40, 40, 60, 60), OpSrc, src, image.Pt(0, 0))
	RasterOp(dst, image.Rect(-40, -40, -20, -20), OpSet, nil, image.Pt(0, 0))

	if diff := countDiff(t, dst, before); diff != 0 {
		t.Errorf("RasterOp with a fully clipped rectangle changed %d pixels; want 0", diff)
	}
}
