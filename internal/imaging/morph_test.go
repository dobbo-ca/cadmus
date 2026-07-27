package imaging

import "testing"

func TestBrickMorphologyMatchesLeptonica(t *testing.T) {
	cases := []struct {
		name   string
		fn     func(*Bitmap, int, int) *Bitmap
		w, h   int
		golden string
	}{
		{"dilate", Dilate, 5, 3, "dilate_5x3.bin"},
		// An even-sided brick has its origin off centre, which makes dilation
		// and erosion sample mirrored windows rather than the same one.
		{"dilate even", Dilate, 4, 2, "dilate_4x2.bin"},
		{"erode", Erode, 3, 7, "erode_3x7.bin"},
		{"open", Open, 5, 5, "open_5x5.bin"},
		{"close", Close, 7, 3, "close_7x3.bin"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := loadGolden(t, "otsu.bin")
			want := loadGolden(t, tc.golden)

			got := tc.fn(in, tc.w, tc.h)

			if diff := countDiff(t, got, want); diff != 0 {
				t.Errorf("%s(%d,%d) differs from Leptonica in %d of %d pixels",
					tc.name, tc.w, tc.h, diff, got.Width*got.Height)
			}
		})
	}
}

// A lone pixel dilated by an even-sided brick spreads off centre: the brick
// origin is at (w/2, h/2), which leaves more of the brick to its left than to
// its right, so a 4x2 brick spreads 2 columns left and 1 right, 1 row up and 0
// down. Getting this backwards shifts every dilation by a pixel.
func TestDilateEvenBrickIsOffCentre(t *testing.T) {
	b := NewBitmap(12, 12, 1)
	b.Set(5, 5, 1)

	got := Dilate(b, 4, 2)

	for y := range 12 {
		for x := range 12 {
			want := uint8(0)
			if x >= 3 && x <= 6 && y >= 4 && y <= 5 {
				want = 1
			}
			if got.At(x, y) != want {
				t.Fatalf("Dilate(4,2) at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

// Leptonica's default boundary condition is asymmetric: everything outside the
// image is OFF for erosion as well as dilation, so eroding a solid image clears
// a border ring rather than leaving it intact.
func TestErodeClearsBorderUnderAsymmetricBC(t *testing.T) {
	const n = 20
	b := NewBitmap(n, n, 1)
	for y := range n {
		for x := range n {
			b.Set(x, y, 1)
		}
	}

	got := Erode(b, 3, 3)

	for y := range n {
		for x := range n {
			want := uint8(0)
			if x >= 1 && x <= n-2 && y >= 1 && y <= n-2 {
				want = 1
			}
			if got.At(x, y) != want {
				t.Fatalf("Erode(3,3) at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

// Opening deletes what the brick does not fit inside and returns everything
// else unchanged: the point of the operation, and the reason despeckling uses it.
func TestOpenRemovesSpecksAndKeepsBlocks(t *testing.T) {
	b := NewBitmap(40, 40, 1)
	fillRect(b, 10, 10, 9, 9)
	fillRect(b, 30, 30, 2, 2)

	got := Open(b, 5, 5)

	for y := range 40 {
		for x := range 40 {
			want := uint8(0)
			if x >= 10 && x < 19 && y >= 10 && y < 19 {
				want = 1
			}
			if got.At(x, y) != want {
				t.Fatalf("Open(5,5) at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

// Closing joins what is closer together than the brick without growing the
// outer extent.
func TestCloseFillsSmallGaps(t *testing.T) {
	b := NewBitmap(40, 40, 1)
	fillRect(b, 10, 10, 5, 5)
	fillRect(b, 17, 10, 5, 5)

	got := Close(b, 5, 1)

	for y := range 40 {
		for x := range 40 {
			want := uint8(0)
			if x >= 10 && x <= 21 && y >= 10 && y < 15 {
				want = 1
			}
			if got.At(x, y) != want {
				t.Fatalf("Close(5,1) at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

func TestMorphologyWithUnitBrickIsIdentity(t *testing.T) {
	b := NewBitmap(16, 16, 1)
	fillRect(b, 3, 4, 5, 6)

	for name, got := range map[string]*Bitmap{
		"Dilate": Dilate(b, 1, 1),
		"Erode":  Erode(b, 1, 1),
		"Open":   Open(b, 1, 1),
		"Close":  Close(b, 1, 1),
	} {
		if diff := countDiff(t, got, b); diff != 0 {
			t.Errorf("%s(1,1) changed %d pixels; want 0", name, diff)
		}
	}
}

func fillRect(b *Bitmap, x0, y0, w, h int) {
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			b.Set(x, y, 1)
		}
	}
}
