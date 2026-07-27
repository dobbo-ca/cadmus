package imaging

import (
	"encoding/binary"
	"image"
	"os"
	"path/filepath"
	"testing"
)

// loadComponentGolden reads a component table produced by dumpConnComp in
// testdata/golden/gen/gen.c: an int32 count, then one record of five int32
// (x, y, w, h, pixel count) per component, all little-endian. The records are
// in the order pixConnComp emits them, which ConnComp must reproduce.
func loadComponentGolden(t *testing.T, name string) []Component {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", name))
	if err != nil {
		t.Skipf("golden %s not present (run make goldens): %v", name, err)
	}
	if len(raw) < 4 {
		t.Fatalf("golden %s is truncated: %d bytes", name, len(raw))
	}
	n := int(int32(binary.LittleEndian.Uint32(raw[0:4])))
	if want := 4 + 20*n; len(raw) != want {
		t.Fatalf("golden %s is %d bytes; want %d for %d components", name, len(raw), want, n)
	}
	comps := make([]Component, n)
	for i := range comps {
		rec := raw[4+20*i:]
		f := func(k int) int { return int(int32(binary.LittleEndian.Uint32(rec[4*k:]))) }
		x, y, w, h := f(0), f(1), f(2), f(3)
		comps[i] = Component{Bounds: image.Rect(x, y, x+w, y+h), Pixels: f(4)}
	}
	return comps
}

func compareComponents(t *testing.T, got, want []Component) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("ConnComp() returned %d components; want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("component %d = {%v, %d px}; want {%v, %d px}",
				i, got[i].Bounds, got[i].Pixels, want[i].Bounds, want[i].Pixels)
		}
	}
}

func TestConnCompMatchesLeptonica(t *testing.T) {
	// The scan image has no diagonal contacts, so its two connectivity goldens
	// are identical; the diagonal operand below is what separates 4 from 8.
	for _, conn := range []int{4, 8} {
		t.Run(map[int]string{4: "conn4", 8: "conn8"}[conn], func(t *testing.T) {
			in := loadGolden(t, "otsu.bin")
			want := loadComponentGolden(t, map[int]string{4: "conncomp4.bin", 8: "conncomp8.bin"}[conn])

			compareComponents(t, ConnComp(in, conn), want)
		})
	}
}

// The diagonal operand is a staircase, two blocks touching only at a corner, a
// lone pixel, and a run along the top border. Its 4- and 8-connected component
// sets differ (20 components against 4), so this is the test that fails if
// ConnComp ignores its connectivity argument or gets the neighbourhood wrong.
func TestConnCompDiagonalMatchesLeptonica(t *testing.T) {
	for _, conn := range []int{4, 8} {
		t.Run(map[int]string{4: "conn4", 8: "conn8"}[conn], func(t *testing.T) {
			in := loadGolden(t, "conncomp_diag_in.bin")
			want := loadComponentGolden(t, map[int]string{
				4: "conncomp_diag4.bin", 8: "conncomp_diag8.bin",
			}[conn])

			compareComponents(t, ConnComp(in, conn), want)
		})
	}
}

func TestConnCompAllBackgroundIsEmpty(t *testing.T) {
	b := NewBitmap(32, 24, 1)
	for _, conn := range []int{4, 8} {
		if got := ConnComp(b, conn); len(got) != 0 {
			t.Errorf("ConnComp(blank, %d) = %v; want no components", conn, got)
		}
	}
}

func TestConnCompSinglePixelIsAComponent(t *testing.T) {
	b := NewBitmap(32, 24, 1)
	b.Set(7, 11, 1)
	want := []Component{{Bounds: image.Rect(7, 11, 8, 12), Pixels: 1}}
	for _, conn := range []int{4, 8} {
		compareComponents(t, ConnComp(b, conn), want)
	}
}

// A component that runs into the image edge must keep its full extent: the
// scan has to look past the border rather than treat it as a boundary that
// truncates the blob.
func TestConnCompBorderComponentIsNotClipped(t *testing.T) {
	const w, h = 16, 12
	b := NewBitmap(w, h, 1)
	for x := range w {
		b.Set(x, 0, 1)
		b.Set(x, h-1, 1)
	}
	for y := range h {
		b.Set(0, y, 1)
		b.Set(w-1, y, 1)
	}
	want := []Component{{Bounds: image.Rect(0, 0, w, h), Pixels: 2*w + 2*(h-2)}}
	for _, conn := range []int{4, 8} {
		compareComponents(t, ConnComp(b, conn), want)
	}
}

// Components come back in raster order of their first foreground pixel, which
// is what Leptonica does and what the golden tables encode.
func TestConnCompOrderIsRasterOrderOfFirstPixel(t *testing.T) {
	b := NewBitmap(20, 20, 1)
	b.Set(15, 2, 1) // later in x, earliest row
	b.Set(1, 5, 1)
	b.Set(9, 5, 1) // same row, later column
	b.Set(3, 12, 1)

	want := []Component{
		{Bounds: image.Rect(15, 2, 16, 3), Pixels: 1},
		{Bounds: image.Rect(1, 5, 2, 6), Pixels: 1},
		{Bounds: image.Rect(9, 5, 10, 6), Pixels: 1},
		{Bounds: image.Rect(3, 12, 4, 13), Pixels: 1},
	}
	compareComponents(t, ConnComp(b, 8), want)
}

// A U-shape whose arms only meet at the bottom: on every row above the bar the
// two arms look like separate runs, so an implementation that decides component
// membership row by row splits it in two.
func TestConnCompJoinsArmsThatMeetLate(t *testing.T) {
	b := NewBitmap(20, 20, 1)
	for y := 2; y <= 10; y++ {
		b.Set(4, y, 1)
		b.Set(10, y, 1)
	}
	for x := 4; x <= 10; x++ {
		b.Set(x, 10, 1)
	}
	want := []Component{{Bounds: image.Rect(4, 2, 11, 11), Pixels: 2*8 + 7}}
	compareComponents(t, ConnComp(b, 4), want)
}

func TestConnCompRejectsBadInput(t *testing.T) {
	t.Run("connectivity", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("ConnComp(b, 6) did not panic")
			}
		}()
		ConnComp(NewBitmap(4, 4, 1), 6)
	})
	t.Run("depth", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("ConnComp(8bpp) did not panic")
			}
		}()
		ConnComp(NewBitmap(4, 4, 8), 8)
	})
}
