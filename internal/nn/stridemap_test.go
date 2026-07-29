package nn

import "testing"

func TestStrideMapRasterOrder(t *testing.T) {
	m := StrideMap{Height: 3, Width: 4}
	if m.Len() != 12 {
		t.Fatalf("Len() = %d; want 12", m.Len())
	}
	// x is the inner dimension: Tesseract's Index::Increment carries from
	// FD_DIMSIZE-1 (width) down to 0.
	if got := m.T(0, 1); got != 1 {
		t.Errorf("T(0,1) = %d; want 1", got)
	}
	if got := m.T(1, 0); got != 4 {
		t.Errorf("T(1,0) = %d; want 4", got)
	}
	for tt := range m.Len() {
		y, x := m.YX(tt)
		if m.T(y, x) != tt {
			t.Errorf("YX/T round trip failed at t=%d: got (%d,%d)", tt, y, x)
		}
	}
}

func TestStrideMapOffset(t *testing.T) {
	m := StrideMap{Height: 3, Width: 4}
	if got, ok := m.Offset(m.T(1, 1), -1, -1); !ok || got != m.T(0, 0) {
		t.Errorf("Offset(T(1,1),-1,-1) = %d,%v; want %d,true", got, ok, m.T(0, 0))
	}
	if _, ok := m.Offset(m.T(0, 0), -1, 0); ok {
		t.Error("Offset above the top row reported in-bounds")
	}
	if _, ok := m.Offset(m.T(0, 3), 0, 1); ok {
		t.Error("Offset past the right edge reported in-bounds")
	}
	if _, ok := m.Offset(m.T(2, 0), 1, 0); ok {
		t.Error("Offset below the bottom row reported in-bounds")
	}
}

// ScaleXY floor-divides, so a partial trailing pooling window is dropped, not
// padded. 36 rows and 100 columns at Mp3,3 give 12 x 33, not 12 x 34.
func TestStrideMapScaleXYTruncates(t *testing.T) {
	got := StrideMap{Height: 36, Width: 100}.ScaleXY(3, 3)
	if got != (StrideMap{Height: 12, Width: 33}) {
		t.Errorf("ScaleXY(3,3) = %+v; want {12 33}", got)
	}
}

func TestStrideMapTransposeAndReduce(t *testing.T) {
	m := StrideMap{Height: 12, Width: 33}
	if got := m.TransposeXY(); got != (StrideMap{Height: 33, Width: 12}) {
		t.Errorf("TransposeXY() = %+v; want {33 12}", got)
	}
	if got := m.TransposeXY().TransposeXY(); got != m {
		t.Errorf("TransposeXY is not an involution: %+v", got)
	}
	if got := m.ReduceWidthTo1(); got != (StrideMap{Height: 12, Width: 1}) {
		t.Errorf("ReduceWidthTo1() = %+v; want {12 1}", got)
	}
}
