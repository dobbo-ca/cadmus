package imaging

import (
	"image"
	"slices"
	"testing"
)

// assertIDs compares a Query result against the ids expected, in the ascending
// order Query promises. No ids expected means the query matched nothing.
func assertIDs(t *testing.T, got []int, want ...int) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Errorf("Query() = %v; want %v", got, want)
	}
}

// Overlapping and disjoint boxes together: a query must report exactly the
// boxes it intersects, whether that is one of them, several, or all.
func TestGridQueryReturnsIntersectingIDs(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(1, image.Rect(0, 0, 20, 20))
	g.Insert(2, image.Rect(15, 15, 25, 25)) // overlaps 1
	g.Insert(3, image.Rect(60, 60, 70, 70)) // disjoint from both

	assertIDs(t, g.Query(image.Rect(0, 0, 5, 5)), 1)
	assertIDs(t, g.Query(image.Rect(16, 16, 17, 17)), 1, 2)
	assertIDs(t, g.Query(image.Rect(62, 62, 63, 63)), 3)
	assertIDs(t, g.Query(image.Rect(0, 0, 100, 100)), 1, 2, 3)
}

func TestGridQueryMatchingNothingIsEmpty(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(1, image.Rect(0, 0, 20, 20))
	g.Insert(2, image.Rect(60, 60, 70, 70))

	assertIDs(t, g.Query(image.Rect(30, 30, 40, 40)))
	assertIDs(t, NewGrid(image.Rect(0, 0, 100, 100), 10).Query(image.Rect(0, 0, 100, 100)))
}

// Two boxes sharing a cell but not touching. An index that returned whole cells
// as candidates would report both; Query filters down to real intersections.
func TestGridQueryIsExactNotCellLevel(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 50)
	g.Insert(1, image.Rect(0, 0, 5, 5))
	g.Insert(2, image.Rect(40, 40, 45, 45)) // same 50x50 cell, disjoint from 1

	assertIDs(t, g.Query(image.Rect(0, 0, 6, 6)), 1)
	assertIDs(t, g.Query(image.Rect(44, 44, 49, 49)), 2)
}

// A box spanning many cells is filed under every one of them, so it must be
// reachable from each and still be reported once, not once per cell.
func TestGridSpanningRectIsFoundOnceFromEveryCell(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(7, image.Rect(5, 5, 95, 95))

	for y := 5; y < 95; y += 10 {
		for x := 5; x < 95; x += 10 {
			assertIDs(t, g.Query(image.Rect(x, y, x+1, y+1)), 7)
		}
	}
	// A query spanning every cell must not report it once per shared cell.
	assertIDs(t, g.Query(image.Rect(0, 0, 100, 100)), 7)
}

// Rectangles are half-open here as everywhere else in this package: boxes that
// share an edge do not intersect.
func TestGridAbuttingRectsDoNotIntersect(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(1, image.Rect(0, 0, 10, 10))

	assertIDs(t, g.Query(image.Rect(10, 0, 20, 10)))
	assertIDs(t, g.Query(image.Rect(9, 0, 19, 10)), 1)
}

// Boxes reaching outside the grid are clamped into the edge cells rather than
// dropped, so a query that reaches them still finds them.
func TestGridClampsOutOfBoundsRects(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(1, image.Rect(-40, -40, -30, -30)) // entirely above and left
	g.Insert(2, image.Rect(90, 90, 200, 200))   // straddles the far corner

	assertIDs(t, g.Query(image.Rect(-45, -45, -35, -35)), 1)
	assertIDs(t, g.Query(image.Rect(150, 150, 160, 160)), 2)
	assertIDs(t, g.Query(image.Rect(95, 95, 96, 96)), 2)
	assertIDs(t, g.Query(image.Rect(20, 20, 30, 30)))
}

// The grid's origin need not be (0,0): cells are laid out from bounds.Min.
func TestGridHandlesNonZeroOrigin(t *testing.T) {
	g := NewGrid(image.Rect(200, 100, 400, 300), 25)
	g.Insert(1, image.Rect(210, 110, 230, 130))
	g.Insert(2, image.Rect(380, 280, 395, 295))

	assertIDs(t, g.Query(image.Rect(220, 120, 221, 121)), 1)
	assertIDs(t, g.Query(image.Rect(200, 100, 400, 300)), 1, 2)
	assertIDs(t, g.Query(image.Rect(300, 200, 310, 210)))
}

// A grid whose bounds are not a whole number of cells still has to cover its
// far edge: the last row and column are partial cells.
func TestGridCoversPartialTrailingCells(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 25, 25), 10)
	g.Insert(1, image.Rect(23, 23, 25, 25))

	assertIDs(t, g.Query(image.Rect(24, 24, 25, 25)), 1)
	assertIDs(t, g.Query(image.Rect(0, 0, 25, 25)), 1)
}

// An empty rectangle covers no pixels, so it neither matches nor is matched.
func TestGridEmptyRectsNeverMatch(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(1, image.Rect(10, 10, 10, 20)) // zero width
	g.Insert(2, image.Rect(30, 30, 40, 40))

	assertIDs(t, g.Query(image.Rect(0, 0, 100, 100)), 2)
	assertIDs(t, g.Query(image.Rect(35, 35, 35, 35)))
}

// Ids are the caller's, not indexes: they may repeat, and a repeated id is
// still reported once.
func TestGridDeduplicatesRepeatedIDs(t *testing.T) {
	g := NewGrid(image.Rect(0, 0, 100, 100), 10)
	g.Insert(5, image.Rect(0, 0, 10, 10))
	g.Insert(5, image.Rect(80, 80, 90, 90))
	g.Insert(2, image.Rect(40, 40, 50, 50))

	assertIDs(t, g.Query(image.Rect(0, 0, 100, 100)), 2, 5)
}

func TestGridRejectsBadInput(t *testing.T) {
	t.Run("cellSize", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("NewGrid(bounds, 0) did not panic")
			}
		}()
		NewGrid(image.Rect(0, 0, 10, 10), 0)
	})
	t.Run("bounds", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Error("NewGrid(empty bounds, 4) did not panic")
			}
		}()
		NewGrid(image.Rect(10, 10, 10, 10), 4)
	})
}
