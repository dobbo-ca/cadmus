package imaging

import (
	"fmt"
	"image"
	"slices"
)

// Grid is a uniform-cell spatial index over bounding boxes — the "bbgrid" that
// later layout passes use to ask which boxes lie near a region without scanning
// every box on the page. Boxes are filed under the cells they cover, so a query
// only examines the boxes filed under the cells it touches.
//
// Cells are cellSize x cellSize in image coordinates. A box larger than a cell
// is filed under every cell it covers, costing one entry per covered cell;
// choosing cellSize near the median box size keeps that near one entry per box.
//
// Boxes reaching outside bounds are clamped into the edge cells rather than
// dropped. Clamping is monotone, so two boxes that intersect in image space
// always share at least one cell and no intersection is lost.
//
// The zero Grid is not usable; call NewGrid. A Grid may be queried concurrently
// once no further inserts are in flight, but Insert is not safe to call
// concurrently with anything.
type Grid struct {
	bounds   image.Rectangle
	cellSize int
	cols     int
	rows     int

	ids   []int             // ids[e] is the caller's id for entry e
	rects []image.Rectangle // rects[e] is entry e's box
	cells [][]int           // cells[row*cols+col] holds the entries covering that cell
}

// NewGrid returns an empty Grid covering bounds with square cells cellSize
// across. The trailing row and column are partial when bounds is not a whole
// number of cells. It panics if cellSize is not positive or bounds is empty.
func NewGrid(bounds image.Rectangle, cellSize int) *Grid {
	if cellSize <= 0 {
		panic(fmt.Sprintf("imaging: NewGrid needs a positive cell size, got %d", cellSize))
	}
	if bounds.Empty() {
		panic(fmt.Sprintf("imaging: NewGrid needs non-empty bounds, got %v", bounds))
	}
	cols := (bounds.Dx() + cellSize - 1) / cellSize
	rows := (bounds.Dy() + cellSize - 1) / cellSize
	return &Grid{
		bounds:   bounds,
		cellSize: cellSize,
		cols:     cols,
		rows:     rows,
		cells:    make([][]int, cols*rows),
	}
}

// Insert files box r under id. Ids are the caller's own labels: they need not
// be unique, dense, or ordered. An empty r covers no cells and can never be
// returned, since an empty rectangle intersects nothing.
func (g *Grid) Insert(id int, r image.Rectangle) {
	e := len(g.ids)
	g.ids = append(g.ids, id)
	g.rects = append(g.rects, r)
	g.eachCell(r, func(i int) {
		g.cells[i] = append(g.cells[i], e)
	})
}

// Query returns the ids of every box intersecting r, in ascending order with
// duplicates removed. Intersection is exact: sharing a cell with r is not
// enough. Rectangles are half-open, so a box that merely abuts r does not
// match. A query that matches nothing returns nil.
func (g *Grid) Query(r image.Rectangle) []int {
	var out []int
	g.eachCell(r, func(i int) {
		for _, e := range g.cells[i] {
			if g.rects[e].Overlaps(r) {
				out = append(out, g.ids[e])
			}
		}
	})
	slices.Sort(out)
	return slices.Compact(out)
}

// eachCell calls fn with the index of every cell r covers, after clamping r to
// the grid. An empty r covers no cells and fn is never called.
func (g *Grid) eachCell(r image.Rectangle, fn func(i int)) {
	if r.Empty() {
		return
	}
	// Max is exclusive, so the last covered cell is the one holding Max-1.
	col0 := clampIndex(r.Min.X-g.bounds.Min.X, g.cellSize, g.cols)
	col1 := clampIndex(r.Max.X-1-g.bounds.Min.X, g.cellSize, g.cols)
	row0 := clampIndex(r.Min.Y-g.bounds.Min.Y, g.cellSize, g.rows)
	row1 := clampIndex(r.Max.Y-1-g.bounds.Min.Y, g.cellSize, g.rows)

	for row := row0; row <= row1; row++ {
		for col := col0; col <= col1; col++ {
			fn(row*g.cols + col)
		}
	}
}

// clampIndex maps an offset from the grid origin to a cell index in [0, n),
// pinning offsets that fall outside the grid to the nearest edge cell.
func clampIndex(offset, cellSize, n int) int {
	if offset < 0 {
		return 0
	}
	return min(offset/cellSize, n-1)
}
