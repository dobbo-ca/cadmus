package imaging

import (
	"fmt"
	"image"
)

// Component is one connected run of foreground pixels: its bounding box and the
// number of foreground pixels inside it. Pixels counts only the component's own
// pixels, so it is at most the area of Bounds and is smaller whenever the shape
// is not a solid rectangle.
type Component struct {
	Bounds image.Rectangle
	Pixels int
}

// connNeighbours lists the pixel offsets each connectivity treats as adjacent:
// edge neighbours only for 4, edge and corner neighbours for 8.
var connNeighbours = map[int][]image.Point{
	4: {{X: -1}, {X: 1}, {Y: -1}, {Y: 1}},
	8: {
		{X: -1, Y: -1}, {Y: -1}, {X: 1, Y: -1},
		{X: -1}, {X: 1},
		{X: -1, Y: 1}, {Y: 1}, {X: 1, Y: 1},
	},
}

// ConnComp returns the connected foreground components of a depth-1 bitmap.
// connectivity is 4 or 8.
//
// Components come back in raster order of their first foreground pixel: by row
// first, then by column within that row. That is the order Leptonica's
// pixConnComp produces, and it is not the same as sorting by Bounds.Min, since
// a component's leftmost pixel need not lie on its topmost row.
//
// An all-background bitmap yields an empty slice. A single foreground pixel is
// a component like any other. A component that reaches the image edge keeps its
// full extent: the edge bounds the scan, it does not cut the component.
func ConnComp(b *Bitmap, connectivity int) []Component {
	if b.Depth != 1 {
		panic(fmt.Sprintf("imaging: ConnComp needs a depth-1 bitmap, got depth %d", b.Depth))
	}
	deltas, ok := connNeighbours[connectivity]
	if !ok {
		panic(fmt.Sprintf("imaging: unsupported connectivity %d, want 4 or 8", connectivity))
	}

	var comps []Component
	// seen marks pixels already assigned to a component, so each is visited
	// once and the raster scan never restarts a component it has finished.
	seen := make([]bool, b.Width*b.Height)
	var stack []image.Point

	for y := range b.Height {
		for x := range b.Width {
			if seen[y*b.Width+x] || b.At(x, y) == 0 {
				continue
			}
			// Flood fill from this seed. Neighbours are marked as they are
			// pushed rather than as they are popped, so a pixel reachable by
			// several paths is queued only once.
			seen[y*b.Width+x] = true
			stack = append(stack[:0], image.Pt(x, y))
			minX, maxX, minY, maxY := x, x, y, y
			pixels := 0

			for len(stack) > 0 {
				p := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				pixels++
				minX = min(minX, p.X)
				maxX = max(maxX, p.X)
				minY = min(minY, p.Y)
				maxY = max(maxY, p.Y)

				for _, d := range deltas {
					nx, ny := p.X+d.X, p.Y+d.Y
					if nx < 0 || nx >= b.Width || ny < 0 || ny >= b.Height {
						continue
					}
					if seen[ny*b.Width+nx] || b.At(nx, ny) == 0 {
						continue
					}
					seen[ny*b.Width+nx] = true
					stack = append(stack, image.Pt(nx, ny))
				}
			}

			comps = append(comps, Component{
				Bounds: image.Rect(minX, minY, maxX+1, maxY+1),
				Pixels: pixels,
			})
		}
	}
	return comps
}
