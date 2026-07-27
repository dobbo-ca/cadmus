package imaging

import "fmt"

// axis selects the direction a one-dimensional pass runs in.
type axis int

const (
	axisX axis = iota
	axisY
)

// Dilate returns b dilated by a solid w x h brick: a pixel of the result is
// foreground when the brick, placed over it, covers any foreground pixel.
//
// The brick's origin sits at (w/2, h/2), which leaves more of an even-sided
// brick to the left of its origin than to the right: a 4x2 brick spreads
// foreground 2 columns left and 1 right, 1 row up and none down. Pixels
// outside the image are background, matching Leptonica's default asymmetric
// boundary condition.
//
// b must be depth 1, and w and h at least 1. The operation is separable, so
// the cost is independent of the brick size.
func Dilate(b *Bitmap, w, h int) *Bitmap {
	checkBrick(b, w, h)
	// Dilation places the brick so that its origin lands on the pixel being
	// written, which reflects the window: a hit j columns into the brick
	// contributes the source pixel w/2 - j columns away.
	t := brickPass(b, w-1-w/2, w/2, 1, axisX)
	return brickPass(t, h-1-h/2, h/2, 1, axisY)
}

// Erode returns b eroded by a solid w x h brick: a pixel of the result is
// foreground only when the brick, placed over it, covers foreground
// everywhere.
//
// Pixels outside the image count as background, so erosion clears a border of
// w/2 on the left, w-1-w/2 on the right and the corresponding rows top and
// bottom. That is Leptonica's default asymmetric boundary condition; it is
// also why closing can eat foreground near an edge.
//
// b must be depth 1, and w and h at least 1.
func Erode(b *Bitmap, w, h int) *Bitmap {
	checkBrick(b, w, h)
	t := brickPass(b, w/2, w-1-w/2, w, axisX)
	return brickPass(t, h/2, h-1-h/2, h, axisY)
}

// Open returns b eroded then dilated by a w x h brick, which deletes
// foreground the brick does not fit inside and leaves the rest where it was.
func Open(b *Bitmap, w, h int) *Bitmap { return Dilate(Erode(b, w, h), w, h) }

// Close returns b dilated then eroded by a w x h brick, which joins foreground
// separated by less than the brick without growing its outer extent.
func Close(b *Bitmap, w, h int) *Bitmap { return Erode(Dilate(b, w, h), w, h) }

// brickPass runs one axis of a separable brick operation. A pixel of the
// result is set when at least minHits foreground pixels lie in the window
// spanning [k-before, k+after] along the axis, where k is the pixel's own
// coordinate.
//
// Both morphological operations fall out of that one rule. Dilation passes
// minHits 1: any hit is enough. Erosion passes the window size: every position
// must hit, and since positions outside the image are never counted, a window
// that overhangs an edge can never reach the total — which is exactly the
// asymmetric boundary condition, with no special-casing of the border.
//
// The window slides, so each pixel costs a constant number of reads whatever
// the brick size.
func brickPass(src *Bitmap, before, after, minHits int, along axis) *Bitmap {
	out := NewBitmap(src.Width, src.Height, 1)

	// n is the length of a line along the axis, lines the number of them.
	n, lines := src.Width, src.Height
	get := func(line, k int) uint8 { return src.At(k, line) }
	set := func(line, k int) { out.Set(k, line, 1) }
	if along == axisY {
		n, lines = src.Height, src.Width
		get = func(line, k int) uint8 { return src.At(line, k) }
		set = func(line, k int) { out.Set(line, k, 1) }
	}

	for line := range lines {
		count := 0
		for k := 0; k <= after && k < n; k++ {
			if get(line, k) != 0 {
				count++
			}
		}
		for k := range n {
			if count >= minHits {
				set(line, k)
			}
			if drop := k - before; drop >= 0 && get(line, drop) != 0 {
				count--
			}
			if add := k + after + 1; add < n && get(line, add) != 0 {
				count++
			}
		}
	}
	return out
}

func checkBrick(b *Bitmap, w, h int) {
	if b.Depth != 1 {
		panic(fmt.Sprintf("imaging: morphology needs a depth-1 bitmap, got depth %d", b.Depth))
	}
	if w < 1 || h < 1 {
		panic(fmt.Sprintf("imaging: brick %dx%d must be at least 1x1", w, h))
	}
}
