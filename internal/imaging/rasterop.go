package imaging

import (
	"fmt"
	"image"
)

// Op is a bitwise raster operation combining a source and a destination pixel.
//
// The constant values are Leptonica's own op codes, and they are a truth table
// rather than an arbitrary enumeration: bit (2*src + dst) of the code is the
// result for that combination of input bits. OpSrcAndDst (0x8), for instance,
// has only the bit for src=1, dst=1 set.
type Op int

const (
	OpClear     Op = 0x0 // 0
	OpNotSrc    Op = 0x3 // ^src
	OpSrcXorDst Op = 0x6 // src ^ dst
	OpSrcAndDst Op = 0x8 // src & dst
	OpSrc       Op = 0xc // src
	OpSrcOrDst  Op = 0xe // src | dst
	OpSet       Op = 0xf // 1
)

// usesSrc reports whether the op reads the source image. The two that do not
// are clipped against the destination alone, which is the one place where
// Leptonica's clipping for an op is not the general two-image clip.
func (op Op) usesSrc() bool { return op != OpClear && op != OpSet }

// apply combines one source and one destination value. The masks are full
// bytes so the same expression serves depth 1, where At returns 0 or 1 and Set
// keeps only the low bit, and depth 8, where the operation is bitwise across
// the whole byte exactly as Leptonica's word-at-a-time blitters make it.
func (op Op) apply(s, d uint8) uint8 {
	var v uint8
	if op&0x1 != 0 {
		v |= ^s & ^d
	}
	if op&0x2 != 0 {
		v |= ^s & d
	}
	if op&0x4 != 0 {
		v |= s & ^d
	}
	if op&0x8 != 0 {
		v |= s & d
	}
	return v
}

// RasterOp combines the source rectangle rooted at sp into the destination
// rectangle dr, pixel by pixel, under op. The source rectangle has the same
// size as dr. dst is modified in place; src is not read for OpSet and OpClear
// and may be nil for those.
//
// Rectangles are clipped, not rejected, and the clipping is the part worth
// getting right — it is what Leptonica's pixRasterop does, and callers rely on
// it to blit near the edge of an image without bounds-checking themselves.
// Both rectangles are trimmed until they lie inside their images, keeping them
// the same size and aligned with each other: a destination corner left of or
// above the origin advances the source origin by the same amount (and the
// reverse for a negative source origin), and an overhang past the right or
// bottom edge of either image shortens both. If nothing survives, dst is
// untouched.
//
// The two bitmaps must have the same depth.
func RasterOp(dst *Bitmap, dr image.Rectangle, op Op, src *Bitmap, sp image.Point) {
	if !op.usesSrc() {
		// Leptonica clips a source-less op against the destination alone.
		// Pointing the source at the destination, aligned with dr, reproduces
		// that: the source origin then moves in step with the destination and
		// the two overhang tests become the same test. The op's result does
		// not depend on the source value, so what is read there is immaterial.
		src, sp = dst, dr.Min
	}
	if src == nil {
		panic(fmt.Sprintf("imaging: RasterOp with op %#x needs a source bitmap", int(op)))
	}
	if src.Depth != dst.Depth {
		panic(fmt.Sprintf("imaging: RasterOp between depth %d and depth %d bitmaps", src.Depth, dst.Depth))
	}

	dx, dy := dr.Min.X, dr.Min.Y
	sx, sy := sp.X, sp.Y
	dw, dh := dr.Dx(), dr.Dy()

	// Clip to the largest rectangle inside both images.
	if dx < 0 {
		sx -= dx
		dw += dx
		dx = 0
	}
	if sx < 0 {
		dx -= sx
		dw += sx
		sx = 0
	}
	dw = min(dw, dst.Width-dx, src.Width-sx)

	if dy < 0 {
		sy -= dy
		dh += dy
		dy = 0
	}
	if sy < 0 {
		dy -= sy
		dh += sy
		sy = 0
	}
	dh = min(dh, dst.Height-dy, src.Height-sy)

	for y := range dh {
		for x := range dw {
			dst.Set(dx+x, dy+y, op.apply(src.At(sx+x, sy+y), dst.At(dx+x, dy+y)))
		}
	}
}
