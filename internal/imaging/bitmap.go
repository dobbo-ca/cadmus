// Package imaging provides Cadmus's pure-Go image primitives: the Bitmap
// substrate plus the binarization, morphology, and analysis operators built on
// it. Coordinates are (x, y) with origin top-left, matching image.Rectangle and
// Leptonica.
package imaging

import (
	"fmt"
	"image"
	"image/color"
)

// Bitmap is an 8-bit grayscale or 1-bit bilevel image. Depth is 1 or 8.
//
// In a depth-1 bitmap, 1 means foreground (ink) and 0 means background. This
// matches Leptonica's convention for binary images and inverts the usual
// "black is 0" grayscale intuition; every operator in this package relies on
// it.
//
// Depth-1 rows are packed MSB-first: pixel x of a row lives in bit 7-(x%8) of
// byte x/8. Rows are Stride bytes apart. Coordinates passed to At and Set must
// lie inside Bounds.
type Bitmap struct {
	Width, Height int
	Depth         int
	Stride        int // bytes per row
	Pix           []byte
}

// NewBitmap allocates a zeroed w x h bitmap of the given depth, which must be
// 1 or 8.
func NewBitmap(w, h, depth int) *Bitmap {
	if depth != 1 && depth != 8 {
		panic(fmt.Sprintf("imaging: unsupported depth %d", depth))
	}
	stride := (w*depth + 7) / 8
	return &Bitmap{
		Width:  w,
		Height: h,
		Depth:  depth,
		Stride: stride,
		Pix:    make([]byte, stride*h),
	}
}

// FromImage converts any image.Image to an 8bpp grayscale bitmap.
func FromImage(img image.Image) *Bitmap {
	r := img.Bounds()
	b := NewBitmap(r.Dx(), r.Dy(), 8)
	for y := range b.Height {
		for x := range b.Width {
			g := color.GrayModel.Convert(img.At(r.Min.X+x, r.Min.Y+y)).(color.Gray)
			b.Pix[y*b.Stride+x] = g.Y
		}
	}
	return b
}

// At returns the value at (x, y): 0 or 1 for depth 1, 0-255 for depth 8.
func (b *Bitmap) At(x, y int) uint8 {
	if b.Depth == 1 {
		return (b.Pix[y*b.Stride+x/8] >> (7 - uint(x)%8)) & 1
	}
	return b.Pix[y*b.Stride+x]
}

// Set writes v at (x, y). For depth 1 only the low bit of v is used.
func (b *Bitmap) Set(x, y int, v uint8) {
	if b.Depth == 1 {
		i := y*b.Stride + x/8
		mask := byte(1) << (7 - uint(x)%8)
		if v&1 != 0 {
			b.Pix[i] |= mask
		} else {
			b.Pix[i] &^= mask
		}
		return
	}
	b.Pix[y*b.Stride+x] = v
}

// Bounds returns the bitmap's extent, always rooted at the origin.
func (b *Bitmap) Bounds() image.Rectangle { return image.Rect(0, 0, b.Width, b.Height) }

// Clone returns a deep copy.
func (b *Bitmap) Clone() *Bitmap {
	out := *b
	out.Pix = make([]byte, len(b.Pix))
	copy(out.Pix, b.Pix)
	return &out
}
