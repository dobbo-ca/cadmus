package imaging

import (
	"fmt"
	"image"
)

// SeedFill returns the binary reconstruction of seed under mask: every
// connected component of mask holding at least one seed pixel, in full, and
// nothing else. connectivity is 4 or 8 and describes the mask's components.
//
// A seed pixel that falls on mask background selects nothing, so the result is
// always a subset of mask. seed and mask must both be depth 1 and the same
// size; neither is modified.
//
// This is Leptonica's pixSeedfillBinary. That one sweeps the image
// raster/anti-raster until nothing changes, up to a fixed cap of 40 sweeps;
// this one floods each seeded component directly, so it reaches the full
// reconstruction whatever the mask's topology and costs one visit per mask
// pixel.
func SeedFill(seed, mask *Bitmap, connectivity int) *Bitmap {
	if seed.Depth != 1 || mask.Depth != 1 {
		panic(fmt.Sprintf("imaging: SeedFill needs depth-1 bitmaps, got seed depth %d and mask depth %d",
			seed.Depth, mask.Depth))
	}
	if seed.Width != mask.Width || seed.Height != mask.Height {
		panic(fmt.Sprintf("imaging: SeedFill seed is %dx%d but mask is %dx%d",
			seed.Width, seed.Height, mask.Width, mask.Height))
	}
	deltas, ok := connNeighbours[connectivity]
	if !ok {
		panic(fmt.Sprintf("imaging: unsupported connectivity %d, want 4 or 8", connectivity))
	}

	out := NewBitmap(mask.Width, mask.Height, 1)
	var stack []image.Point

	for y := range mask.Height {
		for x := range mask.Width {
			// A seed pixel off the mask is dropped, and one inside a component
			// already filled costs nothing: the flood below marks pixels as it
			// pushes them, so out doubles as the visited set.
			if seed.At(x, y) == 0 || mask.At(x, y) == 0 || out.At(x, y) != 0 {
				continue
			}
			out.Set(x, y, 1)
			stack = append(stack[:0], image.Pt(x, y))

			for len(stack) > 0 {
				p := stack[len(stack)-1]
				stack = stack[:len(stack)-1]

				for _, d := range deltas {
					nx, ny := p.X+d.X, p.Y+d.Y
					if nx < 0 || nx >= mask.Width || ny < 0 || ny >= mask.Height {
						continue
					}
					if out.At(nx, ny) != 0 || mask.At(nx, ny) == 0 {
						continue
					}
					out.Set(nx, ny, 1)
					stack = append(stack, image.Pt(nx, ny))
				}
			}
		}
	}
	return out
}

// DistanceFunction returns the city-block distance from every foreground pixel
// of b to the nearest background pixel, as a depth-8 bitmap. Background pixels
// are 0 and the distance grows from 1 at a component's edge.
//
// Everything outside the image counts as background, so a foreground pixel on
// the image border is at distance 1 — Leptonica's L_BOUNDARY_BG. Distances
// saturate at 255 rather than wrapping.
//
// The metric is the 4-connected one, matching
// pixDistanceFunction(pixs, 4, 8, L_BOUNDARY_BG). Chessboard distance, the
// 8-connected variant, is not offered until something needs it.
//
// b must be depth 1.
func DistanceFunction(b *Bitmap) *Bitmap {
	if b.Depth != 1 {
		panic(fmt.Sprintf("imaging: DistanceFunction needs a depth-1 bitmap, got depth %d", b.Depth))
	}

	// Seed every foreground pixel at 1, then relax in two sweeps: a raster
	// sweep propagates distances from above and from the left, an anti-raster
	// sweep from below and from the right. For the city-block metric those two
	// passes are exact — every shortest path to background is monotone in one
	// of the two scan directions.
	out := NewBitmap(b.Width, b.Height, 8)
	for y := range b.Height {
		for x := range b.Width {
			if b.At(x, y) != 0 {
				out.Set(x, y, 1)
			}
		}
	}

	// The one-pixel border keeps the 1 it was seeded with, which is already its
	// distance from the background outside the image, so both sweeps run over
	// the interior only.
	for y := 1; y < b.Height-1; y++ {
		for x := 1; x < b.Width-1; x++ {
			if out.At(x, y) == 0 {
				continue
			}
			m := int(min(out.At(x, y-1), out.At(x-1, y)))
			out.Set(x, y, uint8(min(m, 254)+1))
		}
	}
	for y := b.Height - 2; y > 0; y-- {
		for x := b.Width - 2; x > 0; x-- {
			v := int(out.At(x, y))
			if v == 0 {
				continue
			}
			m := int(min(out.At(x, y+1), out.At(x+1, y)))
			out.Set(x, y, uint8(min(m+1, v)))
		}
	}
	return out
}
