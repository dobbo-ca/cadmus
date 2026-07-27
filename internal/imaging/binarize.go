package imaging

import (
	"fmt"
	"math"
)

// OtsuThreshold returns the grayscale level maximising between-class variance,
// the standard Otsu criterion.
func OtsuThreshold(gray *Bitmap) uint8 {
	var hist [256]int
	for y := range gray.Height {
		for x := range gray.Width {
			hist[gray.At(x, y)]++
		}
	}
	total := gray.Width * gray.Height
	var sum float64
	for i, n := range hist {
		sum += float64(i * n)
	}

	var sumB float64
	var wB int
	var best float64
	var threshold uint8
	for t := range 256 {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += float64(t * hist[t])
		mB := sumB / float64(wB)
		mF := (sum - sumB) / float64(wF)
		between := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)
		if between > best {
			best = between
			threshold = uint8(t)
		}
	}
	return threshold
}

// Otsu binarizes gray at its Otsu threshold. In the result, 1 is foreground
// (ink): a pixel darker than or equal to the threshold becomes 1.
func Otsu(gray *Bitmap) *Bitmap {
	t := OtsuThreshold(gray)
	out := NewBitmap(gray.Width, gray.Height, 1)
	for y := range gray.Height {
		for x := range gray.Width {
			if gray.At(x, y) <= t {
				out.Set(x, y, 1)
			}
		}
	}
	return out
}

// Sauvola binarizes gray against a threshold computed separately for every
// pixel from the statistics of the window around it:
//
//	t = m * (1 - k * (1 - s/128))
//
// where m and s are the mean and standard deviation over that window. A pixel
// strictly darker than its own threshold becomes 1 (foreground/ink). Unlike
// Otsu this survives an illumination gradient, which is why it is the default
// for scanned pages; the flip side is that t is always below m, so a flat
// region is background however dark it is.
//
// window is the full edge length of the square window, rounded down to the
// nearest odd value; the effective window is 2*((window-1)/2)+1 and must be at
// least 5 and no wider than the bitmap. k is Sauvola's factor, typically
// between 0.2 and 0.5. Windows overhanging an edge are filled by reflecting the
// image about that edge with the edge pixel duplicated, so every pixel gets a
// full window.
//
// Cost per pixel is independent of the window size: the mean and mean square
// come from integral images. Where the arithmetic could round either way it
// rounds as Leptonica's pixSauvolaBinarize does — the mean truncated to a byte,
// the mean square rounded to an integer, the threshold truncated — so that the
// two agree pixel for pixel rather than merely closely.
func Sauvola(gray *Bitmap, window int, k float64) *Bitmap {
	half := (window - 1) / 2
	if half < 2 || half+1 > gray.Width || half+1 > gray.Height {
		panic(fmt.Sprintf("imaging: Sauvola window %d invalid for a %dx%d bitmap",
			window, gray.Width, gray.Height))
	}

	// Integral images over the reflected image, which is inset by border on
	// every side. Both are inclusive: cell (x, y) holds the sum over the
	// rectangle from the origin through (x, y). uint32 holds the value sum for
	// any page-sized bitmap, and float64 holds the square sum exactly, since
	// every partial sum is an integer well below 2^53.
	border := half + 1
	pw, ph := gray.Width+2*border, gray.Height+2*border
	sums := make([]uint32, pw*ph)
	squares := make([]float64, pw*ph)
	for y := range ph {
		sy := mirrorIndex(y-border, gray.Height)
		for x := range pw {
			v := uint32(gray.At(mirrorIndex(x-border, gray.Width), sy))
			s, q := v, float64(v)*float64(v)
			if x > 0 {
				s += sums[y*pw+x-1]
				q += squares[y*pw+x-1]
			}
			if y > 0 {
				s += sums[(y-1)*pw+x]
				q += squares[(y-1)*pw+x]
			}
			if x > 0 && y > 0 {
				s -= sums[(y-1)*pw+x-1]
				q -= squares[(y-1)*pw+x-1]
			}
			sums[y*pw+x] = s
			squares[y*pw+x] = q
		}
	}

	// A window of side incr centred on gray pixel (x, y) spans reflected
	// columns x+1..x+incr and rows y+1..y+incr, so the four corners of its
	// integral-image rectangle are at y and y+incr, x and x+incr.
	incr := 2*half + 1
	area := incr * incr
	meanNorm := float32(1 / float64(area))
	squareNorm := 1 / float64(area)

	out := NewBitmap(gray.Width, gray.Height, 1)
	for y := range gray.Height {
		top, bot := y*pw, (y+incr)*pw
		for x := range gray.Width {
			sum := sums[bot+x+incr] - sums[bot+x] - sums[top+x+incr] + sums[top+x]
			square := squares[bot+x+incr] - squares[bot+x] - squares[top+x+incr] + squares[top+x]

			mean := int(uint8(float32(sum) * meanNorm))
			meanSquare := int(uint32(squareNorm*square + 0.5))
			// var(v) = E(v^2) - E(v)^2. The two are quantised independently
			// above, so guard the subtraction rather than trusting the identity.
			variance := max(meanSquare-mean*mean, 0)
			sd := float64(float32(math.Sqrt(float64(variance))))

			threshold := int(float64(mean) * (1 - k*(1-sd/128)))
			if int(gray.At(x, y)) < threshold {
				out.Set(x, y, 1)
			}
		}
	}
	return out
}

// mirrorIndex maps t into [0, n) by reflecting about the edges with the edge
// element duplicated: -1 and 0 both give 0, n-1 and n both give n-1. Values
// more than n outside the range are not handled.
func mirrorIndex(t, n int) int {
	switch {
	case t < 0:
		return -1 - t
	case t >= n:
		return 2*n - 1 - t
	default:
		return t
	}
}
