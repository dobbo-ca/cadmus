package imaging

import (
	"fmt"
	"math"
)

// Angles are in radians throughout this package, clockwise positive in a
// coordinate system whose y axis increases downward — the same convention
// Leptonica uses. The search bounds below are in degrees only because that is
// the unit page skew is naturally specified in.
const (
	skewSweepRangeDeg  = 5.0  // the coarse sweep covers +-this
	skewSweepStepDeg   = 0.1  // ... in steps of this
	skewRefineRangeDeg = 0.1  // the refinement covers +-this around the coarse best
	skewRefineStepDeg  = 0.01 // ... in steps of this
)

// SkewAngle estimates the skew of b, in radians: the angle by which its text
// lines are tilted away from horizontal, clockwise positive. Rotate(b,
// -SkewAngle(b)) therefore straightens the page.
//
// Note the sign. Leptonica's pixFindSkew returns the opposite quantity, the
// angle required to deskew, which is the negative of this one.
//
// The estimate is the angle that maximises the variance of the horizontal
// projection profile: text lines pile into few rows when they are level with
// the raster and smear across many when they are not, so the profile is
// spikiest at the true skew. Candidates are swept over +-5 degrees in 0.1
// degree steps and then refined over +-0.1 degrees in 0.01 degree steps around
// the coarse winner, so 0.01 degrees is the resolution of the result. Tilt
// beyond 5 degrees is not page skew and is not looked for.
//
// An empty bitmap has no skew and returns 0. b must be depth 1.
func SkewAngle(b *Bitmap) float64 {
	if b.Depth != 1 {
		panic(fmt.Sprintf("imaging: SkewAngle needs a depth-1 bitmap, got depth %d", b.Depth))
	}

	// Collect the foreground once. Every candidate angle then costs a pass over
	// the ink rather than over the page, which is the difference between
	// scoring 122 angles and scoring a handful of them.
	type point struct{ x, y int32 }
	var ink []point
	for y := range b.Height {
		for x := range b.Width {
			if b.At(x, y) != 0 {
				ink = append(ink, point{int32(x), int32(y)})
			}
		}
	}
	if len(ink) == 0 {
		return 0
	}

	// Scoring shears the page vertically rather than rotating it: a pixel at
	// (x, y) lands in row y + (x - xc)*tan(angle), which is the row a real
	// rotation would put it in up to an overall 1/cos(angle) scaling of the
	// row axis, and that scaling cannot move the maximum because it applies to
	// every row alike.
	//
	// Ink is split between the two rows straddling that position in proportion
	// to the fractional part, rather than dropped into the nearer one. Rounding
	// to a whole row instead makes the score a staircase in the angle, and on a
	// page whose lines are only a few hundred pixels wide the steps are wider
	// than the accuracy being asked for: measured against known rotations of
	// the golden page, rounding is off by up to 0.08 degrees where splitting is
	// off by 0.01, the sweep's own resolution.
	//
	// The profile is padded by the largest shift any candidate can produce, so
	// no ink is ever dropped off the ends and every candidate is scored over
	// the same number of bins. The profile's mean is then the same for all of
	// them, and maximising the variance reduces to maximising the sum of
	// squares.
	pad := int(float64(b.Width)/2*
		math.Tan((skewSweepRangeDeg+skewRefineRangeDeg)*math.Pi/180)) + 1
	profile := make([]float64, b.Height+2*pad+1)
	lo := make([]int, b.Width)
	frac := make([]float64, b.Width)
	xc := float64(b.Width) / 2

	score := func(deg float64) float64 {
		tan := math.Tan(deg * math.Pi / 180)
		for x := range b.Width {
			shift := (float64(x) - xc) * tan
			floor := math.Floor(shift)
			lo[x] = int(floor) + pad
			frac[x] = shift - floor
		}
		clear(profile)
		for _, p := range ink {
			i, f := int(p.y)+lo[p.x], frac[p.x]
			profile[i] += 1 - f
			profile[i+1] += f
		}
		var sum float64
		for _, n := range profile {
			sum += n * n
		}
		return sum
	}

	// The angle being searched for is the one that straightens the page; the
	// skew the page carries is its negative, which is what gets returned.
	best, bestScore := 0.0, math.Inf(-1)
	sweep := func(centre, radius, step float64) {
		for i := 0; i <= int(math.Round(2*radius/step)); i++ {
			deg := centre - radius + float64(i)*step
			if s := score(deg); s > bestScore {
				bestScore, best = s, deg
			}
		}
	}
	sweep(0, skewSweepRangeDeg, skewSweepStepDeg)
	sweep(best, skewRefineRangeDeg, skewRefineStepDeg)

	return -best * math.Pi / 180
}

// Rotate returns b rotated about its centre by radians, clockwise for a
// positive angle. The result is the same size as b, so the corners are cut off
// and background is brought in behind them.
//
// This is a nearest-neighbour rotation: every destination pixel takes the
// value of the source pixel the inverse rotation lands on, with the centre at
// (Width/2, Height/2) and no interpolation. It matches Leptonica's
// pixRotateBySampling, which is what pixRotate itself uses for 1 bpp beyond
// about 6 degrees. Below that pixRotate substitutes a three-shear
// approximation for speed; this does not, so small rotations here are true
// rotations rather than sheared ones.
//
// b must be depth 1.
func Rotate(b *Bitmap, radians float64) *Bitmap {
	if b.Depth != 1 {
		panic(fmt.Sprintf("imaging: Rotate needs a depth-1 bitmap, got depth %d", b.Depth))
	}

	// The source coordinate is truncated, not rounded, so a coordinate that
	// lands a hair either side of a whole pixel selects a different source row.
	// Leptonica carries that coordinate in float32; the products here are taken
	// in float64 and the result rounded to float32 once, which is both the more
	// accurate way to reach the same precision and what makes the result match
	// pixRotateBySampling pixel for pixel.
	angle := float32(radians)
	sina := float64(float32(math.Sin(float64(angle))))
	cosa := float64(float32(math.Cos(float64(angle))))

	out := NewBitmap(b.Width, b.Height, 1)
	xcen, ycen := b.Width/2, b.Height/2
	for i := range b.Height {
		ydif := float64(ycen - i)
		for j := range b.Width {
			xdif := float64(xcen - j)
			x := xcen + int(float32(-xdif*cosa-ydif*sina))
			if x < 0 || x >= b.Width {
				continue
			}
			y := ycen + int(float32(-ydif*cosa+xdif*sina))
			if y < 0 || y >= b.Height {
				continue
			}
			if b.At(x, y) != 0 {
				out.Set(j, i, 1)
			}
		}
	}
	return out
}
