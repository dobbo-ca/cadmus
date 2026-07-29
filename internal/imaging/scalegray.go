// This file is a Go translation of pixScale and the 8bpp routines it reaches,
// from Leptonica 1.87.0 (https://github.com/DanBloomberg/leptonica), licensed
// under the BSD 2-Clause License: src/scale1.c (pixScale, pixScaleGeneral,
// pixScaleGrayLI, pixScaleGray2xLI, pixScaleGray4xLI, pixScaleAreaMap,
// pixScaleAreaMap2, pixScaleSmooth and their low-level kernels), src/enhance.c
// (pixUnsharpMasking) and src/pix2.c (pixCopyBorder). The translation is not
// verbatim. See NOTICE.
//
// Recorded from the installed 1.87.0 source, as Task 12 Step 1 requires:
//
//  1. The identity short-circuit is exact equality with no tolerance —
//     pixScaleGeneral opens with
//     `if (scalex == 1.0 && scaley == 1.0) return pixCopy(NULL, pixs);`
//     so pixScale(pix, 1.0, 1.0) returns an unmodified copy and never
//     resamples. Note the comparison promotes the l_float32 factor to double,
//     which is reproduced below.
//
//  2. The 8bpp dispatch is NOT the three-way table the plan quoted from
//     master, and pixScale is not pixScaleGeneral: pixScale picks default
//     sharpening parameters and pixScaleGeneral sharpens with them. In full,
//     for scalex == scaley == f:
//
//     pixScale:         sharpfract = f < 0.7 ? 0.2 : 0.4
//     sharpwidth = f < 0.7 ? 1   : 2
//     pixScaleGeneral:  f == 1.0            -> pixCopy
//     f <  0.02           -> pixScaleSmooth
//     0.02 <= f < 0.7     -> pixScaleAreaMap
//     f >= 0.7            -> pixScaleGrayLI
//     then, on the reduction branch, pixUnsharpMasking when f > 0.2, and on
//     the interpolation branch when f < 1.4.
//
//     The reduction and interpolation routines have their own exact-factor
//     special cases, which are different kernels rather than fast paths for
//     the same arithmetic: pixScaleAreaMap cascades pixScaleAreaMap2 at
//     0.5/0.25/0.125/0.0625, and pixScaleGrayLI uses pixScaleGray2xLI at 2.0
//     and pixScaleGray4xLI at 4.0. A 36-pixel target height reaches 0.5 from a
//     72-pixel crop and 2.0 from an 18-pixel crop, so all of them are
//     transcribed here rather than clamped into a neighbour.
//
//     pixUnsharpMasking with halfwidth 1 or 2 routes through
//     pixUnsharpMaskingFast -> pixUnsharpMaskingGrayFast -> (direction is
//     L_BOTH_DIRECTIONS) pixUnsharpMaskingGray2D, so only the 2D form is
//     needed.
//
//  3. scaleGrayLILow and scaleGrayAreaMapLow live in src/scale1.c in 1.87.0,
//     not in the src/scalelow.c the plan named; that file no longer exists.
//     Their bodies are transcribed below as scaleGrayLILow and
//     scaleGrayAreaMapLow, statement for statement.
//
// Leptonica computes in l_float32 and truncates toward zero on every cast to
// l_int32, so the arithmetic below is float32 with explicit conversions at the
// rounding points C performs. Where a literal promotes an expression to double
// — `scy * (i + 1.0)`, `+ 0.5` — that promotion is reproduced too, because it
// changes the result.

package imaging

import "fmt"

// ScaleGray scales an 8bpp bitmap isotropically, reproducing Leptonica's
// pixScale(src, factor, factor). That is what ImageData::PreScale calls to fit
// a line crop to the network's input height, sharpening included, so the
// output dimensions come from Leptonica's own rounding rather than from
// round(factor * src.Height).
func ScaleGray(src *Bitmap, factor float64) *Bitmap {
	if src.Depth != 8 {
		panic(fmt.Sprintf("imaging: ScaleGray needs depth 8, got %d", src.Depth))
	}
	if factor <= 0 {
		panic(fmt.Sprintf("imaging: ScaleGray needs factor > 0, got %v", factor))
	}
	scale := float32(factor)
	// pixScale halves its default sharpening parameters below 0.7.
	sharpFract, sharpWidth := float32(0.4), 2
	if float64(scale) < 0.7 {
		sharpFract, sharpWidth = 0.2, 1
	}
	return scaleGeneral(src, scale, sharpFract, sharpWidth)
}

// scaleGeneral is pixScaleGeneral restricted to 8bpp and to equal x and y
// factors.
func scaleGeneral(src *Bitmap, scale, sharpFract float32, sharpWidth int) *Bitmap {
	if float64(scale) == 1.0 {
		return src.Clone()
	}
	var scaled *Bitmap
	var sharpen bool
	if float64(scale) < 0.7 { // low-pass filter for anti-aliasing
		if float64(scale) < 0.02 { // whole-pixel low-pass filter
			scaled = scaleSmooth(src, scale)
		} else { // fractional pixel low-pass filter
			scaled = scaleAreaMap(src, scale)
		}
		sharpen = float64(scale) > 0.2
	} else { // linear interpolation
		scaled = scaleGrayLI(src, scale)
		sharpen = float64(scale) < 1.4
	}
	if sharpen && sharpFract > 0 && sharpWidth > 0 {
		return unsharpMaskingGray2D(scaled, sharpWidth, sharpFract)
	}
	return scaled
}

// scaleGrayLI is pixScaleGrayLI. Its scale == 1.0 fast case is unreachable
// here, scaleGeneral having already short-circuited it.
func scaleGrayLI(src *Bitmap, scale float32) *Bitmap {
	if float64(scale) == 2.0 {
		return scaleGray2xLI(src)
	}
	if float64(scale) == 4.0 {
		return scaleGray4xLI(src)
	}
	dst := NewBitmap(scaledDim(scale, src.Width), scaledDim(scale, src.Height), 8)
	scaleGrayLILow(dst, src)
	return dst
}

// scaledDim is Leptonica's `(l_int32)(scale * (l_float32)n + 0.5)`: a float32
// product, then a double addition, then truncation.
func scaledDim(scale float32, n int) int {
	return int(float64(scale*float32(n)) + 0.5)
}

// scaleGrayLILow divides each src pixel into 16x16 sub-pixels and takes each
// dest pixel as the area-weighted blend of the four src pixels nearest its
// upper-left corner.
func scaleGrayLILow(dst, src *Bitmap) {
	ws, hs := src.Width, src.Height
	wd, hd := dst.Width, dst.Height
	// (scx, scy) map dest coords to src coords, in sixteenths of a pixel.
	scx := 16 * float32(ws) / float32(wd)
	scy := 16 * float32(hs) / float32(hd)
	wm2, hm2 := ws-2, hs-2

	for i := range hd {
		ypm := int(scy * float32(i))
		yp, yf := ypm>>4, ypm&0x0f
		for j := range wd {
			xpm := int(scx * float32(j))
			xp, xf := xpm>>4, xpm&0x0f

			v00val := int(src.At(xp, yp))
			var v01val, v10val, v11val int
			switch {
			case xp <= wm2 && yp <= hm2:
				v10val = int(src.At(xp+1, yp))
				v01val = int(src.At(xp, yp+1))
				v11val = int(src.At(xp+1, yp+1))
			case yp > hm2 && xp <= wm2: // pixels near the bottom
				v01val = v00val
				v10val = int(src.At(xp+1, yp))
				v11val = v10val
			case xp > wm2 && yp <= hm2: // pixels near the right side
				v01val = int(src.At(xp, yp+1))
				v10val = v00val
				v11val = v01val
			default: // pixels at the lower-right corner
				v10val, v01val, v11val = v00val, v00val, v00val
			}

			v00 := (16 - xf) * (16 - yf) * v00val
			v10 := xf * (16 - yf) * v10val
			v01 := (16 - xf) * yf * v01val
			v11 := xf * yf * v11val
			dst.Set(j, i, uint8((v00+v01+v10+v11+128)/256))
		}
	}
}

// scaleGray2xLI is pixScaleGray2xLI: 2x expansion where each src pixel becomes
// four dest pixels, taken as sp1, (sp1+sp2)/2, (sp1+sp3)/2 and
// (sp1+sp2+sp3+sp4)/4 over the src pixel and its right, lower and lower-right
// neighbours.
func scaleGray2xLI(src *Bitmap) *Bitmap {
	dst := NewBitmap(2*src.Width, 2*src.Height, 8)
	for i := range src.Height {
		scaleGray2xLILine(dst, 2*i, src, i, i == src.Height-1)
	}
	return dst
}

func scaleGray2xLILine(dst *Bitmap, id int, src *Bitmap, is int, lastLine bool) {
	wsm := src.Width - 1
	if lastLine {
		// No lower src line: both dest lines come from this one.
		sval2 := int(src.At(0, is))
		for j, jd := 0, 0; j < wsm; j, jd = j+1, jd+2 {
			sval1 := sval2
			sval2 = int(src.At(j+1, is))
			dst.Set(jd, id, uint8(sval1))
			dst.Set(jd, id+1, uint8(sval1))
			dst.Set(jd+1, id, uint8((sval1+sval2)/2))
			dst.Set(jd+1, id+1, uint8((sval1+sval2)/2))
		}
		last := uint8(sval2)
		dst.Set(2*wsm, id, last)
		dst.Set(2*wsm+1, id, last)
		dst.Set(2*wsm, id+1, last)
		dst.Set(2*wsm+1, id+1, last)
		return
	}

	sval2 := int(src.At(0, is))
	sval4 := int(src.At(0, is+1))
	for j, jd := 0, 0; j < wsm; j, jd = j+1, jd+2 {
		sval1, sval3 := sval2, sval4
		sval2 = int(src.At(j+1, is))
		sval4 = int(src.At(j+1, is+1))
		dst.Set(jd, id, uint8(sval1))
		dst.Set(jd+1, id, uint8((sval1+sval2)/2))
		dst.Set(jd, id+1, uint8((sval1+sval3)/2))
		dst.Set(jd+1, id+1, uint8((sval1+sval2+sval3+sval4)/4))
	}
	sval1, sval3 := sval2, sval4
	dst.Set(2*wsm, id, uint8(sval1))
	dst.Set(2*wsm+1, id, uint8(sval1))
	dst.Set(2*wsm, id+1, uint8((sval1+sval3)/2))
	dst.Set(2*wsm+1, id+1, uint8((sval1+sval3)/2))
}

// scaleGray4xLI is pixScaleGray4xLI: 4x expansion where each src pixel becomes
// sixteen dest pixels, bilinearly weighted over the src pixel and its right,
// lower and lower-right neighbours in quarters.
func scaleGray4xLI(src *Bitmap) *Bitmap {
	dst := NewBitmap(4*src.Width, 4*src.Height, 8)
	for i := range src.Height {
		scaleGray4xLILine(dst, 4*i, src, i, i == src.Height-1)
	}
	return dst
}

func scaleGray4xLILine(dst *Bitmap, id int, src *Bitmap, is int, lastLine bool) {
	wsm := src.Width - 1
	wsm4 := 4 * wsm
	if lastLine {
		// No lower src line: all four dest lines come from this one.
		s2 := int(src.At(0, is))
		for j, jd := 0, 0; j < wsm; j, jd = j+1, jd+4 {
			s1 := s2
			s2 = int(src.At(j+1, is))
			s1t, s2t := 3*s1, 3*s2
			for r := range 4 {
				dst.Set(jd, id+r, uint8(s1))
				dst.Set(jd+1, id+r, uint8((s1t+s2)/4))
				dst.Set(jd+2, id+r, uint8((s1+s2)/2))
				dst.Set(jd+3, id+r, uint8((s1+s2t)/4))
			}
		}
		last := uint8(s2)
		for r := range 4 {
			for k := range 4 {
				dst.Set(wsm4+k, id+r, last)
			}
		}
		return
	}

	s2 := int(src.At(0, is))
	s4 := int(src.At(0, is+1))
	for j, jd := 0, 0; j < wsm; j, jd = j+1, jd+4 {
		s1, s3 := s2, s4
		s2 = int(src.At(j+1, is))
		s4 = int(src.At(j+1, is+1))
		s1t, s2t, s3t, s4t := 3*s1, 3*s2, 3*s3, 3*s4
		dst.Set(jd, id, uint8(s1))
		dst.Set(jd+1, id, uint8((s1t+s2)/4))
		dst.Set(jd+2, id, uint8((s1+s2)/2))
		dst.Set(jd+3, id, uint8((s1+s2t)/4))
		dst.Set(jd, id+1, uint8((s1t+s3)/4))
		dst.Set(jd+1, id+1, uint8((9*s1+s2t+s3t+s4)/16))
		dst.Set(jd+2, id+1, uint8((s1t+s2t+s3+s4)/8))
		dst.Set(jd+3, id+1, uint8((s1t+9*s2+s3+s4t)/16))
		dst.Set(jd, id+2, uint8((s1+s3)/2))
		dst.Set(jd+1, id+2, uint8((s1t+s2+s3t+s4)/8))
		dst.Set(jd+2, id+2, uint8((s1+s2+s3+s4)/4))
		dst.Set(jd+3, id+2, uint8((s1+s2t+s3+s4t)/8))
		dst.Set(jd, id+3, uint8((s1+s3t)/4))
		dst.Set(jd+1, id+3, uint8((s1t+s2+9*s3+s4t)/16))
		dst.Set(jd+2, id+3, uint8((s1+s2+s3t+s4t)/8))
		dst.Set(jd+3, id+3, uint8((s1+s2t+s3t+9*s4)/16))
	}
	s1, s3 := s2, s4
	s1t, s3t := 3*s1, 3*s3
	for k := range 4 {
		dst.Set(wsm4+k, id, uint8(s1))
		dst.Set(wsm4+k, id+1, uint8((s1t+s3)/4))
		dst.Set(wsm4+k, id+2, uint8((s1+s3)/2))
		dst.Set(wsm4+k, id+3, uint8((s1+s3t)/4))
	}
}

// scaleAreaMap is pixScaleAreaMap for 8bpp. The exact 2x, 4x, 8x and 16x
// reductions are separate kernels, not fast paths: pixScaleAreaMap2 averages
// whole 2x2 blocks with a shift, where the general routine weights fractional
// pixels and rounds.
func scaleAreaMap(src *Bitmap, scale float32) *Bitmap {
	switch float64(scale) {
	case 0.5:
		return scaleAreaMap2(src)
	case 0.25:
		return scaleAreaMap2(scaleAreaMap2(src))
	case 0.125:
		return scaleAreaMap2(scaleAreaMap2(scaleAreaMap2(src)))
	case 0.0625:
		return scaleAreaMap2(scaleAreaMap2(scaleAreaMap2(scaleAreaMap2(src))))
	}
	dst := NewBitmap(scaledDim(scale, src.Width), scaledDim(scale, src.Height), 8)
	scaleGrayAreaMapLow(dst, src)
	return dst
}

// scaleAreaMap2 is pixScaleAreaMap2 for 8bpp: each dest pixel is the mean of a
// 2x2 src block, truncated by a right shift.
func scaleAreaMap2(src *Bitmap) *Bitmap {
	dst := NewBitmap(src.Width/2, src.Height/2, 8)
	for i := range dst.Height {
		for j := range dst.Width {
			val := int(src.At(2*j, 2*i)) + int(src.At(2*j+1, 2*i)) +
				int(src.At(2*j, 2*i+1)) + int(src.At(2*j+1, 2*i+1))
			dst.Set(j, i, uint8(val>>2))
		}
	}
	return dst
}

// scaleGrayAreaMapLow subdivides every src pixel into 256 sub-pixels and
// weights it by the number of sub-pixels the dest pixel covers.
func scaleGrayAreaMapLow(dst, src *Bitmap) {
	ws, hs := src.Width, src.Height
	wd, hd := dst.Width, dst.Height
	scx := 16 * float32(ws) / float32(wd)
	scy := 16 * float32(hs) / float32(hd)
	wm2, hm2 := ws-2, hs-2

	for i := range hd {
		// The lower corner promotes to double in Leptonica, the upper one does
		// not; the asymmetry is deliberate here because it is observable.
		yu := int(scy * float32(i))
		yl := int(float64(scy) * (float64(i) + 1.0))
		yup, yuf := yu>>4, yu&0x0f
		ylp, ylf := yl>>4, yl&0x0f
		dely := ylp - yup
		for j := range wd {
			xu := int(scx * float32(j))
			xl := int(float64(scx) * (float64(j) + 1.0))
			xup, xuf := xu>>4, xu&0x0f
			xlp, xlf := xl>>4, xl&0x0f
			delx := xlp - xup

			// If near the edge, just use a src pixel value.
			if xlp > wm2 || ylp > hm2 {
				dst.Set(j, i, src.At(xup, yup))
				continue
			}

			// The area summed over, in sub-pixels. Quantization makes it vary,
			// so it cannot be taken as the constant scx * scy.
			area := ((16 - xuf) + 16*(delx-1) + xlf) *
				((16 - yuf) + 16*(dely-1) + ylf)

			v00 := (16 - xuf) * (16 - yuf) * int(src.At(xup, yup))
			v10 := xlf * (16 - yuf) * int(src.At(xlp, yup))
			v01 := (16 - xuf) * ylf * int(src.At(xup, yup+dely))
			v11 := xlf * ylf * int(src.At(xlp, yup+dely))
			vin := 0
			for k := 1; k < dely; k++ { // full interior src pixels
				for m := 1; m < delx; m++ {
					vin += 256 * int(src.At(xup+m, yup+k))
				}
			}
			vmid := 0
			for k := 1; k < dely; k++ { // left side
				vmid += (16 - xuf) * 16 * int(src.At(xup, yup+k))
			}
			for k := 1; k < dely; k++ { // right side
				vmid += xlf * 16 * int(src.At(xlp, yup+k))
			}
			for m := 1; m < delx; m++ { // top side
				vmid += 16 * (16 - yuf) * int(src.At(xup+m, yup))
			}
			for m := 1; m < delx; m++ { // bottom side
				vmid += 16 * ylf * int(src.At(xup+m, yup+dely))
			}
			dst.Set(j, i, uint8((v00+v01+v10+v11+vin+vmid+128)/area))
		}
	}
}

// scaleSmooth is pixScaleSmooth for 8bpp: a flat filter whose width tracks the
// reduction ratio, evaluated only at the subsampling locations.
func scaleSmooth(src *Bitmap, scale float32) *Bitmap {
	ws, hs := src.Width, src.Height
	// The ideal filter full width is 1/scale, never below 2.
	isize := min(10000, max(2, int(float64(1.0/scale)+0.5)))
	if ws < isize || hs < isize {
		// Ridiculously small scaling factor: one pixel from the middle.
		dst := NewBitmap(1, 1, 8)
		dst.Set(0, 0, src.At(ws/2, hs/2))
		return dst
	}

	wd := max(1, scaledDim(scale, ws))
	hd := max(1, scaledDim(scale, hs))
	dst := NewBitmap(wd, hd, 8)
	norm := 1.0 / float32(isize*isize)
	wratio := float32(ws) / float32(wd)
	hratio := float32(hs) / float32(hd)

	// The UL corner of the src square that each dest pixel averages.
	srow := make([]int, hd)
	for i := range hd {
		srow[i] = min(int(hratio*float32(i)), hs-isize)
	}
	scol := make([]int, wd)
	for j := range wd {
		scol[j] = min(int(wratio*float32(j)), ws-isize)
	}

	for i := range hd {
		for j := range wd {
			val := 0
			for m := range isize {
				for n := range isize {
					val += int(src.At(scol[j]+n, srow[i]+m))
				}
			}
			dst.Set(j, i, uint8(int(float32(val)*norm)))
		}
	}
	return dst
}

// unsharpMaskingGray2D is pixUnsharpMaskingGray2D, the routine
// pixUnsharpMasking reaches for halfwidth 1 and 2. The low pass is computed
// separably; with L the low-pass value, I the src pixel and f the fraction,
// the sharpened value is I + f * (I - L).
func unsharpMaskingGray2D(src *Bitmap, halfwidth int, fract float32) *Bitmap {
	w, h := src.Width, src.Height
	// The pixels the loops below never reach keep their src values.
	dst := copyBorder(src, halfwidth)

	// Horizontal smoothing, into an intermediate float buffer. The columns the
	// loop skips stay zero and are never read back.
	sums := make([]float32, w*h)
	for i := range h {
		for j := halfwidth; j < w-halfwidth; j++ {
			total := 0
			for k := -halfwidth; k <= halfwidth; k++ {
				total += int(src.At(j+k, i))
			}
			sums[i*w+j] = float32(total)
		}
	}

	full := float32(2*halfwidth + 1)
	norm := 1.0 / (full * full)
	for i := halfwidth; i < h-halfwidth; i++ {
		for j := halfwidth; j < w-halfwidth; j++ {
			// Vertical smoothing finishes the low-pass filter. The additions
			// run left to right in float32, as Leptonica's do.
			var col float32
			for k := -halfwidth; k <= halfwidth; k++ {
				col = float32(col + sums[(i+k)*w+j])
			}
			lowpass := float32(norm * col)
			sval := float32(src.At(j, i))
			highpass := float32(fract * float32(sval-lowpass))
			ival := int(float64(float32(sval+highpass)) + 0.5)
			dst.Set(j, i, uint8(max(0, min(255, ival))))
		}
	}
	return dst
}

// copyBorder is pixCopyBorder with a single width: a zeroed bitmap of the same
// size as src whose outermost width rows and columns are copied from src.
func copyBorder(src *Bitmap, width int) *Bitmap {
	dst := NewBitmap(src.Width, src.Height, src.Depth)
	for y := range src.Height {
		inRow := y < width || y >= src.Height-width
		for x := range src.Width {
			if inRow || x < width || x >= src.Width-width {
				dst.Set(x, y, src.At(x, y))
			}
		}
	}
	return dst
}
