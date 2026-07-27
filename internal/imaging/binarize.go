package imaging

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
