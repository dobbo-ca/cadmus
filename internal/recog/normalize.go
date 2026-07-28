// This file is a Go translation of NetworkIO::FromPixes,
// NetworkIO::Copy2DImage, NetworkIO::SetPixel and ComputeBlackWhite in
// src/lstm/networkio.cpp, Input::PreparePixInput in src/lstm/input.cpp,
// ImageData::PreScale in src/ccstruct/imagedata.cpp, and STATS::ile in
// src/ccstruct/statistc.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"
	"image"

	"github.com/dobbo-ca/cadmus/internal/imaging"
	"github.com/dobbo-ca/cadmus/internal/nn"
)

// Normalized is a line image prepared for the network.
type Normalized struct {
	// Input is the network's input tensor: InputHeight rows by the scaled
	// image's width, one feature per pixel.
	Input *nn.Tensor
	// ScaleFactor converts a network timestep to a horizontal distance in the
	// original line image: XScale divided by the scaling factor applied.
	ScaleFactor float64
	Black       float32
	Contrast    float32
}

// Normalize scales img isotropically towards inputHeight pixels tall, contrast
// stretches it from a single mid-height scanline, and packs it into the
// network's input tensor.
//
// There is deliberately no centring, no padding, no fixed width and no
// mean/standard-deviation normalization: Tesseract does none of those, and the
// output width is simply whatever the isotropic scale produces.
//
// The tensor's height is inputHeight even when the scaler produces something
// else. NetworkIO::FromPixes builds the stride map from the StaticShape
// (height 36, width 0 → the pix's own width), and Copy2DImage then iterates
// `y < target_height`, filling every row past pixGetHeight(pix) with
// NetworkIO::Randomize. Those draws come from the same TRand that Convolve
// consumes, so skipping them would desynchronise the edge padding of the whole
// feature map — which is exactly the failure Task 14's per-layer diff is least
// able to explain.
func Normalize(img image.Image, inputHeight, xScale int, rnd *nn.Rand) (*Normalized, error) {
	if rnd == nil {
		return nil, fmt.Errorf("recog: Normalize needs the recognizer's randomizer")
	}
	grey := imaging.FromImage(img)
	if grey.Height <= 0 || grey.Width <= 0 {
		return nil, fmt.Errorf("recog: empty line image %dx%d", grey.Width, grey.Height)
	}
	imFactor := float64(float32(inputHeight) / float32(grey.Height))
	scaled := imaging.ScaleGray(grey, imFactor)
	// Input::PrepareLSTMInputs' only rejection, with min_width =
	// network->XScaleFactor(). Note it is OR, and it tests the SCALED
	// dimensions, both of them.
	if scaled.Width < xScale || scaled.Height < xScale {
		return nil, fmt.Errorf("recog: scaled line is %dx%d, below the network's minimum of %d in either dimension",
			scaled.Width, scaled.Height, xScale)
	}

	black, white := computeBlackWhite(scaled)
	contrast := (white - black) / 2
	if contrast <= 0 {
		contrast = 1
	}

	// The map width is the scaled pix's own width: FromPixes uses shape.width()
	// only when it is non-zero, and eng's is 0. Copy2DImage's
	// `if (width > target_width) width = target_width` is therefore a no-op
	// here; a model with a fixed input width would need it, and Build already
	// hard-errors on anything other than a height-fixed, width-free shape.
	in := nn.NewTensor(nn.StrideMap{Height: inputHeight, Width: scaled.Width}, 1)
	v := make([]float64, 1)
	for y := range inputHeight {
		x := 0
		if y < scaled.Height {
			for ; x < in.Map.Width; x++ {
				// SetPixel's arithmetic is float32 throughout, and the result
				// is deliberately NOT clipped: a pixel outside
				// [black, black+2*contrast] lands outside [-1, 1].
				fp := (float32(scaled.At(x, y))-black)/contrast - 1
				v[0] = float64(fp)
				in.WriteTimeStep(in.Map.T(y, x), v)
			}
		}
		// NetworkIO::Randomize for the tail of a short row, and for every column
		// of a row the scaler never produced.
		for ; x < in.Map.Width; x++ {
			v[0] = rnd.SignedRand(1.0)
			in.WriteTimeStep(in.Map.T(y, x), v)
		}
	}

	return &Normalized{
		Input:       in,
		ScaleFactor: float64(xScale) / imFactor,
		Black:       black,
		Contrast:    contrast,
	}, nil
}

// computeBlackWhite is ComputeBlackWhite. It reads exactly one scanline, at
// height/2, on the assumption that a horizontal line through the middle of a
// single text line passes through some ink.
func computeBlackWhite(b *imaging.Bitmap) (black, white float32) {
	var mins, maxes stats
	if b.Width >= 3 {
		y := b.Height / 2
		prev := int(b.At(0, y))
		curr := int(b.At(1, y))
		for x := 1; x+1 < b.Width; x++ {
			next := int(b.At(x+1, y))
			if (curr < prev && curr <= next) || (curr <= prev && curr < next) {
				mins.add(curr, 1)
			}
			if (curr > prev && curr >= next) || (curr >= prev && curr > next) {
				maxes.add(curr, 1)
			}
			prev, curr = curr, next
		}
	}
	if mins.total == 0 {
		mins.add(0, 1)
	}
	if maxes.total == 0 {
		maxes.add(255, 1)
	}
	return float32(mins.ile(0.25)), float32(maxes.ile(0.75))
}

// stats is STATS over the fixed bucket range [0, 255] that ComputeBlackWhite
// uses. Only add and ile are needed.
type stats struct {
	buckets [256]int
	total   int
}

func (s *stats) add(value, count int) {
	if value < 0 {
		value = 0
	} else if value > 255 {
		value = 255
	}
	s.buckets[value] += count
	s.total += count
}

// ile is STATS::ile: the fractile value such that frac of the samples are
// below it, interpolated linearly inside the bucket that crosses the target.
// The target is clipped to at least 1, so ile(0) is not the range minimum.
func (s *stats) ile(frac float64) float64 {
	if s.total == 0 {
		return 0
	}
	target := frac * float64(s.total)
	if target < 1 {
		target = 1
	} else if target > float64(s.total) {
		target = float64(s.total)
	}
	sum, index := 0, 0
	for index < len(s.buckets) && float64(sum) < target {
		sum += s.buckets[index]
		index++
	}
	if index > 0 {
		return float64(index) - (float64(sum)-target)/float64(s.buckets[index-1])
	}
	return 0
}
