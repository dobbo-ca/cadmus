package imaging

import "testing"

func TestScaleGrayMatchesLeptonica(t *testing.T) {
	src := loadGolden(t, "gray.bin")
	up := loadGolden(t, "scale_up_in.bin")
	for _, tc := range []struct {
		name   string
		src    *Bitmap
		factor float64
	}{
		{"scale_identity", src, 1.0},
		{"scale_li", src, 0.80},
		{"scale_areamap2", src, 0.50},
		{"scale_areamap", src, 0.35},
		{"scale_smooth", src, 0.015},
		{"scale_2xli", up, 2.0},
		{"scale_4xli", up, 4.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := loadGolden(t, tc.name+".bin")
			got := ScaleGray(tc.src, tc.factor)
			if got.Width != want.Width || got.Height != want.Height {
				t.Fatalf("ScaleGray(%v) size = %dx%d; want %dx%d", tc.factor, got.Width, got.Height, want.Width, want.Height)
			}
			var diff, maxDelta int
			for y := range got.Height {
				for x := range got.Width {
					d := int(got.At(x, y)) - int(want.At(x, y))
					if d != 0 {
						diff++
						if d < 0 {
							d = -d
						}
						if d > maxDelta {
							maxDelta = d
						}
					}
				}
			}
			if diff != 0 {
				total := got.Width * got.Height
				t.Errorf("ScaleGray(%v) differs from Leptonica in %d of %d pixels (%.4f%%), max delta %d grey levels",
					tc.factor, diff, total, 100*float64(diff)/float64(total), maxDelta)
			}
		})
	}
}

// The identity case must be byte-exact and must not resample, because Task 17's
// 36-pixel corpus relies on it to take the scaler out of the loop entirely.
func TestScaleGrayIdentityIsExact(t *testing.T) {
	src := loadGolden(t, "gray.bin")
	got := ScaleGray(src, 1.0)
	for y := range src.Height {
		for x := range src.Width {
			if got.At(x, y) != src.At(x, y) {
				t.Fatalf("ScaleGray(1.0) changed pixel (%d,%d): %d -> %d", x, y, src.At(x, y), got.At(x, y))
			}
		}
	}
}
