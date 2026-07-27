package imaging

import "testing"

func countForeground(b *Bitmap) int {
	var n int
	for y := range b.Height {
		for x := range b.Width {
			if b.At(x, y) != 0 {
				n++
			}
		}
	}
	return n
}

func TestSeedFillMatchesLeptonica(t *testing.T) {
	// The crop has no diagonal-only contacts, so its two goldens are
	// identical; the diagonal operand below is what separates 4 from 8.
	for _, conn := range []int{4, 8} {
		t.Run(map[int]string{4: "conn4", 8: "conn8"}[conn], func(t *testing.T) {
			mask := loadGolden(t, "seedfill_mask_in.bin")
			seed := loadGolden(t, "seedfill_seed_in.bin")
			want := loadGolden(t, map[int]string{4: "seedfill4.bin", 8: "seedfill8.bin"}[conn])

			got := SeedFill(seed, mask, conn)

			if diff := countDiff(t, got, want); diff != 0 {
				t.Errorf("SeedFill(conn=%d) differs from Leptonica in %d of %d pixels",
					conn, diff, got.Width*got.Height)
			}
		})
	}
}

// Guards the golden rather than the code: a fill that came out empty, or that
// swallowed the whole mask, would let a badly broken SeedFill pass the
// comparison above.
func TestSeedFillGoldenIsAProperSubsetOfTheMask(t *testing.T) {
	mask := loadGolden(t, "seedfill_mask_in.bin")
	want := loadGolden(t, "seedfill4.bin")

	filled, total := countForeground(want), countForeground(mask)
	if filled == 0 || filled >= total {
		t.Errorf("seedfill golden covers %d of %d mask pixels; want a proper non-empty subset",
			filled, total)
	}
}

// The diagonal operand is a staircase, two blocks touching only at a corner, a
// lone pixel, and a run along the top border. Seeded at the head of the
// staircase and inside the first block, 8-connectivity fills the staircase and
// both blocks while 4-connectivity fills a single pixel and one block — so this
// is the test that fails if SeedFill ignores its connectivity argument. The
// third seed pixel lies on background and must be dropped under both.
func TestSeedFillDiagonalMatchesLeptonica(t *testing.T) {
	for _, conn := range []int{4, 8} {
		t.Run(map[int]string{4: "conn4", 8: "conn8"}[conn], func(t *testing.T) {
			mask := loadGolden(t, "conncomp_diag_in.bin")
			seed := loadGolden(t, "seedfill_diag_seed_in.bin")
			want := loadGolden(t, map[int]string{
				4: "seedfill_diag4.bin", 8: "seedfill_diag8.bin",
			}[conn])

			got := SeedFill(seed, mask, conn)

			if diff := countDiff(t, got, want); diff != 0 {
				t.Errorf("SeedFill(diagonal, conn=%d) differs from Leptonica in %d of %d pixels",
					conn, diff, got.Width*got.Height)
			}
		})
	}
}

// One seed pixel selects its entire mask component; a seed pixel that falls on
// background selects nothing, so the result is always a subset of the mask.
func TestSeedFillSelectsWholeComponentsAndDropsSeedOffMask(t *testing.T) {
	mask := NewBitmap(30, 30, 1)
	fillRect(mask, 3, 3, 6, 6)   // seeded: comes back whole
	fillRect(mask, 20, 20, 4, 4) // unseeded: must not appear

	seed := NewBitmap(30, 30, 1)
	seed.Set(5, 5, 1)   // inside the first block
	seed.Set(15, 15, 1) // on background

	got := SeedFill(seed, mask, 8)

	for y := range 30 {
		for x := range 30 {
			want := uint8(0)
			if x >= 3 && x < 9 && y >= 3 && y < 9 {
				want = 1
			}
			if got.At(x, y) != want {
				t.Fatalf("SeedFill() at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

func TestSeedFillEmptySeedGivesEmptyResult(t *testing.T) {
	mask := NewBitmap(20, 20, 1)
	fillRect(mask, 2, 2, 10, 10)

	got := SeedFill(NewBitmap(20, 20, 1), mask, 4)

	if n := countForeground(got); n != 0 {
		t.Errorf("SeedFill(empty seed) set %d pixels; want 0", n)
	}
}

func TestSeedFillDoesNotModifyItsInputs(t *testing.T) {
	mask := NewBitmap(20, 20, 1)
	fillRect(mask, 2, 2, 10, 10)
	seed := NewBitmap(20, 20, 1)
	seed.Set(5, 5, 1)
	maskBefore, seedBefore := mask.Clone(), seed.Clone()

	SeedFill(seed, mask, 4)

	if diff := countDiff(t, mask, maskBefore); diff != 0 {
		t.Errorf("SeedFill changed %d mask pixels; want 0", diff)
	}
	if diff := countDiff(t, seed, seedBefore); diff != 0 {
		t.Errorf("SeedFill changed %d seed pixels; want 0", diff)
	}
}

func TestSeedFillRejectsBadInput(t *testing.T) {
	cases := map[string]func(){
		"connectivity": func() { SeedFill(NewBitmap(4, 4, 1), NewBitmap(4, 4, 1), 6) },
		"seed depth":   func() { SeedFill(NewBitmap(4, 4, 8), NewBitmap(4, 4, 1), 4) },
		"mask depth":   func() { SeedFill(NewBitmap(4, 4, 1), NewBitmap(4, 4, 8), 4) },
		"size":         func() { SeedFill(NewBitmap(4, 4, 1), NewBitmap(5, 4, 1), 4) },
	}
	for name, fn := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("SeedFill with a bad %s did not panic", name)
				}
			}()
			fn()
		})
	}
}

func TestDistanceFunctionMatchesLeptonica(t *testing.T) {
	in := loadGolden(t, "dist_in.bin")
	want := loadGolden(t, "dist4.bin")

	got := DistanceFunction(in)

	if got.Depth != 8 {
		t.Fatalf("DistanceFunction() depth = %d; want 8", got.Depth)
	}
	if diff := countDiff(t, got, want); diff != 0 {
		t.Errorf("DistanceFunction() differs from Leptonica in %d of %d pixels",
			diff, got.Width*got.Height)
	}
}

// Inside a rectangle the city-block distance to the nearest background pixel is
// one more than the distance to the nearest edge of the rectangle, which makes
// the whole result checkable by hand. Everything outside is 0.
func TestDistanceFunctionOnARectangle(t *testing.T) {
	const w, h = 24, 26
	b := NewBitmap(w, h, 1)
	fillRect(b, 4, 6, 10, 9)

	got := DistanceFunction(b)

	for y := range h {
		for x := range w {
			want := 0
			if x >= 4 && x < 14 && y >= 6 && y < 15 {
				want = 1 + min(min(x-4, 13-x), min(y-6, 14-y))
			}
			if int(got.At(x, y)) != want {
				t.Fatalf("DistanceFunction() at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

// Everything outside the image counts as background, so a foreground pixel on
// the image border is at distance 1 and a solid image is a pyramid. Treating
// the border as foreground instead would let the distances float upward there.
func TestDistanceFunctionTreatsOutsideAsBackground(t *testing.T) {
	const n = 21
	b := NewBitmap(n, n, 1)
	fillRect(b, 0, 0, n, n)

	got := DistanceFunction(b)

	for y := range n {
		for x := range n {
			want := 1 + min(min(x, n-1-x), min(y, n-1-y))
			if int(got.At(x, y)) != want {
				t.Fatalf("DistanceFunction() at (%d,%d) = %d; want %d", x, y, got.At(x, y), want)
			}
		}
	}
}

// The result is 8bpp, so distances above 255 saturate rather than wrap. A solid
// 600x600 block has a true centre distance of 300.
func TestDistanceFunctionSaturatesAtByteMax(t *testing.T) {
	const n = 600
	b := NewBitmap(n, n, 1)
	fillRect(b, 0, 0, n, n)

	got := DistanceFunction(b)

	if v := got.At(n/2-1, n/2-1); v != 255 {
		t.Errorf("DistanceFunction() at the centre = %d; want 255 (saturated)", v)
	}
	// Every pixel at least 254 from an edge has a true distance of 255 or more,
	// so the whole square [254,345] must be a flat plateau: nothing may wrap
	// back down.
	for y := 254; y <= 345; y++ {
		for x := 254; x <= 345; x++ {
			if got.At(x, y) != 255 {
				t.Fatalf("DistanceFunction() at (%d,%d) = %d; want 255", x, y, got.At(x, y))
			}
		}
	}
}

func TestDistanceFunctionRejectsBadInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("DistanceFunction(8bpp) did not panic")
		}
	}()
	DistanceFunction(NewBitmap(4, 4, 8))
}
