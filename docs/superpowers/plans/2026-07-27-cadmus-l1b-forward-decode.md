# Cadmus L1b — Forward Pass and Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A line image in, the correct text out, matching `tesseract --psm 13` on
a committed corpus of line crops — with per-word confidence and per-word bounding
boxes, because Kleio measures confidence spatially.

> **The oracle is `--psm 13`, not `--psm 7`, and that is a correction to the
> design spec.** `Tesseract::LSTMRecognizeWord` (`src/ccmain/linerec.cpp:229-250`)
> feeds the LSTM the *whole* image only for `PSM_SINGLE_WORD` and `PSM_RAW_LINE`:
>
> ```cpp
> if (tessedit_pageseg_mode == PSM_SINGLE_WORD || tessedit_pageseg_mode == PSM_RAW_LINE) {
>   word_box = TBOX(0, 0, ImageWidth(), ImageHeight());
> } else {
>   float baseline = row->base_line((word_box.left() + word_box.right()) / 2);
>   if (baseline + row->descenders() < word_box.bottom()) word_box.set_bottom(...);
>   if (baseline + row->x_height() + row->ascenders() > word_box.top()) word_box.set_top(...);
> }
> ImageData *im_data = GetRectImage(word_box, block, kImagePadding, &word_box);
> ```
>
> For `--psm 7` (`PSM_SINGLE_LINE`) the box is a per-word, baseline-derived
> sub-crop, padded by `kImagePadding = 4` and clipped to the image. Its height is
> essentially never the crop's height, and it is not even the full line
> horizontally. Cadmus's `Recognize(img)` normalizes the whole crop, so `--psm 7`
> is not an oracle for what Cadmus computes. `--psm 13` is: `word_box` becomes the
> full image, `GetRectImage` pads by 4 and then `*revised_box &= image_box` clips
> straight back, so Tesseract's LSTM sees exactly the committed PNG.
>
> The design spec (`docs/superpowers/specs/2026-07-27-cadmus-design.md` §L1) still
> says `--psm 7`. Update it when this plan lands; do not change the plan back.

**Architecture:** Three new packages over the `internal/tessdata` loader that L1a
completed. `internal/nn` is the loader-agnostic tensor runtime (StrideMap,
Tensor, Matrix, the activation tables, and one type per network layer).
`internal/dict` is the DAWG lexicon. `internal/recog` is the bridge: it builds an
`nn` graph from a parsed `tessdata` layer tree, normalizes a line image to the
36-pixel input height, runs the graph, and decodes the softmax output to text.

**The decode ships in two stages, and the order is not negotiable:**

1. **Stage 1 — greedy best-path CTC, no dictionary** (Tasks 17-20). Per-timestep
   argmax, drop the blank, collapse repeats. This is enough to prove the forward
   pass is correct, and it is the only thing that can prove it, because a wrong
   forward pass is undebuggable through a beam search. If the text is wrong at
   Stage 1, the bug is in the tensor runtime or the normalization, and Task 14's
   per-layer activation diff localizes it to a single layer.
2. **Stage 2 — dictionary-weighted beam search** (Tasks 21-24). Only started once
   Stage 1's character error rate against the oracle is at its floor.

**Tech Stack:** Go 1.26, standard library only. Test-only external tooling, all
of it used by golden-*generation* scripts and none of it required to run
`go test ./...`:

| tool | used by |
|---|---|
| `tesseract` 5.5.3 | Task 17 oracle capture |
| `text2image` (ships with tesseract) | Task 17 corpus rendering |
| `combine_tessdata` (ships with tesseract) | Tasks 15, 21 cross-checks |
| ImageMagick 7 (`magick`) | Task 17 `gen.sh` trim/depth/identify |
| `leptonica` 1.87.0 (headers + lib) | Tasks 12, 17 golden harness |
| a locally-built *instrumented* `tesseract` | Task 14 activation dump |

---

## Global Constraints

- **Go 1.26.** Module `github.com/dobbo-ca/cadmus`.
- **Zero third-party dependencies.** `go.sum` must stay empty for the life of
  this plan. CI already enforces this. If a task seems to need a dependency,
  stop and raise it.
- **No cgo.** Everything builds and tests with `CGO_ENABLED=0`.
- **FLOAT weights only, from `tessdata_best`.** int8 (`kInt8Flag`, `TF_INT_MODE`)
  is out of scope; L1a already asserts the model is float and fails loudly
  otherwise. Nothing in this plan may re-introduce an int8 code path.
- **Apache-2.0 attribution.** Every file that is a Go translation of Tesseract
  source carries the header described in `internal/doc.go`, naming the source
  file(s). Original work (the corpus harness, the grid, the CLI plumbing) carries
  no such header.
- **Oracles are test-only.** `tesseract`, `text2image` and Leptonica may be
  invoked by golden-*generation* tooling, never by tests. `go test ./...` must
  pass on a machine with none of them installed: every test that needs an
  external tool calls `t.Skipf`, never `t.Fatal`.
- **Certainties are computed in `float64` where Tesseract uses `float32`.**
  `NetworkIO::ProbToCertainty` is declared `static float ProbToCertainty(float)`
  (`src/lstm/networkio.h:187`), and every certainty, rating and score in
  `RecodeBeamSearch` is a `float`. Cadmus uses `float64` throughout
  `internal/recog`. This is a deliberate decision, not an oversight: the
  divergence is ~1e-7 per node, it compounds through the `min`/`sum` over a whole
  line and then through `100 + 35*c`, and it is worth roughly 0.00001 percentage
  points of reported confidence — far below the granularity Kleio's gate uses.
  If a confidence number ever disagrees with Tesseract's by more than 0.01, this
  is *not* the cause; look at the dict ratio and the span rule first.
- **Bit-exactness with Tesseract is NOT a goal; text equality is.** Tesseract's
  arm64 build dispatches its dot product to `DotProductNEON` and its activations
  come from 4096-entry interpolated lookup tables whose error is ~1.5e-6 — ten
  orders of magnitude above double eps. Cadmus reproduces the *tables* exactly
  (Task 4) because that error is systematic and shared; it does not attempt to
  reproduce summation order. Acceptance is final text plus a per-layer activation
  tolerance, not bitwise equality.
- **No fused multiply-add in Go.** The Go spec permits the compiler to contract
  `a*b + c` into a single FMA, across statements. Every product that feeds an
  accumulation in `internal/nn` must be wrapped in an explicit `float64(...)`
  conversion, which the spec requires the compiler to honour as a rounding point.
  Task 5 ships a regression test for this.
- **Single image, batch size 1.** Tesseract's `StrideMap` carries an `FD_BATCH`
  dimension so several differently-sized images can share one `NetworkIO`; that
  is why its code is full of per-batch valid widths, padding, and
  `ZeroInvalidElements`. Cadmus recognizes one line at a time. The batch
  dimension is elided throughout, every position in the map is valid, and there
  is no padding. This is stated once here and relied on by Tasks 2, 3, 6-11.
- **Multi-code recoders are out of scope.** For `eng` every `RecodedCharID` has
  `length == 1`, so network output index → recoder code → unichar id is a flat
  near-permutation, and the whole partial-code machinery in `RecodeBeamSearch`
  is a no-op. L1b implements the length-1 fast path and hard-errors on any model
  whose recoder contains a longer code (CJK, Indic). The restriction lives in
  `recog.NewRecognizer`, not in the loader — `internal/tessdata` loads multi-code
  recoders happily, and L1a's parser already does. Task 16 adds the guard and
  files the follow-up bead.

---

## Preconditions: L1a is a HARD blocking dependency, and it has not run yet

**L1a is a separate, unexecuted plan.** It lives at
`docs/superpowers/plans/2026-07-27-cadmus-l1a-model-loading.md`. As of this
writing `internal/tessdata` is still at the L0-spike state: it exports
`MatrixShape{Rows, Cols int; Int8 bool}` with the weight payload *skipped*
(`skip2DArray`), `ParseNetwork(data []byte, swap bool) (*Layer, error)` rather
than `ParseRecognizer`, a `parseRecognizerTrailer` that reads and discards
`null_char`/`sample_iteration`/`training_flags`, and no `unicharset.go`,
`recoder.go` or `dawg.go` at all. `testdata/fetch.sh` still prefers the Homebrew
copy, so the committed `testdata/eng.traineddata` is the *legacy int8* model.

**Nothing in this plan can be started before L1a is complete, and this plan must
not re-do any of L1a's work.** Every item below is L1a's deliverable, not L1b's.

- [ ] **Precondition gate — run this first; if any line fails, STOP and execute
  L1a to completion**

```bash
cd /Users/christopherdobbyn/work/dobbo-ca/cadmus
go doc ./internal/tessdata | grep -E 'func LoadModel|func ParseRecognizer|func ParseUnicharset|func ParseRecoder|func ParseDawg|type Matrix|type Model'
ls -l testdata/eng.traineddata          # want 15400601 bytes
go run ./cmd/cadmusdump testdata/eng.traineddata | head -20
```

Expected: the dump shows `Lfys64`, `Lfx96`, `Lrx96`, `Lfx512`, `Softmax` with
`no=111`, and `Series` with `weights=1461007` — the `tessdata_best` 4.1.0
topology.

**The exact failure signature that means "L1a has not run":** the dump prints
`Lfys48 / Lfx96 / Lrx96 / Lfx192`, `weights=385807`, and every matrix tagged
`(int8)`. That is the Homebrew model, which this plan's Global Constraints
explicitly reject. Do not attempt to work around it here — re-baselining
`fetch.sh` and loading float weight *values* is L1a Tasks 1 and 3, roughly 40% of
the loader, and duplicating it in L1b would produce two conflicting
`internal/tessdata` implementations.

### The exact L1a API this plan consumes

Transcribed from the L1a plan's Interfaces blocks. **Use these spellings
verbatim.** Where a code block later in this plan uses a different spelling
(this plan was drafted before L1a's names were fixed), the table below wins and
the mismatch is a compile error, not a silent bug.

```go
// internal/tessdata — L1a Task 3
type Matrix struct {
	Rows, Cols int
	Values     []float64        // NOT "W"; row-major, last column of each row is the bias
}
func (m *Matrix) At(row, col int) float64
func (m *Matrix) Stats() (min, max, mean float64)

type InputShape struct{ Batch, Height, Width, Depth, LossType int }

type Layer struct {
	Type       LayerType
	Name       string
	NumInputs  int
	NumOutputs int
	NumWeights int
	Flags      int32
	Matrices   []Matrix
	Children   []*Layer

	HalfX, HalfY int         // Convolve
	XScale       int         // Maxpool, Reconfig
	YScale       int         // Maxpool, Reconfig
	Shape        *InputShape // Input only
	NA           int         // LSTM family
}

// internal/tessdata — L1a Task 4
type Recognizer struct {
	Root              *Layer
	NetworkStr        string
	TrainingFlags     int32
	TrainingIteration int32
	SampleIteration   int32
	NullChar          int32   // NOT int — convert at the call site
}
func ParseRecognizer(data []byte, swap bool) (*Recognizer, error)

// internal/tessdata — L1a Task 5. Accessor methods, NOT an exported Chars slice.
type Unichar struct {
	Text       string
	Normed     string
	Script     string
	Properties uint32          // NOT int
}
func ParseUnicharset(data []byte) (*Unicharset, error)
func (u *Unicharset) Size() int
func (u *Unicharset) Char(id int) (Unichar, bool)
func (u *Unicharset) Text(id int) string
func (u *Unicharset) Normed(id int) string
func (u *Unicharset) IsSpaceDelimited(id int) bool
const (UnicharSpace = 0; UnicharJoined = 1; UnicharBroken = 2)
const (PropAlpha uint32 = 0x1; PropLower = 0x2; PropUpper = 0x4; PropDigit = 0x8; PropPunct = 0x10)

// internal/tessdata — L1a Task 6. Codes are int32 and Encode is a METHOD.
func ParseRecoder(data []byte, swap bool) (*Recoder, error)
func (rc *Recoder) Size() int
func (rc *Recoder) CodeRange() int
func (rc *Recoder) MaxCodeLen() int
func (rc *Recoder) Encode(unicharID int) []int32
func (rc *Recoder) DecodeUnichar(code []int32) (int, bool)
func (rc *Recoder) IsValidFirstCode(code int32) bool

// internal/tessdata — L1a Task 7. NO DawgType, NO edge accessors; see L1b Task 21.
type Dawg struct{ UnicharsetSize int /* unexported edges, masks, bit offsets */ }
func ParseDawg(data []byte, swap bool) (*Dawg, error)
func (d *Dawg) NumEdges() int
func (d *Dawg) Contains(ids []int) bool   // word-dawg semantics ONLY
func (d *Dawg) HasPrefix(ids []int) bool  // word-dawg semantics ONLY

// internal/tessdata — L1a Task 8
type Model struct {
	Version    string
	Swapped    bool
	Recognizer *Recognizer
	Unicharset *Unicharset
	Recoder    *Recoder
	PuncDawg   *Dawg // nil when the component is absent
	WordDawg   *Dawg
	NumberDawg *Dawg
}
func LoadModel(data []byte) (*Model, error)
func (m *Model) Softmax() *Layer
func (m *Model) NumOutputs() int
func (m *Model) NullChar() int
```

L1a also supplies the fixture-loading test helper `loadRealRecognizer(t)` (with a
`loadRealNetwork` shim) in `internal/tessdata`. **Do not declare a second one.**

### Tasks of this plan that L1a supersedes

| L1b task | Status after L1a | What is left to do |
|---|---|---|
| **Task 1** — retain layer geometry | **Superseded.** L1a Task 3 adds `HalfX`, `HalfY`, `XScale`, `YScale`, `Shape`, `NA` to `Layer` with these exact names, and says so explicitly. | Verification only. Do not touch `network.go`. |
| **Task 15** — unicharset parser | **Superseded.** L1a Task 5 creates `internal/tessdata/unicharset.go`. | Verify L1a's accessors cover what `internal/recog` needs; adapt call sites. |
| **Task 16** — recoder parser | **Superseded.** L1a Task 6 creates `internal/tessdata/recoder.go`, and it supports multi-code entries rather than rejecting them. | Move the single-code restriction into `recog.NewRecognizer`, where it belongs. |
| **Task 21** — DAWG reader | **Partly superseded.** L1a Task 7 creates `internal/tessdata/dawg.go` with the wire format and `Contains`/`HasPrefix`. It deliberately omits `DawgType` and the edge-level accessors, and names L1b as their owner. | Extend the existing file with `EdgeChar`/`NextNode`/`EndOfWord`/`UnicharID`; put `DawgType` in `internal/dict`. |
| Tasks 2-14, 17-20, 22-24 | Untouched by L1a. | As written. |

Record the reconciliation once, so the next reader does not re-derive it:

```bash
bd update cad-l1b --append-notes "L1a landed first: Tasks 1/15/16 reduced to verification, Task 21 reduced to extending internal/tessdata/dawg.go."
```

---

## File Structure

| File | Responsibility |
|---|---|
| `internal/tessdata/network.go` | **L1a Task 3** — Task 1 verifies only, changes nothing |
| `internal/tessdata/unicharset.go` | **L1a Task 5** — Task 15 verifies only, creates nothing |
| `internal/tessdata/recoder.go` | **L1a Task 6** — Task 16 verifies only, creates nothing |
| `internal/tessdata/dawg.go` | **L1a Task 7**, *extended* by Task 21 with edge-level accessors |
| `internal/nn/stridemap.go` | Task 2: (y,x) ↔ timestep, scaling, transposition |
| `internal/nn/tensor.go` | Task 3: float32 activation buffer, Tesseract's `NetworkIO` |
| `internal/nn/tables.go` | Task 4: generated — Tanh and Logistic lookup tables |
| `internal/nn/gen/gen.go` | Task 4: `//go:build ignore` table transcriber |
| `internal/nn/funcs.go` | Task 4: `Tanh`, `Logistic`, `SoftmaxInPlace` |
| `internal/nn/matrix.go` | Task 5: weight matrix and `DotVector` |
| `internal/nn/layer.go` | Task 6: the `Layer` interface, `Input`, `Series`, `Parallel`, `Reversed` |
| `internal/nn/rand.go` | Task 7: `TRand`, Tesseract's LCG |
| `internal/nn/convolve.go` | Task 7: 3x3 im2col gather with randomized edge padding |
| `internal/nn/reconfig.go` | Task 8: `Maxpool` and `Reconfig` |
| `internal/nn/fullyconnected.go` | Task 9: `Tanh`/`Softmax`/`Relu`/`Linear`/`Logistic` layers |
| `internal/nn/lstm.go` | Task 10: the LSTM cell, plain and summarizing |
| `internal/recog/build.go` | Task 11: `tessdata.Layer` tree → `nn` graph |
| `internal/imaging/scalegray.go` | Task 12: Leptonica-exact 8bpp scaling |
| `internal/recog/normalize.go` | Task 13: line image → input `Tensor` |
| `cmd/cadmusdump/activations.go` | Task 14: `-activations` dump for the differential diff |
| `testdata/tessdebug/` | Task 14: the instrumented-Tesseract patch and capture script |
| `testdata/golden/gen/scaleline.c` | Task 17: exact-height PNG rescaler for the `h36` arm |
| `testdata/lines/` | Task 17: the line-crop corpus, ground truth, and oracle output |
| `internal/recog/decode.go` | Task 18: greedy CTC decode |
| `internal/recog/certainty.go` | Task 19: per-character certainty, rating, boundaries |
| `internal/recog/words.go` | Task 20: word segmentation, confidence, boxes, public types |
| `internal/dict/dict.go` | Task 22: `DawgType`, the `DawgPosition` machinery, `LetterIsOkay` |
| `internal/recog/beam.go` | Task 23: `RecodeBeamSearch` |

---

## Task 1: Verify the layer geometry L1a retained

**This task creates and changes nothing.** L1a Task 3 already adds `HalfX`,
`HalfY`, `XScale`, `YScale`, `Shape *InputShape` and `NA` to `tessdata.Layer`,
using exactly these names, and its own text says so: *"These names are chosen to
match L1b, and they supersede L1b Task 1 … Do not re-implement it."* Re-adding
the fields would not compile; re-adding the test helper would collide with
L1a's `loadRealRecognizer`.

What remains is a five-minute confirmation that the values the runtime depends on
are actually what this plan's later tasks hard-code.

**Files:** none.

- [ ] **Step 1: Confirm the fields exist and carry the right values**

```bash
cd /Users/christopherdobbyn/work/dobbo-ca/cadmus
go test ./internal/tessdata/ -run 'TestParseNetworkRealModelWeights|TestParseNetworkRetainsLayerGeometry' -v
go run ./cmd/cadmusdump testdata/eng.traineddata
```

Expected, cross-checked against the model's own spec string
`[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`:

| layer | field | value |
|---|---|---|
| `Convolve` | `HalfX`, `HalfY` | 1, 1 (`Ct3,3`) |
| `Maxpool` | `XScale`, `YScale` | 3, 3 (`Mp3,3`) |
| `Input` | `Shape` | `{Batch:1 Height:36 Width:0 Depth:1 LossType:0}` |
| `Lfys64` | `NA`, gate shape | 80, 64x81, 4 matrices |
| `Lfx96` | `NA`, gate shape | 160, 96x161, 4 matrices |
| `Lrx96` | `NA`, gate shape | 192, 96x193, 4 matrices |
| `Lfx512` | `NA`, gate shape | 608, 512x609, 4 matrices |

**If L1a's tests do not already assert every row of that table**, add the missing
assertions to L1a's `internal/tessdata/network_test.go` — do not start a second
test file, and do not re-declare a fixture helper.

- [ ] **Step 2: Confirm `XScale == 3` specifically**

The bounding-box arithmetic in Task 20 and the `xScaleFactor` walk in Task 11
both depend on the network's x reduction factor being 3, read out of the
serialized layer rather than inferred from the spec string.

```bash
go run ./cmd/cadmusdump testdata/eng.traineddata | grep -i maxpool
```

**If it is not 3**, every `scale_factor` constant in Task 20 is wrong. Recompute
`XScaleFactor` with the corrected `xScaleFactor` from Task 11 Step 3 — which is
`stack[0]` for plumbing and the *product* only for `Series`, matching
`Plumbing::XScaleFactor` and `Series::XScaleFactor` — and note the discrepancy in
the bead before continuing.

- [ ] **Step 3: Nothing to commit**

This task produces no diff. If `git status` shows changes under
`internal/tessdata/`, they are L1a's and belong to L1a's commits.

---

## Task 2: StrideMap

**Files:**
- Create: `internal/nn/stridemap.go`
- Test: `internal/nn/stridemap_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces: `StrideMap` with `Len`, `T`, `YX`, `Offset`, `ScaleXY`,
  `TransposeXY`, `ReduceWidthTo1`.

- [ ] **Step 1: Write the failing test**

Create `internal/nn/stridemap_test.go`:

```go
package nn

import "testing"

func TestStrideMapRasterOrder(t *testing.T) {
	m := StrideMap{Height: 3, Width: 4}
	if m.Len() != 12 {
		t.Fatalf("Len() = %d; want 12", m.Len())
	}
	// x is the inner dimension: Tesseract's Index::Increment carries from
	// FD_DIMSIZE-1 (width) down to 0.
	if got := m.T(0, 1); got != 1 {
		t.Errorf("T(0,1) = %d; want 1", got)
	}
	if got := m.T(1, 0); got != 4 {
		t.Errorf("T(1,0) = %d; want 4", got)
	}
	for tt := range m.Len() {
		y, x := m.YX(tt)
		if m.T(y, x) != tt {
			t.Errorf("YX/T round trip failed at t=%d: got (%d,%d)", tt, y, x)
		}
	}
}

func TestStrideMapOffset(t *testing.T) {
	m := StrideMap{Height: 3, Width: 4}
	if got, ok := m.Offset(m.T(1, 1), -1, -1); !ok || got != m.T(0, 0) {
		t.Errorf("Offset(T(1,1),-1,-1) = %d,%v; want %d,true", got, ok, m.T(0, 0))
	}
	if _, ok := m.Offset(m.T(0, 0), -1, 0); ok {
		t.Error("Offset above the top row reported in-bounds")
	}
	if _, ok := m.Offset(m.T(0, 3), 0, 1); ok {
		t.Error("Offset past the right edge reported in-bounds")
	}
	if _, ok := m.Offset(m.T(2, 0), 1, 0); ok {
		t.Error("Offset below the bottom row reported in-bounds")
	}
}

// ScaleXY floor-divides, so a partial trailing pooling window is dropped, not
// padded. 36 rows and 100 columns at Mp3,3 give 12 x 33, not 12 x 34.
func TestStrideMapScaleXYTruncates(t *testing.T) {
	got := StrideMap{Height: 36, Width: 100}.ScaleXY(3, 3)
	if got != (StrideMap{Height: 12, Width: 33}) {
		t.Errorf("ScaleXY(3,3) = %+v; want {12 33}", got)
	}
}

func TestStrideMapTransposeAndReduce(t *testing.T) {
	m := StrideMap{Height: 12, Width: 33}
	if got := m.TransposeXY(); got != (StrideMap{Height: 33, Width: 12}) {
		t.Errorf("TransposeXY() = %+v; want {33 12}", got)
	}
	if got := m.TransposeXY().TransposeXY(); got != m {
		t.Errorf("TransposeXY is not an involution: %+v", got)
	}
	if got := m.ReduceWidthTo1(); got != (StrideMap{Height: 12, Width: 1}) {
		t.Errorf("ReduceWidthTo1() = %+v; want {12 1}", got)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -v`
Expected: FAIL — `undefined: StrideMap`.

- [ ] **Step 3: Implement**

Create `internal/nn/stridemap.go`:

```go
// This file is a Go translation of src/lstm/stridemap.cpp and
// src/lstm/stridemap.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// StrideMap maps a 2-D (y, x) position in a feature map to the flat timestep
// index used by Tensor, and back.
//
// Tesseract's StrideMap carries a batch dimension (FD_BATCH) so that several
// differently-sized images can share one NetworkIO, which is why its code is
// full of per-batch valid widths, padding, and ZeroInvalidElements. Cadmus
// recognizes exactly one line image at a time, so the batch dimension is always
// 1 and is elided here: there is one (Height, Width) pair, every position is
// valid, and there is nothing to zero. Raster order is preserved — y is the
// outer dimension and x the inner, exactly as StrideMap::Index::Increment
// carries from FD_DIMSIZE-1 down to 0.
type StrideMap struct {
	Height, Width int
}

// Len is the number of timesteps in the map.
func (s StrideMap) Len() int { return s.Height * s.Width }

// T returns the timestep index of (y, x).
func (s StrideMap) T(y, x int) int { return y*s.Width + x }

// YX inverts T.
func (s StrideMap) YX(t int) (y, x int) { return t / s.Width, t % s.Width }

// Offset returns the timestep dy rows and dx columns from t, and reports
// whether that position is inside the map. It is StrideMap::Index::AddOffset
// applied to FD_HEIGHT and FD_WIDTH in turn.
func (s StrideMap) Offset(t, dy, dx int) (int, bool) {
	y, x := s.YX(t)
	y += dy
	x += dx
	if y < 0 || y >= s.Height || x < 0 || x >= s.Width {
		return 0, false
	}
	return s.T(y, x), true
}

// ScaleXY is StrideMap::ScaleXY. Both dimensions are integer-divided, so a
// partial trailing pooling window is dropped rather than padded.
func (s StrideMap) ScaleXY(xFactor, yFactor int) StrideMap {
	return StrideMap{Height: s.Height / yFactor, Width: s.Width / xFactor}
}

// TransposeXY is StrideMap::TransposeXY.
func (s StrideMap) TransposeXY() StrideMap {
	return StrideMap{Height: s.Width, Width: s.Height}
}

// ReduceWidthTo1 is StrideMap::ReduceWidthTo1, used by NT_LSTM_SUMMARY.
func (s StrideMap) ReduceWidthTo1() StrideMap {
	return StrideMap{Height: s.Height, Width: 1}
}

func (s StrideMap) String() string { return fmt.Sprintf("%dx%d", s.Height, s.Width) }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/stridemap.go internal/nn/stridemap_test.go
git commit -m "feat(nn): add the StrideMap timestep index"
```

---

## Task 3: Tensor, the activation buffer

Tesseract's `NetworkIO` stores activations as **float32** (`GENERIC_2D_ARRAY<float> f_`)
while every arithmetic register — weights, dot products, gate values, cell state
— is **float64**. Every `WriteTimeStep` narrows and every `ReadTimeStep` widens.
This matters because the LSTM's own recurrent output round-trips through a
`NetworkIO` (`source_`), so the rounding is *inside* the recurrence. Storing
float64 here is a silent ~1e-7 divergence per timestep that compounds through
four stacked LSTMs.

**Files:**
- Create: `internal/nn/tensor.go`
- Test: `internal/nn/tensor_test.go`

**Interfaces:**
- Consumes: `StrideMap`.
- Produces:

```go
type Tensor struct {
	Map      StrideMap
	Features int
	// unexported float32 backing store
}

func NewTensor(m StrideMap, features int) *Tensor
func (x *Tensor) Row(t int) []float32
func (x *Tensor) ReadTimeStep(t int, dst []float64)
func (x *Tensor) WriteTimeStep(t int, src []float64)
func (x *Tensor) WriteTimeStepPart(t, offset, n int, src []float64)
func (x *Tensor) CopyTimeStep(t int, src *Tensor, srcT int)
func (x *Tensor) CopyTimeStepPart(t, offset, n int, src *Tensor, srcT, srcOffset int)
func (x *Tensor) MaxpoolTimeStep(t int, src *Tensor, srcT int)
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/tensor_test.go`:

```go
package nn

import (
	"math"
	"testing"
)

func TestTensorRoundsThroughFloat32(t *testing.T) {
	x := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	// A value that is representable in float64 but not float32. Storing and
	// reloading must lose the low bits, exactly as NetworkIO does.
	const v = 1.0 + 1e-12
	x.WriteTimeStep(0, []float64{v})
	got := make([]float64, 1)
	x.ReadTimeStep(0, got)
	if got[0] == v {
		t.Fatalf("ReadTimeStep returned %v unchanged; the store is not float32", got[0])
	}
	if got[0] != float64(float32(v)) {
		t.Fatalf("ReadTimeStep() = %v; want %v", got[0], float64(float32(v)))
	}
}

func TestTensorWriteTimeStepPart(t *testing.T) {
	x := NewTensor(StrideMap{Height: 1, Width: 2}, 5)
	x.WriteTimeStep(0, []float64{1, 2, 3, 4, 5})
	x.WriteTimeStepPart(0, 2, 2, []float64{9, 8})
	got := make([]float64, 5)
	x.ReadTimeStep(0, got)
	want := []float64{1, 2, 9, 8, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("after WriteTimeStepPart = %v; want %v", got, want)
		}
	}
	// Timestep 1 must be untouched.
	x.ReadTimeStep(1, got)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("timestep 1 feature %d = %v; want 0", i, v)
		}
	}
}

func TestTensorCopyTimeStepPart(t *testing.T) {
	src := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	src.WriteTimeStep(0, []float64{7, 8, 9})
	dst := NewTensor(StrideMap{Height: 1, Width: 1}, 6)
	dst.CopyTimeStepPart(0, 3, 3, src, 0, 0)
	got := make([]float64, 6)
	dst.ReadTimeStep(0, got)
	want := []float64{0, 0, 0, 7, 8, 9}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("CopyTimeStepPart = %v; want %v", got, want)
		}
	}
}

func TestTensorMaxpoolTimeStep(t *testing.T) {
	a := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	a.WriteTimeStep(0, []float64{1, 5, -3})
	b := NewTensor(StrideMap{Height: 1, Width: 1}, 3)
	b.WriteTimeStep(0, []float64{4, 2, -1})
	a.MaxpoolTimeStep(0, b, 0)
	got := make([]float64, 3)
	a.ReadTimeStep(0, got)
	want := []float64{4, 5, -1}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 0 {
			t.Fatalf("MaxpoolTimeStep = %v; want %v", got, want)
		}
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run TestTensor -v`
Expected: FAIL — `undefined: NewTensor`.

- [ ] **Step 3: Implement**

Create `internal/nn/tensor.go`:

```go
// This file is a Go translation of src/lstm/networkio.cpp and
// src/lstm/networkio.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

// Tensor is Tesseract's NetworkIO on the float path: a Height x Width grid of
// Features-long activation vectors addressed by flat timestep.
//
// The backing store is float32 even though every arithmetic register in the
// network is float64. That is deliberate and load-bearing: NetworkIO stores
// GENERIC_2D_ARRAY<float>, WriteTimeStep narrows with static_cast<float>, and
// ReadTimeStep widens back. The LSTM's recurrent output passes through a
// NetworkIO (source_) on every timestep, so the narrowing sits inside the
// recurrence. Widening this to float64 diverges from Tesseract by ~1e-7 per
// timestep and compounds through four stacked LSTMs.
type Tensor struct {
	Map      StrideMap
	Features int
	data     []float32
}

func NewTensor(m StrideMap, features int) *Tensor {
	return &Tensor{Map: m, Features: features, data: make([]float32, m.Len()*features)}
}

// Row returns the activation vector at timestep t, aliasing the backing store.
func (x *Tensor) Row(t int) []float32 {
	return x.data[t*x.Features : (t+1)*x.Features]
}

// ReadTimeStep widens timestep t into dst, which must have length Features.
func (x *Tensor) ReadTimeStep(t int, dst []float64) {
	for i, v := range x.Row(t) {
		dst[i] = float64(v)
	}
}

// WriteTimeStep narrows the first Features values of src into timestep t.
func (x *Tensor) WriteTimeStep(t int, src []float64) {
	row := x.Row(t)
	for i := range row {
		row[i] = float32(src[i])
	}
}

// WriteTimeStepPart narrows the first n values of src into timestep t starting
// at feature index offset.
func (x *Tensor) WriteTimeStepPart(t, offset, n int, src []float64) {
	row := x.Row(t)[offset : offset+n]
	for i := range row {
		row[i] = float32(src[i])
	}
}

// CopyTimeStep copies a whole timestep from src. Feature counts must match.
func (x *Tensor) CopyTimeStep(t int, src *Tensor, srcT int) {
	copy(x.Row(t), src.Row(srcT))
}

// CopyTimeStepPart is NetworkIO::CopyTimeStepGeneral: it copies n features from
// src's timestep srcT starting at srcOffset into timestep t starting at offset.
func (x *Tensor) CopyTimeStepPart(t, offset, n int, src *Tensor, srcT, srcOffset int) {
	copy(x.Row(t)[offset:offset+n], src.Row(srcT)[srcOffset:srcOffset+n])
}

// MaxpoolTimeStep takes the elementwise maximum of timestep t and src's
// timestep srcT, in place. Tesseract also records which source timestep won,
// for the backward pass; L1b is inference only, so that is omitted.
func (x *Tensor) MaxpoolTimeStep(t int, src *Tensor, srcT int) {
	dst := x.Row(t)
	for i, v := range src.Row(srcT) {
		if dst[i] < v {
			dst[i] = v
		}
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -run TestTensor -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/tensor.go internal/nn/tensor_test.go
git commit -m "feat(nn): add the float32 activation tensor"
```

---

## Task 4: Activation lookup tables and softmax

Tesseract does **not** call `tanh` or `exp` for its activations. It interpolates
4096-entry lookup tables at 1/256 spacing (`src/lstm/functions.h:34-72`), which
carries a piecewise-linear error of roughly `(1/256)²/8 · max|f''|` — about
**1.5e-6 for tanh and 1.8e-7 for logistic**, ten orders of magnitude above
double eps. Using Go's `math.Tanh` instead diverges from Tesseract at the sixth
decimal on the *first* activation, and the error compounds through the
recurrence and four stacked LSTMs.

The tables are generated by `src/lstm/generate_lut.py` with Python's `"%a"`
format, which is the shortest round-tripping decimal, so the shipped constants
recover losslessly through `strconv.ParseFloat`. **Transcribe them; do not
regenerate them.**

Softmax is the exception — it is *not* tabulated and calls `std::exp` directly.

**Files:**
- Create: `internal/nn/gen/gen.go`, `internal/nn/tables.go` (generated),
  `internal/nn/funcs.go`
- Test: `internal/nn/funcs_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces: `Tanh(x float64) float64`, `Logistic(x float64) float64`,
  `TanhInPlace([]float64)`, `LogisticInPlace([]float64)`,
  `SoftmaxInPlace([]float64)`; package constants `tableSize = 4096`,
  `tableScale = 256.0`.

- [ ] **Step 1: Write the transcriber**

Create `internal/nn/gen/gen.go`:

```go
//go:build ignore

// Command gen transcribes Tesseract's Tanh and Logistic activation lookup
// tables out of src/lstm/functions.cpp into Go source.
//
// The tables are the shortest round-tripping decimal representation of the
// exact IEEE-754 doubles Tesseract uses, so ParseFloat/FormatFloat round-trips
// them losslessly. They are transcribed rather than regenerated because Go's
// math.Tanh and CPython's math.tanh are not guaranteed bit-identical, and the
// table values are what Tesseract's arithmetic actually sees.
//
// Usage — note the explicit .go path. `go run ./internal/nn/gen` fails with
// "build constraints exclude all Go files" because of the //go:build ignore
// above; naming the file bypasses package selection:
//
//	go run ./internal/nn/gen/gen.go -src /path/to/tesseract/src/lstm/functions.cpp \
//	    -out internal/nn/tables.go
package main

import (
	"bytes"
	"crypto/sha256"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
	"strconv"
	"strings"
)

const tableSize = 4096

func main() {
	src := flag.String("src", "", "path to tesseract src/lstm/functions.cpp")
	out := flag.String("out", "internal/nn/tables.go", "output Go file")
	flag.Parse()
	if *src == "" {
		log.Fatal("gen: -src is required")
	}

	raw, err := os.ReadFile(*src)
	if err != nil {
		log.Fatalf("gen: %v", err)
	}
	sum := sha256.Sum256(raw)

	tanh, err := extract(string(raw), "TanhTable")
	if err != nil {
		log.Fatalf("gen: %v", err)
	}
	logistic, err := extract(string(raw), "LogisticTable")
	if err != nil {
		log.Fatalf("gen: %v", err)
	}

	var b bytes.Buffer
	fmt.Fprintf(&b, `// Code generated by internal/nn/gen from Tesseract's
// src/lstm/functions.cpp. DO NOT EDIT.
//
// Source file SHA-256: %x
//
// These tables are part of Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0.

package nn

`, sum)
	writeTable(&b, "tanhTable", tanh)
	writeTable(&b, "logisticTable", logistic)

	formatted, err := format.Source(b.Bytes())
	if err != nil {
		log.Fatalf("gen: formatting output: %v", err)
	}
	if err := os.WriteFile(*out, formatted, 0o644); err != nil {
		log.Fatalf("gen: %v", err)
	}
	fmt.Printf("wrote %s (%d + %d entries, source sha256 %x)\n", *out, len(tanh), len(logistic), sum)
}

// extract pulls the float literals of `const TFloat <name>[] = { ... };`.
func extract(src, name string) ([]float64, error) {
	marker := "const TFloat " + name + "[] = {"
	i := strings.Index(src, marker)
	if i < 0 {
		return nil, fmt.Errorf("%s not found", name)
	}
	rest := src[i+len(marker):]
	j := strings.Index(rest, "};")
	if j < 0 {
		return nil, fmt.Errorf("%s is not terminated", name)
	}
	var out []float64
	for _, line := range strings.Split(rest[:j], "\n") {
		line = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(line), ","))
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("%s: parsing %q: %w", name, line, err)
		}
		out = append(out, v)
	}
	if len(out) != tableSize {
		return nil, fmt.Errorf("%s has %d entries; want %d", name, len(out), tableSize)
	}
	return out, nil
}

func writeTable(b *bytes.Buffer, name string, vals []float64) {
	fmt.Fprintf(b, "var %s = [%d]float64{\n", name, len(vals))
	for _, v := range vals {
		fmt.Fprintf(b, "\t%s,\n", strconv.FormatFloat(v, 'g', -1, 64))
	}
	fmt.Fprint(b, "}\n\n")
}
```

- [ ] **Step 2: Generate the tables**

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
# If the clone is gone: git clone --depth 1 https://github.com/tesseract-ocr/tesseract "$SP/tess"
# The .go suffix is required: gen.go is //go:build ignore, so `go run ./internal/nn/gen`
# exits with "build constraints exclude all Go files in …".
go run ./internal/nn/gen/gen.go -src "$SP/tess/src/lstm/functions.cpp" -out internal/nn/tables.go
head -12 internal/nn/tables.go
grep -c ',$' internal/nn/tables.go   # expect 8192
```

Expected: `wrote internal/nn/tables.go (4096 + 4096 entries, ...)`. If either
table does not have exactly 4096 entries the generator exits non-zero — do not
work around it, the file format changed and the parser must be fixed.

Also add a `Makefile` target so the provenance is not folklore:

```make
# Regenerates the activation tables from a Tesseract checkout. Manual step;
# commit the result. TESS=/path/to/tesseract make tables
# gen.go carries //go:build ignore, so the recipe names the FILE, not the dir.
tables:
	go run ./internal/nn/gen/gen.go -src $(TESS)/src/lstm/functions.cpp -out internal/nn/tables.go
```

- [ ] **Step 3: Write the failing test**

Create `internal/nn/funcs_test.go`:

```go
package nn

import (
	"math"
	"testing"
)

func TestTablesAreComplete(t *testing.T) {
	if len(tanhTable) != tableSize || len(logisticTable) != tableSize {
		t.Fatalf("table sizes = %d, %d; want %d each", len(tanhTable), len(logisticTable), tableSize)
	}
	if tanhTable[0] != 0 {
		t.Errorf("tanhTable[0] = %v; want 0", tanhTable[0])
	}
	if logisticTable[0] != 0.5 {
		t.Errorf("logisticTable[0] = %v; want 0.5", logisticTable[0])
	}
	for i := 1; i < tableSize; i++ {
		if tanhTable[i] <= tanhTable[i-1] {
			t.Fatalf("tanhTable is not strictly increasing at %d", i)
		}
		if logisticTable[i] <= logisticTable[i-1] {
			t.Fatalf("logisticTable is not strictly increasing at %d", i)
		}
	}
}

// Exactly on a table stop, the interpolation weight is zero, so the result must
// be the table entry itself.
func TestTanhHitsTableStopsExactly(t *testing.T) {
	for _, i := range []int{1, 7, 255, 1000, 4094} {
		x := float64(i) / tableScale
		if got := Tanh(x); got != tanhTable[i] {
			t.Errorf("Tanh(%v) = %v; want tanhTable[%d] = %v", x, got, i, tanhTable[i])
		}
		if got := Logistic(x); got != logisticTable[i] {
			t.Errorf("Logistic(%v) = %v; want logisticTable[%d] = %v", x, got, i, logisticTable[i])
		}
	}
}

func TestActivationSymmetryAndSaturation(t *testing.T) {
	if Tanh(0) != 0 {
		t.Errorf("Tanh(0) = %v; want 0", Tanh(0))
	}
	if Logistic(0) != 0.5 {
		t.Errorf("Logistic(0) = %v; want 0.5", Logistic(0))
	}
	// index >= tableSize-1 returns exactly 1; the negative branch mirrors.
	const sat = 4095.0 / tableScale
	if Tanh(sat) != 1 {
		t.Errorf("Tanh(%v) = %v; want exactly 1", sat, Tanh(sat))
	}
	if Tanh(-sat) != -1 {
		t.Errorf("Tanh(%v) = %v; want exactly -1", -sat, Tanh(-sat))
	}
	if Logistic(sat) != 1 {
		t.Errorf("Logistic(%v) = %v; want exactly 1", sat, Logistic(sat))
	}
	if Logistic(-sat) != 0 {
		t.Errorf("Logistic(%v) = %v; want exactly 0", -sat, Logistic(-sat))
	}
	if got := Logistic(-1.5); got != 1-Logistic(1.5) {
		t.Errorf("Logistic(-1.5) = %v; want 1-Logistic(1.5) = %v", got, 1-Logistic(1.5))
	}
}

// The whole point of transcribing the tables: our Tanh is measurably NOT
// math.Tanh. If this test starts passing with a tiny bound, someone
// regenerated the tables with Go's libm and the port silently diverged from
// Tesseract's arithmetic.
func TestTanhIsTheInterpolatedTableNotLibm(t *testing.T) {
	var maxDiff float64
	for i := 0; i < 200000; i++ {
		x := float64(i) * 8.0 / 200000
		d := math.Abs(Tanh(x) - math.Tanh(x))
		if d > maxDiff {
			maxDiff = d
		}
	}
	// Piecewise-linear error for tanh at h=1/256 is (h^2/8)*max|f''| ~ 1.5e-6.
	if maxDiff < 1e-7 {
		t.Fatalf("max |Tanh - math.Tanh| = %g; too small — the tables look like they were regenerated with Go's libm instead of transcribed from functions.cpp", maxDiff)
	}
	if maxDiff > 2e-6 {
		t.Fatalf("max |Tanh - math.Tanh| = %g; larger than the 1.5e-6 interpolation bound — the table or the interpolation is wrong", maxDiff)
	}
	t.Logf("max |Tanh - math.Tanh| = %g (expected ~1.5e-6)", maxDiff)
}

func TestSoftmaxInPlace(t *testing.T) {
	v := []float64{1, 2, 3}
	SoftmaxInPlace(v)
	var sum float64
	for _, p := range v {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("softmax sums to %v; want 1", sum)
	}
	if !(v[0] < v[1] && v[1] < v[2]) {
		t.Errorf("softmax did not preserve ordering: %v", v)
	}
	// Max subtraction must make a large constant offset a no-op.
	w := []float64{1001, 1002, 1003}
	SoftmaxInPlace(w)
	for i := range v {
		if math.Abs(v[i]-w[i]) > 1e-15 {
			t.Errorf("softmax is not shift invariant: %v vs %v", v, w)
		}
	}
	// Anything more than kMaxSoftmaxActivation below the max clips to exp(-86),
	// so it is small but never exactly zero.
	z := []float64{0, -1000}
	SoftmaxInPlace(z)
	if z[1] <= 0 {
		t.Errorf("softmax produced a zero probability %v; the -86 clip is missing", z[1])
	}
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `go test ./internal/nn/ -run 'TestTables|TestTanh|TestActivation|TestSoftmax' -v`
Expected: FAIL — `undefined: Tanh`.

- [ ] **Step 5: Implement**

Create `internal/nn/funcs.go`:

```go
// This file is a Go translation of src/lstm/functions.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "math"

const (
	// tableSize and tableScale are kTableSize and kScaleFactor. Table index i
	// corresponds to x = i/256; indices 0..4094 are usable because the
	// interpolation reads index+1.
	tableSize  = 4096
	tableScale = 256.0

	// maxSoftmaxActivation is kMaxSoftmaxActivation: the limit on the negative
	// range of the exp input, which guarantees a non-zero probability.
	maxSoftmaxActivation = 86.0
)

// Tanh is Tesseract's tabulated hyperbolic tangent: linear interpolation
// between 4096 samples at 1/256 spacing, mirrored for negative inputs and
// saturating to exactly 1 at |x| >= 4095/256.
//
// The comparison against tableSize-1 happens before the truncation to an index
// so that a large x cannot overflow the conversion; C++ relies on the
// implementation-defined behaviour of static_cast<unsigned> instead.
func Tanh(x float64) float64 {
	if x < 0 {
		return -Tanh(-x)
	}
	x *= tableScale
	if x >= tableSize-1 {
		return 1
	}
	i := int(x)
	t0, t1 := tanhTable[i], tanhTable[i+1]
	return t0 + (t1-t0)*(x-float64(i))
}

// Logistic is Tesseract's tabulated logistic sigmoid. The negative branch is
// the complement, 1 - Logistic(-x), so it returns exactly 0 at
// x <= -4095/256.
func Logistic(x float64) float64 {
	if x < 0 {
		return 1 - Logistic(-x)
	}
	x *= tableScale
	if x >= tableSize-1 {
		return 1
	}
	i := int(x)
	l0, l1 := logisticTable[i], logisticTable[i+1]
	return l0 + (l1-l0)*(x-float64(i))
}

// TanhInPlace is FuncInplace<GFunc>.
func TanhInPlace(v []float64) {
	for i, x := range v {
		v[i] = Tanh(x)
	}
}

// LogisticInPlace is FuncInplace<FFunc>.
func LogisticInPlace(v []float64) {
	for i, x := range v {
		v[i] = Logistic(x)
	}
}

// SoftmaxInPlace is SoftmaxInPlace from src/lstm/functions.h. Unlike the gate
// activations it is not tabulated: it subtracts the maximum, clips to
// [-86, 0], and calls exp directly. Go's math.Exp and the system libm may
// differ by under one ulp; that is the one divergence from Tesseract this
// package accepts and cannot remove without shipping an exp implementation.
func SoftmaxInPlace(v []float64) {
	if len(v) == 0 {
		return
	}
	maxOut := v[0]
	for _, x := range v[1:] {
		if x > maxOut {
			maxOut = x
		}
	}
	var total float64
	for i, x := range v {
		p := x - maxOut
		if p < -maxSoftmaxActivation {
			p = -maxSoftmaxActivation
		} else if p > 0 {
			p = 0
		}
		e := math.Exp(p)
		total += e
		v[i] = e
	}
	if total > 0 {
		for i := range v {
			v[i] /= total
		}
	}
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS. `TestTanhIsTheInterpolatedTableNotLibm` logs a max difference
around 1.5e-6.

- [ ] **Step 7: Commit**

```bash
git add internal/nn/gen internal/nn/tables.go internal/nn/funcs.go internal/nn/funcs_test.go Makefile
git commit -m "feat(nn): transcribe Tesseract's activation tables and add softmax"
```

---

## Task 5: Weight matrix and dot product

`WeightMatrix::MatrixDotVector` (`src/lstm/weightmatrix.cpp:99`) computes, for
each output row `i`, the dot product of the row's first `dim2-1` weights with
the input, then adds the row's last weight as the bias against an implicit 1.0.
Storage is row-major over `dim1` despite `matrix.h`'s "column-major" comment —
the effective address math is `w[i][j] == array_[i*dim2 + j]`.

**Files:**
- Create: `internal/nn/matrix.go`
- Test: `internal/nn/matrix_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces:

```go
type Matrix struct {
	Outputs int       // dim1
	Inputs  int       // dim2-1; the last column of each row is the bias
	W       []float64 // row-major, len == Outputs*(Inputs+1)
}

func NewMatrix(outputs, inputs int, w []float64) (*Matrix, error)
func (m *Matrix) DotVector(u, v []float64)
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/matrix_test.go`:

```go
package nn

import (
	"math"
	"testing"
)

func TestMatrixDotVectorAddsTheBiasColumn(t *testing.T) {
	// Two outputs, three inputs, so each row is 3 weights plus a bias.
	m, err := NewMatrix(2, 3, []float64{
		1, 2, 3, 10,
		0, 0, 1, -5,
	})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	v := make([]float64, 2)
	m.DotVector([]float64{1, 1, 1}, v)
	if v[0] != 16 || v[1] != -4 {
		t.Fatalf("DotVector() = %v; want [16 -4]", v)
	}
}

func TestNewMatrixRejectsWrongWeightCount(t *testing.T) {
	if _, err := NewMatrix(2, 3, make([]float64, 7)); err == nil {
		t.Fatal("NewMatrix with 7 weights for a 2x4 matrix: want error, got nil")
	}
}

// The Go spec allows the compiler to contract a*b+c into a single FMA, across
// statements. That would change the low bits of every dot product in the
// network relative to a scalar C++ build. DotVector must round each product
// before accumulating; this input distinguishes the two.
//
//	term 0: 1.0 * -(1+2^-26)              exact, total = -(1+2^-26)
//	term 1: (1+2^-27) * (1+2^-27)         exact value 1+2^-26+2^-54
//
// Rounded first, term 1 is 1+2^-26 and the total is exactly 0.
// Fused, the exact product is added and the total is 2^-54.
func TestMatrixDotVectorDoesNotFuseMultiplyAdd(t *testing.T) {
	e27 := math.Ldexp(1, -27)
	e26 := math.Ldexp(1, -26)
	m, err := NewMatrix(1, 2, []float64{1, 1 + e27, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	v := make([]float64, 1)
	m.DotVector([]float64{-(1 + e26), 1 + e27}, v)
	if v[0] != 0 {
		t.Fatalf("DotVector() = %g; want exactly 0. The compiler fused the multiply-add; wrap each product in float64(...)", v[0])
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run TestMatrix -v`
Expected: FAIL — `undefined: NewMatrix`.

- [ ] **Step 3: Implement**

Create `internal/nn/matrix.go`:

```go
// This file is a Go translation of src/lstm/weightmatrix.cpp and
// src/ccstruct/matrix.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Matrix is one of Tesseract's WeightMatrix objects on the float path: an
// Outputs x (Inputs+1) row-major array of float64 weights whose last column is
// the bias, applied against an implicit 1.0.
//
// GENERIC_2D_ARRAY's header comment claims column-major storage, but its
// address math is index(column, row) = column*dim2 + row, so w[i][j] is
// array_[i*dim2 + j] — contiguous rows of length dim2, with i the output and j
// the input.
type Matrix struct {
	Outputs int
	Inputs  int
	W       []float64
}

func NewMatrix(outputs, inputs int, w []float64) (*Matrix, error) {
	if outputs <= 0 || inputs < 0 {
		return nil, fmt.Errorf("nn: invalid matrix shape %dx%d", outputs, inputs+1)
	}
	if want := outputs * (inputs + 1); len(w) != want {
		return nil, fmt.Errorf("nn: matrix %dx%d needs %d weights, got %d", outputs, inputs+1, want, len(w))
	}
	return &Matrix{Outputs: outputs, Inputs: inputs, W: w}, nil
}

// DotVector is WeightMatrix::MatrixDotVector: v[i] = sum_j W[i][j]*u[j] plus
// the row's bias, added after the whole dot product exactly as
// MatrixDotVectorInternal does.
//
// Each product is wrapped in an explicit float64 conversion. The Go spec
// permits the compiler to contract a multiply and an add into a single fused
// operation "possibly across statements"; an explicit conversion is the
// documented way to force the intermediate rounding, and without it every dot
// product in the network drifts from a scalar C++ build.
func (m *Matrix) DotVector(u, v []float64) {
	stride := m.Inputs + 1
	for i := range m.Outputs {
		row := m.W[i*stride : (i+1)*stride]
		var total float64
		for j := range m.Inputs {
			total += float64(row[j] * u[j])
		}
		v[i] = total + row[m.Inputs]
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -run TestMatrix -v`
Expected: PASS, three tests. If
`TestMatrixDotVectorDoesNotFuseMultiplyAdd` fails, the `float64(...)` wrapper
was dropped — restore it, do not relax the test.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/matrix.go internal/nn/matrix_test.go
git commit -m "feat(nn): add the weight matrix and its unfused dot product"
```

---

## Task 6: The Layer interface and the plumbing layers

`Series`, `Parallel`, `Replicated` and the three `Reversed` variants
(`RTLReversed`, `TTBReversed`, `XYTranspose`) all derive from `Plumbing`, which
holds a stack of children. `Input::Forward` is `*output = input`.

`Reversed::Forward` transforms the input, runs its single child, and applies the
**same** transform to the child's output; both transforms are involutions, so
positions realign. `XYTranspose` is what collapses the 2-D feature map to a 1-D
sequence: it turns a (H=12, W=W/3) map into (H=W/3, W=12) so each transposed
*row* is one original x-column, the `SummLSTM` child emits one vector per row,
and transposing back gives (H=1, W=W/3).

`Parallel::Forward` runs every child on the same input and concatenates their
outputs along the feature axis (`CopyPacking`); all children must produce the
same timestep count.

**Files:**
- Create: `internal/nn/layer.go`
- Test: `internal/nn/layer_test.go`

**Interfaces:**
- Consumes: `Tensor`, `StrideMap`.
- Produces:

```go
type Layer interface {
	Name() string
	NumOutputs() int
	Forward(in *Tensor) (*Tensor, error)
}

type Input struct{ /* name, features */ }
type Series struct{ /* name, Stack []Layer */ }
type Parallel struct{ /* name, Stack []Layer */ }

type ReversalKind int
const (
	ReverseX ReversalKind = iota  // NT_XREVERSED / RTLReversed
	ReverseY                      // NT_YREVERSED / TTBReversed
	TransposeXY                   // NT_XYTRANSPOSE
)
type Reversed struct{ /* name, Kind, Sub Layer */ }

func NewInput(name string, features int) *Input
func NewSeries(name string, stack []Layer) (*Series, error)
func NewParallel(name string, stack []Layer) (*Parallel, error)
func NewReversed(name string, kind ReversalKind, sub Layer) *Reversed
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/layer_test.go`:

```go
package nn

import "testing"

// stamp is a test layer that writes its own id into every output feature, so
// composition order is observable.
type stamp struct {
	id       float64
	features int
}

func (s *stamp) Name() string    { return "stamp" }
func (s *stamp) NumOutputs() int { return s.features }
func (s *stamp) Forward(in *Tensor) (*Tensor, error) {
	out := NewTensor(in.Map, s.features)
	row := make([]float64, s.features)
	for i := range row {
		row[i] = s.id
	}
	for t := range in.Map.Len() {
		out.WriteTimeStep(t, row)
	}
	return out, nil
}

func TestInputIsIdentity(t *testing.T) {
	in := NewTensor(StrideMap{Height: 2, Width: 2}, 1)
	in.WriteTimeStep(3, []float64{7})
	out, err := NewInput("Input", 1).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 1)
	out.ReadTimeStep(3, got)
	if got[0] != 7 {
		t.Errorf("Input.Forward changed the data: %v", got[0])
	}
}

func TestSeriesRunsInOrder(t *testing.T) {
	s, err := NewSeries("Series", []Layer{&stamp{id: 1, features: 2}, &stamp{id: 2, features: 3}})
	if err != nil {
		t.Fatalf("NewSeries() error = %v", err)
	}
	out, err := s.Forward(NewTensor(StrideMap{Height: 1, Width: 1}, 1))
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Features != 3 {
		t.Fatalf("Series output features = %d; want 3 (the last layer's)", out.Features)
	}
	got := make([]float64, 3)
	out.ReadTimeStep(0, got)
	for _, v := range got {
		if v != 2 {
			t.Fatalf("Series output = %v; want all 2 (the last layer ran last)", got)
		}
	}
}

func TestParallelConcatenatesFeatures(t *testing.T) {
	p, err := NewParallel("Parallel", []Layer{&stamp{id: 1, features: 2}, &stamp{id: 2, features: 3}})
	if err != nil {
		t.Fatalf("NewParallel() error = %v", err)
	}
	if p.NumOutputs() != 5 {
		t.Fatalf("Parallel.NumOutputs() = %d; want 5", p.NumOutputs())
	}
	out, err := p.Forward(NewTensor(StrideMap{Height: 1, Width: 1}, 1))
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 5)
	out.ReadTimeStep(0, got)
	want := []float64{1, 1, 2, 2, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Parallel output = %v; want %v", got, want)
		}
	}
}

// A Reversed wrapping an identity child must be a no-op overall, because the
// transform is applied to the input and again to the output.
func TestReversedIsANoOpAroundIdentity(t *testing.T) {
	for _, kind := range []ReversalKind{ReverseX, ReverseY, TransposeXY} {
		in := NewTensor(StrideMap{Height: 3, Width: 4}, 1)
		for tt := range in.Map.Len() {
			in.WriteTimeStep(tt, []float64{float64(tt)})
		}
		out, err := NewReversed("rev", kind, NewInput("id", 1)).Forward(in)
		if err != nil {
			t.Fatalf("kind %d: Forward() error = %v", kind, err)
		}
		if out.Map != in.Map {
			t.Fatalf("kind %d: output map = %v; want %v", kind, out.Map, in.Map)
		}
		got := make([]float64, 1)
		for tt := range in.Map.Len() {
			out.ReadTimeStep(tt, got)
			if got[0] != float64(tt) {
				t.Fatalf("kind %d: t=%d = %v; want %v", kind, tt, got[0], float64(tt))
			}
		}
	}
}

// The transform itself must be the documented one: RTLReversed mirrors x
// within each row, XYTranspose swaps the axes.
func TestReverseDataTransforms(t *testing.T) {
	src := NewTensor(StrideMap{Height: 2, Width: 3}, 1)
	for tt := range src.Map.Len() {
		src.WriteTimeStep(tt, []float64{float64(tt)})
	}
	got := make([]float64, 1)

	x := reverseData(src, ReverseX)
	x.ReadTimeStep(x.Map.T(0, 0), got)
	if got[0] != 2 {
		t.Errorf("ReverseX dst(0,0) = %v; want 2 (src(0,2))", got[0])
	}

	tr := reverseData(src, TransposeXY)
	if tr.Map != (StrideMap{Height: 3, Width: 2}) {
		t.Fatalf("TransposeXY map = %v; want {3 2}", tr.Map)
	}
	tr.ReadTimeStep(tr.Map.T(2, 1), got)
	if got[0] != float64(src.Map.T(1, 2)) {
		t.Errorf("TransposeXY dst(2,1) = %v; want src(1,2) = %v", got[0], float64(src.Map.T(1, 2)))
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run 'TestInput|TestSeries|TestParallel|TestReverse' -v`
Expected: FAIL — `undefined: NewInput`.

- [ ] **Step 3: Implement**

Create `internal/nn/layer.go`:

```go
// This file is a Go translation of src/lstm/network.cpp, src/lstm/plumbing.cpp,
// src/lstm/series.cpp, src/lstm/parallel.cpp, src/lstm/reversed.cpp and
// src/lstm/input.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Layer is one node of a runnable network graph.
//
// Forward returns a freshly allocated tensor rather than filling a caller-owned
// buffer. Tesseract threads a NetworkScratch through every call to recycle
// buffers; that is a performance concern, and reintroducing it before there is
// a measurement would be premature.
type Layer interface {
	// Name is the layer's serialized name. Activation dumps key on it.
	Name() string
	// NumOutputs is the feature count of the tensor Forward produces.
	NumOutputs() int
	// Forward propagates in and returns the layer's output.
	Forward(in *Tensor) (*Tensor, error)
}

// Input is NT_INPUT. Input::Forward is `*output = input`; returning the input
// tensor unchanged is safe because no layer mutates its own input.
type Input struct {
	name     string
	features int
}

func NewInput(name string, features int) *Input { return &Input{name: name, features: features} }

func (l *Input) Name() string                    { return l.name }
func (l *Input) NumOutputs() int                 { return l.features }
func (l *Input) Forward(in *Tensor) (*Tensor, error) { return in, nil }

// Series is NT_SERIES: each layer's output is the next layer's input.
type Series struct {
	name  string
	Stack []Layer
}

func NewSeries(name string, stack []Layer) (*Series, error) {
	if len(stack) == 0 {
		return nil, fmt.Errorf("nn: series %q has no layers", name)
	}
	return &Series{name: name, Stack: stack}, nil
}

func (l *Series) Name() string    { return l.name }
func (l *Series) NumOutputs() int { return l.Stack[len(l.Stack)-1].NumOutputs() }

func (l *Series) Forward(in *Tensor) (*Tensor, error) {
	cur := in
	for _, sub := range l.Stack {
		out, err := sub.Forward(cur)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", l.name, err)
		}
		cur = out
	}
	return cur, nil
}

// Parallel is NT_PARALLEL and NT_REPLICATED: every child sees the same input
// and their outputs are concatenated along the feature axis, in stack order.
type Parallel struct {
	name  string
	Stack []Layer
}

func NewParallel(name string, stack []Layer) (*Parallel, error) {
	if len(stack) == 0 {
		return nil, fmt.Errorf("nn: parallel %q has no layers", name)
	}
	return &Parallel{name: name, Stack: stack}, nil
}

func (l *Parallel) Name() string { return l.name }

func (l *Parallel) NumOutputs() int {
	n := 0
	for _, sub := range l.Stack {
		n += sub.NumOutputs()
	}
	return n
}

func (l *Parallel) Forward(in *Tensor) (*Tensor, error) {
	parts := make([]*Tensor, len(l.Stack))
	for i, sub := range l.Stack {
		out, err := sub.Forward(in)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", l.name, err)
		}
		if i > 0 && out.Map != parts[0].Map {
			return nil, fmt.Errorf("nn: parallel %q child %d produced map %v; want %v", l.name, i, out.Map, parts[0].Map)
		}
		parts[i] = out
	}
	out := NewTensor(parts[0].Map, l.NumOutputs())
	for t := range out.Map.Len() {
		dst := out.Row(t)
		off := 0
		for _, p := range parts {
			copy(dst[off:off+p.Features], p.Row(t))
			off += p.Features
		}
	}
	return out, nil
}

// ReversalKind selects which of Reversed's three transforms applies.
type ReversalKind int

const (
	// ReverseX is NT_XREVERSED (RTLReversed): mirror x within each row.
	ReverseX ReversalKind = iota
	// ReverseY is NT_YREVERSED (TTBReversed): mirror y within each column.
	ReverseY
	// TransposeXY is NT_XYTRANSPOSE: swap the two spatial dimensions.
	TransposeXY
)

// Reversed is Tesseract's Reversed plumbing: it applies its transform to the
// input, runs its single child, and applies the same transform to the child's
// output. Both transforms are involutions, so output positions realign with
// input positions.
type Reversed struct {
	name string
	Kind ReversalKind
	Sub  Layer
}

func NewReversed(name string, kind ReversalKind, sub Layer) *Reversed {
	return &Reversed{name: name, Kind: kind, Sub: sub}
}

func (l *Reversed) Name() string    { return l.name }
func (l *Reversed) NumOutputs() int { return l.Sub.NumOutputs() }

func (l *Reversed) Forward(in *Tensor) (*Tensor, error) {
	out, err := l.Sub.Forward(reverseData(in, l.Kind))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", l.name, err)
	}
	return reverseData(out, l.Kind), nil
}

// reverseData is NetworkIO::CopyWithXReversal, CopyWithYReversal and
// CopyWithXYTranspose. Cadmus has one batch, so the per-batch valid width
// Tesseract mirrors within is simply the map width.
func reverseData(src *Tensor, kind ReversalKind) *Tensor {
	m := src.Map
	if kind == TransposeXY {
		m = m.TransposeXY()
	}
	dst := NewTensor(m, src.Features)
	for y := range src.Map.Height {
		for x := range src.Map.Width {
			var dt int
			switch kind {
			case ReverseX:
				dt = m.T(y, src.Map.Width-1-x)
			case ReverseY:
				dt = m.T(src.Map.Height-1-y, x)
			case TransposeXY:
				dt = m.T(x, y)
			}
			dst.CopyTimeStep(dt, src, src.Map.T(y, x))
		}
	}
	return dst
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/layer.go internal/nn/layer_test.go
git commit -m "feat(nn): add the layer interface and the plumbing layers"
```

---

## Task 7: TRand and Convolve

`Convolve` holds **no weights**. It is a pure 3x3 im2col gather at stride 1: the
output map is the same size as the input, and each output timestep stacks the
`ni_` channels of its 9 neighbours into `9*ni_` features. The learned part is the
`Tanh` `FullyConnected` layer that follows it (`[16x10]` for eng: 9 inputs plus
a bias).

Feature index for the tap at offset `(dx, dy)` and channel `c`:
`((dx+half_x)*(2*half_y+1) + (dy+half_y))*ni + c` — x-major, then y, then channel.

**Off-image taps are filled with random values, not zeros.** `NetworkIO::Randomize`
draws `randomizer->SignedRand(1.0)` per feature from the recognizer's LCG. The
whole 3-value column is drawn when the *x* tap is off-image; a single value when
only the *y* tap is off. The draw count and order are position-dependent, so
reproducing them needs the same LCG, the same seed, and the same traversal.

The seed is `sample_iteration_ * 0x10000001`, followed by one discarded
`IntRand()` (`LSTMRecognizer::SetRandomSeed`). `RecognizeLine` calls
`SetRandomSeed()` immediately before `PreparePixInput`.

**`Convolve` is not the first consumer — `Copy2DImage` is, and only sometimes.**
For `eng` the input shape's width is 0, so the stride map takes the scaled pix's
own width and the per-row tail loop draws nothing. But the shape's *height* is
36, and `Copy2DImage` iterates `y < 36` regardless of `pixGetHeight(pix)`: if
`pixScale` rounded to 35 rows, the whole 36th row is `Randomize`d, consuming
`width` draws before `Convolve` ever runs. Task 13 reproduces that; the same
`*nn.Rand` must be handed to `Normalize` and to every `Convolve` in the graph, in
that order. Only when the scaled height lands exactly on 36 does the randomizer
enter `Convolve::Forward` in the state `set_seed(s); IntRand()` leaves it.

**Files:**
- Create: `internal/nn/rand.go`, `internal/nn/convolve.go`
- Test: `internal/nn/convolve_test.go`

**Interfaces:**
- Consumes: `Tensor`, `StrideMap`.
- Produces:

```go
type Rand struct{ /* seed uint64 */ }
func NewRand(seed uint64) *Rand    // set_seed then one discarded IntRand
func (r *Rand) IntRand() int32
func (r *Rand) SignedRand(rng float64) float64

type Convolve struct{ /* name, HalfX, HalfY, NI int; Rand *Rand */ }
func NewConvolve(name string, ni, halfX, halfY int, rnd *Rand) *Convolve
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/convolve_test.go`:

```go
package nn

import "testing"

// The LCG and the >>33 extraction are load-bearing: Convolve's edge padding
// consumes them in a position-dependent order, so a wrong generator shows up
// as noise on the image border and nowhere else.
func TestRandMatchesTesseractLCG(t *testing.T) {
	// Reproduce TRand by hand: seed_ = 5, then two iterations.
	const mul, inc = 6364136223846793005, 1442695040888963407
	seed := uint64(5)
	seed = seed*mul + inc
	first := int32(seed >> 33)
	seed = seed*mul + inc
	second := int32(seed >> 33)

	// NewRand performs the discarded IntRand that SetRandomSeed does, so the
	// first value it hands out is the *second* iterate.
	r := NewRand(5)
	if got := r.IntRand(); got != second {
		t.Fatalf("IntRand() = %d; want %d (NewRand must discard one iterate, first was %d)", got, second, first)
	}
	if got := r.IntRand() < 0; got {
		t.Fatal("IntRand() returned a negative value; seed_>>33 must fit in 31 bits")
	}
}

// The range is CLOSED at both ends. IntRand() returns seed_>>33, whose maximum
// is exactly INT32_MAX, so range*2*INT32_MAX/INT32_MAX - range == +range
// exactly. Tesseract's own comment says "in the range [-range, range]". A
// half-open assertion passes with probability 1 - 2^-31 per draw, which makes it
// a time bomb rather than a test.
func TestSignedRandRange(t *testing.T) {
	r := NewRand(814136 * 0x10000001)
	for range 1000 {
		v := r.SignedRand(1.0)
		if v < -1 || v > 1 {
			t.Fatalf("SignedRand(1.0) = %v; want [-1, 1]", v)
		}
	}
}

// A 1x1 map with ni=1 means every one of the 9 taps except the centre is
// off-image, so 8 of the 9 output features are random draws and only the
// centre carries the input.
func TestConvolveGathersAndRandomizesEdges(t *testing.T) {
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	in.WriteTimeStep(0, []float64{0.5})

	c := NewConvolve("Convolve", 1, 1, 1, NewRand(1))
	if c.NumOutputs() != 9 {
		t.Fatalf("NumOutputs() = %d; want 9", c.NumOutputs())
	}
	out, err := c.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != in.Map {
		t.Fatalf("Convolve changed the map to %v; stride is 1, it must not", out.Map)
	}
	got := make([]float64, 9)
	out.ReadTimeStep(0, got)
	// Feature index of the (dx=0, dy=0) tap is ((0+1)*3 + (0+1))*1 = 4.
	if got[4] != 0.5 {
		t.Errorf("centre tap (feature 4) = %v; want 0.5", got[4])
	}
	for i, v := range got {
		if i == 4 {
			continue
		}
		if v == 0 {
			t.Errorf("feature %d = 0; off-image taps must be randomized, not zeroed", i)
		}
	}
}

// Interior taps must gather the correct neighbours, x-major then y.
func TestConvolveFeatureLayout(t *testing.T) {
	in := NewTensor(StrideMap{Height: 3, Width: 3}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt) + 1})
	}
	out, err := NewConvolve("Convolve", 1, 1, 1, NewRand(1)).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 9)
	out.ReadTimeStep(in.Map.T(1, 1), got) // the fully interior position
	// f = ((dx+1)*3 + (dy+1))*1, source value = T(1+dy, 1+dx) + 1
	for _, tc := range []struct{ dx, dy int }{
		{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1},
	} {
		f := ((tc.dx+1)*3 + (tc.dy + 1))
		want := float64(in.Map.T(1+tc.dy, 1+tc.dx)) + 1
		if got[f] != want {
			t.Errorf("tap (dx=%d,dy=%d) at feature %d = %v; want %v", tc.dx, tc.dy, f, got[f], want)
		}
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run 'TestRand|TestSignedRand|TestConvolve' -v`
Expected: FAIL — `undefined: NewRand`.

- [ ] **Step 3: Implement**

Create `internal/nn/rand.go`:

```go
// This file is a Go translation of the TRand class in src/ccutil/helpers.h and
// LSTMRecognizer::SetRandomSeed in src/lstm/lstmrecognizer.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

// Rand is Tesseract's TRand: a linear congruential generator with Knuth's
// constants, used to fill the off-image taps of a convolution.
//
// This is not a general-purpose RNG and must not be replaced with math/rand.
// The draw sequence is part of the network's output: Convolve consumes it in a
// position-dependent order at every image border, so a different generator
// changes the activations along the entire edge of the feature map.
type Rand struct {
	seed uint64
}

// NewRand seeds the generator and discards one iterate, reproducing
// LSTMRecognizer::SetRandomSeed, which calls set_seed followed by IntRand.
// The caller passes sample_iteration * 0x10000001, from the LSTM trailer.
func NewRand(seed uint64) *Rand {
	r := &Rand{seed: seed}
	r.IntRand()
	return r
}

// IntRand steps the generator and returns a value in [0, math.MaxInt32].
func (r *Rand) IntRand() int32 {
	r.seed = r.seed*6364136223846793005 + 1442695040888963407
	return int32(r.seed >> 33)
}

// SignedRand returns a value in [-rng, rng] — closed at both ends, because
// IntRand()'s maximum is exactly INT32_MAX and the division is therefore exactly
// 1 at the top of the range. Tesseract's comment on TRand::SignedRand says the
// same; do not "tighten" this to a half-open interval.
func (r *Rand) SignedRand(rng float64) float64 {
	return rng*2.0*float64(r.IntRand())/2147483647.0 - rng
}
```

Create `internal/nn/convolve.go`:

```go
// This file is a Go translation of src/lstm/convolve.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Convolve is NT_CONVOLVE: a stride-1 gather that stacks the
// (2*HalfX+1) x (2*HalfY+1) neighbourhood of every position into
// (2*HalfX+1)*(2*HalfY+1)*NI output features. It holds no weights — the
// learned part is the FullyConnected layer that follows it, so `Ct3,3,16`
// is this layer plus a 16x10 Tanh.
//
// Feature order is x-major, then y, then channel:
//
//	f = ((dx+HalfX) * (2*HalfY+1) + (dy+HalfY)) * NI + c
//
// Off-image taps are filled from the recognizer's LCG, not with zeros. Whether
// that matters for recognized text is unmeasured; what is certain is that
// substituting zeros will not match Tesseract's activations, so the border of
// every feature map would differ and Task 14's per-layer diff would be unusable.
type Convolve struct {
	name         string
	HalfX, HalfY int
	NI           int
	Rand         *Rand
}

func NewConvolve(name string, ni, halfX, halfY int, rnd *Rand) *Convolve {
	return &Convolve{name: name, HalfX: halfX, HalfY: halfY, NI: ni, Rand: rnd}
}

func (l *Convolve) Name() string { return l.name }

// NumOutputs recomputes no_ exactly as Convolve::DeSerialize does, overwriting
// whatever the serialized header claimed.
func (l *Convolve) NumOutputs() int {
	return l.NI * (2*l.HalfX + 1) * (2*l.HalfY + 1)
}

func (l *Convolve) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: convolve %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	yScale := 2*l.HalfY + 1
	out := NewTensor(in.Map, l.NumOutputs())
	for t := range in.Map.Len() {
		row := out.Row(t)
		outIX := 0
		for dx := -l.HalfX; dx <= l.HalfX; dx, outIX = dx+1, outIX+yScale*l.NI {
			if _, ok := in.Map.Offset(t, 0, dx); !ok {
				// The whole column of taps is outside the image.
				l.randomize(row[outIX : outIX+yScale*l.NI])
				continue
			}
			outIY := outIX
			for dy := -l.HalfY; dy <= l.HalfY; dy, outIY = dy+1, outIY+l.NI {
				src, ok := in.Map.Offset(t, dy, dx)
				if !ok {
					l.randomize(row[outIY : outIY+l.NI])
					continue
				}
				copy(row[outIY:outIY+l.NI], in.Row(src))
			}
		}
	}
	return out, nil
}

// randomize is NetworkIO::Randomize on the float path.
func (l *Convolve) randomize(dst []float32) {
	for i := range dst {
		dst[i] = float32(l.Rand.SignedRand(1.0))
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS.

- [ ] **Step 5: Verify the draw order against the C++ traversal**

The plan's loop nests `dy` inside `dx` and checks the `dx` tap first, which is
what `convolve.cpp` does — but the draw *count* per position is what the LCG
sequence depends on, and getting it wrong is invisible until Task 14. Confirm by
reading the source:

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
sed -n '/void Convolve::Forward/,/^}/p' "$SP/tess/src/lstm/convolve.cpp"
```

Check three things against the Go: the outer loop is over `x` with
`out_ix += y_scale * ni_`; an out-of-range `x` draws `y_scale * ni_` values in
one call; an out-of-range `y` draws `ni_`. **If any differs, fix the Go to match
the C++ and say so in the task report** — the C++ is the specification, this
plan is not.

- [ ] **Step 6: Commit**

```bash
git add internal/nn/rand.go internal/nn/convolve.go internal/nn/convolve_test.go
git commit -m "feat(nn): add the TRand generator and the convolution gather"
```

---

## Task 8: Maxpool and Reconfig

`Reconfig` stacks an `XScale` x `YScale` block of input timesteps into
`ni*XScale*YScale` output features. `Maxpool` shares its deserialization and its
traversal but overrides the output depth to `ni` and takes the elementwise
maximum over the block instead of concatenating.

Both call `ResizeScaled`, which floor-divides both dimensions, so **a partial
trailing window is dropped, not padded** — and because the dimensions are
floored first, every tap in every window is in range and the `AddOffset`
validity guard never fires.

**Files:**
- Create: `internal/nn/reconfig.go`
- Test: `internal/nn/reconfig_test.go`

**Interfaces:**
- Consumes: `Tensor`, `StrideMap`.
- Produces:

```go
type Reconfig struct{ /* name, XScale, YScale, NI int; Max bool */ }
func NewReconfig(name string, ni, xScale, yScale int) *Reconfig
func NewMaxpool(name string, ni, xScale, yScale int) *Reconfig
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/reconfig_test.go`:

```go
package nn

import "testing"

func TestMaxpoolTakesTheBlockMaximum(t *testing.T) {
	in := NewTensor(StrideMap{Height: 3, Width: 6}, 2)
	// Feature 0 counts up, feature 1 counts down, so the max of each is at a
	// different corner of the window.
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt), float64(in.Map.Len() - tt)})
	}
	mp := NewMaxpool("Maxpool", 2, 3, 3)
	if mp.NumOutputs() != 2 {
		t.Fatalf("Maxpool.NumOutputs() = %d; want 2 (ni, not ni*xs*ys)", mp.NumOutputs())
	}
	out, err := mp.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != (StrideMap{Height: 1, Width: 2}) {
		t.Fatalf("Maxpool map = %v; want {1 2}", out.Map)
	}
	got := make([]float64, 2)
	out.ReadTimeStep(0, got)
	// Window covers y in [0,3), x in [0,3): timesteps 0,1,2,6,7,8,12,13,14.
	if got[0] != 14 {
		t.Errorf("feature 0 = %v; want 14", got[0])
	}
	if got[1] != float64(in.Map.Len()) {
		t.Errorf("feature 1 = %v; want %v", got[1], float64(in.Map.Len()))
	}
}

// Height 36 at Mp3,3 gives 12 rows; a width of 100 gives 33 columns and the
// last column of input is discarded.
func TestMaxpoolDropsPartialWindows(t *testing.T) {
	in := NewTensor(StrideMap{Height: 36, Width: 100}, 1)
	out, err := NewMaxpool("Maxpool", 1, 3, 3).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if out.Map != (StrideMap{Height: 12, Width: 33}) {
		t.Errorf("Maxpool map = %v; want {12 33}", out.Map)
	}
}

func TestReconfigStacksTheBlock(t *testing.T) {
	in := NewTensor(StrideMap{Height: 2, Width: 2}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{float64(tt) + 1})
	}
	rc := NewReconfig("Reconfig", 1, 2, 2)
	if rc.NumOutputs() != 4 {
		t.Fatalf("Reconfig.NumOutputs() = %d; want 4", rc.NumOutputs())
	}
	out, err := rc.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 4)
	out.ReadTimeStep(0, got)
	// Feature offset is (x*y_scale + y)*ni, source is T(y, x)+1.
	want := []float64{1, 3, 2, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Reconfig output = %v; want %v", got, want)
		}
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run 'TestMaxpool|TestReconfig' -v`
Expected: FAIL — `undefined: NewMaxpool`.

- [ ] **Step 3: Implement**

Create `internal/nn/reconfig.go`:

```go
// This file is a Go translation of src/lstm/reconfig.cpp and
// src/lstm/maxpool.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Reconfig is NT_RECONFIG and, with Max set, NT_MAXPOOL. The two share their
// serialized fields and their traversal; Maxpool differs only in taking the
// elementwise maximum over the window instead of concatenating it, and in
// forcing the output depth back to ni.
//
// The output map is floor-divided in both dimensions (StrideMap::ScaleXY), so a
// partial trailing window is dropped rather than padded — and because the
// dimensions are floored first, every tap of every window is in range.
type Reconfig struct {
	name           string
	XScale, YScale int
	NI             int
	Max            bool
}

func NewReconfig(name string, ni, xScale, yScale int) *Reconfig {
	return &Reconfig{name: name, XScale: xScale, YScale: yScale, NI: ni}
}

func NewMaxpool(name string, ni, xScale, yScale int) *Reconfig {
	return &Reconfig{name: name, XScale: xScale, YScale: yScale, NI: ni, Max: true}
}

func (l *Reconfig) Name() string { return l.name }

func (l *Reconfig) NumOutputs() int {
	if l.Max {
		return l.NI
	}
	return l.NI * l.XScale * l.YScale
}

func (l *Reconfig) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	if l.XScale <= 0 || l.YScale <= 0 {
		return nil, fmt.Errorf("nn: %q has invalid scale %dx%d", l.name, l.XScale, l.YScale)
	}
	outMap := in.Map.ScaleXY(l.XScale, l.YScale)
	out := NewTensor(outMap, l.NumOutputs())
	for oy := range outMap.Height {
		for ox := range outMap.Width {
			ot := outMap.T(oy, ox)
			origin := in.Map.T(oy*l.YScale, ox*l.XScale)
			if l.Max {
				out.CopyTimeStep(ot, in, origin)
			}
			for dx := range l.XScale {
				for dy := range l.YScale {
					src := in.Map.T(oy*l.YScale+dy, ox*l.XScale+dx)
					if l.Max {
						out.MaxpoolTimeStep(ot, in, src)
					} else {
						out.CopyTimeStepPart(ot, (dx*l.YScale+dy)*l.NI, l.NI, in, src, 0)
					}
				}
			}
		}
	}
	return out, nil
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/reconfig.go internal/nn/reconfig_test.go
git commit -m "feat(nn): add maxpool and reconfig"
```

---

## Task 9: FullyConnected

One weight matrix, one matvec per timestep, then the type's nonlinearity in
place. `NT_TANH` uses the tabulated tanh, `NT_LOGISTIC` the tabulated logistic,
`NT_RELU` `max(0,x)`, `NT_LINEAR` nothing, `NT_SOFTMAX` and `NT_SOFTMAX_NO_CTC`
the softmax. `NT_POSCLIP` and `NT_SYMCLIP` clip to `[0,1]` and `[-1,1]`.

**Files:**
- Create: `internal/nn/fullyconnected.go`
- Test: `internal/nn/fullyconnected_test.go`

**Interfaces:**
- Consumes: `Matrix`, `Tensor`, `Tanh`, `Logistic`, `SoftmaxInPlace`.
- Produces:

```go
type Activation int
const (
	ActLinear Activation = iota
	ActTanh
	ActLogistic
	ActRelu
	ActSoftmax
	ActPosClip
	ActSymClip
)

type FullyConnected struct{ /* name string; Act Activation; W *Matrix */ }
func NewFullyConnected(name string, act Activation, w *Matrix) *FullyConnected
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/fullyconnected_test.go`:

```go
package nn

import (
	"math"
	"testing"
)

func TestFullyConnectedAppliesMatrixThenActivation(t *testing.T) {
	// One output, one input: y = tanh(2*x + 0).
	w, err := NewMatrix(1, 1, []float64{2, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	fc := NewFullyConnected("ConvNL", ActTanh, w)
	if fc.NumOutputs() != 1 {
		t.Fatalf("NumOutputs() = %d; want 1", fc.NumOutputs())
	}
	in := NewTensor(StrideMap{Height: 1, Width: 2}, 1)
	in.WriteTimeStep(0, []float64{0.25})
	in.WriteTimeStep(1, []float64{-0.25})
	out, err := fc.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 1)
	out.ReadTimeStep(0, got)
	if want := float64(float32(Tanh(0.5))); got[0] != want {
		t.Errorf("t=0 = %v; want %v", got[0], want)
	}
	out.ReadTimeStep(1, got)
	if want := float64(float32(Tanh(-0.5))); got[0] != want {
		t.Errorf("t=1 = %v; want %v", got[0], want)
	}
	if out.Map != in.Map {
		t.Errorf("FullyConnected changed the map to %v", out.Map)
	}
}

func TestFullyConnectedSoftmaxNormalisesPerTimestep(t *testing.T) {
	w, err := NewMatrix(3, 1, []float64{1, 0, 2, 0, 3, 0})
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 1)
	in.WriteTimeStep(0, []float64{1})
	out, err := NewFullyConnected("Output", ActSoftmax, w).Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	got := make([]float64, 3)
	out.ReadTimeStep(0, got)
	var sum float64
	for _, p := range got {
		sum += p
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("softmax outputs sum to %v; want 1", sum)
	}
}

func TestFullyConnectedRejectsWrongInputWidth(t *testing.T) {
	w, _ := NewMatrix(1, 3, []float64{1, 1, 1, 0})
	in := NewTensor(StrideMap{Height: 1, Width: 1}, 2)
	if _, err := NewFullyConnected("x", ActLinear, w).Forward(in); err == nil {
		t.Fatal("Forward with 2 features into a 3-input matrix: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run TestFullyConnected -v`
Expected: FAIL — `undefined: NewFullyConnected`.

- [ ] **Step 3: Implement**

Create `internal/nn/fullyconnected.go`:

```go
// This file is a Go translation of src/lstm/fullyconnected.cpp from Tesseract
// OCR (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Activation selects the nonlinearity FullyConnected::ForwardTimeStep applies.
type Activation int

const (
	ActLinear   Activation = iota // NT_LINEAR
	ActTanh                       // NT_TANH
	ActLogistic                   // NT_LOGISTIC
	ActRelu                       // NT_RELU
	ActSoftmax                    // NT_SOFTMAX and NT_SOFTMAX_NO_CTC
	ActPosClip                    // NT_POSCLIP
	ActSymClip                    // NT_SYMCLIP
)

// FullyConnected is Tesseract's FullyConnected: one weight matrix applied per
// timestep, followed by the type's nonlinearity.
//
// Tesseract iterates t over the raw buffer width and zeroes the padding
// afterwards with ZeroInvalidElements. Cadmus has one batch and therefore no
// padding, so every timestep is valid and the cleanup pass does not exist.
type FullyConnected struct {
	name string
	Act  Activation
	W    *Matrix
}

func NewFullyConnected(name string, act Activation, w *Matrix) *FullyConnected {
	return &FullyConnected{name: name, Act: act, W: w}
}

func (l *FullyConnected) Name() string    { return l.name }
func (l *FullyConnected) NumOutputs() int { return l.W.Outputs }

func (l *FullyConnected) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.W.Inputs {
		return nil, fmt.Errorf("nn: %q got %d input features, want %d", l.name, in.Features, l.W.Inputs)
	}
	out := NewTensor(in.Map, l.W.Outputs)
	u := make([]float64, l.W.Inputs)
	v := make([]float64, l.W.Outputs)
	for t := range in.Map.Len() {
		in.ReadTimeStep(t, u)
		l.W.DotVector(u, v)
		l.activate(v)
		out.WriteTimeStep(t, v)
	}
	return out, nil
}

func (l *FullyConnected) activate(v []float64) {
	switch l.Act {
	case ActLinear:
	case ActTanh:
		TanhInPlace(v)
	case ActLogistic:
		LogisticInPlace(v)
	case ActRelu:
		for i, x := range v {
			if x < 0 {
				v[i] = 0
			}
		}
	case ActSoftmax:
		SoftmaxInPlace(v)
	case ActPosClip:
		clipInPlace(v, 0, 1)
	case ActSymClip:
		clipInPlace(v, -1, 1)
	}
}

func clipInPlace(v []float64, lo, hi float64) {
	for i, x := range v {
		if x < lo {
			v[i] = lo
		} else if x > hi {
			v[i] = hi
		}
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/fullyconnected.go internal/nn/fullyconnected_test.go
git commit -m "feat(nn): add the fully connected layer"
```

---

## Task 10: The LSTM cell

The cell, exactly as `LSTM::Forward` computes it (`src/lstm/lstm.cpp:291-490`):

```
source[t] = [ input[t] (NI) | h_{t-1} (NS) | bias implied ]     columns 0..NA-1
u         = source[t] read back as float64                      <- float32 round trip
CI  = Tanh    (W_CI  . u)
GI  = Logistic(W_GI  . u)
GF1 = Logistic(W_GF1 . u)
GO  = Logistic(W_GO  . u)
c_t = clip(GF1 * c_{t-1} + CI * GI, -100, +100)
h_t = Tanh(c_t) * GO
```

Four things are easy to get wrong and all four are load-bearing:

1. **`h_{t-1}` round-trips through float32** before re-entering the matvec,
   because `source_` is a `NetworkIO`. The rounding is inside the recurrence.
2. **The clip is applied every timestep** and the clipped value is what carries
   forward. `kStateClip = 100.0`.
3. **`h_t` uses the already-clipped `c_t`.**
4. **State and output are zeroed at the end of every row.** Every stride-map row
   is an independent sequence starting from `c = h = 0`.

`NT_LSTM_SUMMARY` differs in exactly two ways: the output map is
`ReduceWidthTo1`, and only the **last** `h` of each row is emitted. "Summary" is
not a sum, a mean or a max — it is the final hidden output, and the cell state
is discarded.

**Out of scope, hard-error:** 2-D LSTMs (the `GFS` gate) and the
`NT_LSTM_SOFTMAX` / `NT_LSTM_SOFTMAX_ENCODED` nested-softmax variants. Neither
appears in any `tessdata_best` Latin model, neither was exercised by the
research, and shipping unexercised code is how a port acquires silent bugs.

**Files:**
- Create: `internal/nn/lstm.go`
- Test: `internal/nn/lstm_test.go`

**Interfaces:**
- Consumes: `Matrix`, `Tensor`, `Tanh`, `Logistic`.
- Produces:

```go
type LSTM struct {
	// name, NI, NS, NA int; Summary bool; Gates [4]*Matrix
}
func NewLSTM(name string, ni, na int, summary bool, gates [4]*Matrix) (*LSTM, error)

const (
	GateCI = iota  // cell input,  Tanh
	GateGI         // input gate,  Logistic
	GateGF1        // forget gate, Logistic
	GateGO         // output gate, Logistic
)
```

- [ ] **Step 1: Write the failing test**

Create `internal/nn/lstm_test.go`:

```go
package nn

import (
	"math"
	"strings"
	"testing"
)

// gates builds four 1x(na+1) matrices with the given input weight, recurrent
// weight and bias, so the cell can be driven analytically.
func gates(t *testing.T, in, rec, bias [4]float64) [4]*Matrix {
	t.Helper()
	var out [4]*Matrix
	for g := range out {
		m, err := NewMatrix(1, 2, []float64{in[g], rec[g], bias[g]})
		if err != nil {
			t.Fatalf("NewMatrix() error = %v", err)
		}
		out[g] = m
	}
	return out
}

func TestLSTMCellMatchesTheEquations(t *testing.T) {
	// Recurrent weights zero, so h_{t-1} does not feed back and each timestep
	// depends only on the input and the carried state.
	g := gates(t,
		[4]float64{1, 1, 1, 1}, // input weights
		[4]float64{0, 0, 0, 0}, // recurrent weights
		[4]float64{0, 0, 0, 0}) // biases
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 2}, 1)
	in.WriteTimeStep(0, []float64{0.5})
	in.WriteTimeStep(1, []float64{0.25})
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	// Reference, computed with the same tabulated activations.
	state := 0.0
	got := make([]float64, 1)
	for tt, x := range []float64{0.5, 0.25} {
		ci, gi, gf, go_ := Tanh(x), Logistic(x), Logistic(x), Logistic(x)
		state = state*gf + ci*gi
		want := float64(float32(Tanh(state) * go_))
		out.ReadTimeStep(tt, got)
		if math.Abs(got[0]-want) > 1e-9 {
			t.Errorf("t=%d output = %v; want %v", tt, got[0], want)
		}
	}
}

// The state is clipped to +/-100 every timestep, and the clipped value carries
// forward. A huge constant input drives it to the clip and pins it there.
func TestLSTMClipsTheStateEveryTimestep(t *testing.T) {
	g := gates(t,
		[4]float64{100, 100, 100, 100},
		[4]float64{0, 0, 0, 0},
		[4]float64{0, 0, 0, 0})
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 1, Width: 400}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{1})
	}
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	// Tanh(100) saturates to exactly 1 and GO saturates to 1, so a clipped
	// state gives an output of exactly 1. Without the clip the state still
	// saturates the tanh, so instead assert the state directly via the
	// exported probe.
	got := make([]float64, 1)
	out.ReadTimeStep(399, got)
	if got[0] != 1 {
		t.Errorf("saturated output = %v; want exactly 1", got[0])
	}
	// LastState is the state as computed at the final timestep, BEFORE the
	// end-of-row reset. Reading it after the reset would always give 0 and the
	// assertion would be vacuous — the map is one row 400 wide, so the last
	// timestep IS the end of a row.
	if s := l.LastState()[0]; s != 100 {
		t.Errorf("final cell state = %v; want exactly 100 (kStateClip)", s)
	}
}

// Every row is an independent sequence: the state and output are zeroed at the
// end of each row, so two identical rows must produce identical outputs.
func TestLSTMResetsStateAtEndOfRow(t *testing.T) {
	g := gates(t,
		[4]float64{1, 1, 1, 1},
		[4]float64{0.5, 0.5, 0.5, 0.5},
		[4]float64{0, 0, 0, 0})
	l, err := NewLSTM("L", 1, 2, false, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 2, Width: 3}, 1)
	for y := range 2 {
		for x := range 3 {
			in.WriteTimeStep(in.Map.T(y, x), []float64{float64(x) * 0.3})
		}
	}
	out, err := l.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	a := make([]float64, 1)
	b := make([]float64, 1)
	for x := range 3 {
		out.ReadTimeStep(out.Map.T(0, x), a)
		out.ReadTimeStep(out.Map.T(1, x), b)
		if a[0] != b[0] {
			t.Fatalf("x=%d: row 0 = %v, row 1 = %v; the state was not reset between rows", x, a[0], b[0])
		}
	}
}

// SummLSTM emits only the last h of each row, into a width-1 map.
func TestLSTMSummaryEmitsOnlyTheLastStep(t *testing.T) {
	g := gates(t,
		[4]float64{1, 1, 1, 1},
		[4]float64{0, 0, 0, 0},
		[4]float64{0, 0, 0, 0})
	plain, _ := NewLSTM("L", 1, 2, false, g)
	summ, err := NewLSTM("Lfys", 1, 2, true, g)
	if err != nil {
		t.Fatalf("NewLSTM() error = %v", err)
	}
	in := NewTensor(StrideMap{Height: 2, Width: 4}, 1)
	for tt := range in.Map.Len() {
		in.WriteTimeStep(tt, []float64{0.1 * float64(tt%4+1)})
	}
	full, err := plain.Forward(in)
	if err != nil {
		t.Fatalf("plain Forward() error = %v", err)
	}
	got, err := summ.Forward(in)
	if err != nil {
		t.Fatalf("summary Forward() error = %v", err)
	}
	if got.Map != (StrideMap{Height: 2, Width: 1}) {
		t.Fatalf("summary map = %v; want {2 1}", got.Map)
	}
	a := make([]float64, 1)
	b := make([]float64, 1)
	for y := range 2 {
		full.ReadTimeStep(full.Map.T(y, 3), a)
		got.ReadTimeStep(got.Map.T(y, 0), b)
		if a[0] != b[0] {
			t.Errorf("row %d summary = %v; want the plain LSTM's last step %v", y, b[0], a[0])
		}
	}
}

// gatesNA builds four 1x(na+1) matrices of zeros, so a construction-time shape
// rejection can be provoked at an arbitrary na. Using `gates` here would not
// work: its matrices are always 1x3 (Inputs == 2), so NewLSTM's earlier
// "gate has N input columns, want na=M" check fires first and the shape branch
// under test is never reached.
func gatesNA(t *testing.T, na int) [4]*Matrix {
	t.Helper()
	var out [4]*Matrix
	for g := range out {
		m, err := NewMatrix(1, na, make([]float64, na+1))
		if err != nil {
			t.Fatalf("NewMatrix() error = %v", err)
		}
		out[g] = m
	}
	return out
}

func TestNewLSTMRejectsUnsupportedShapes(t *testing.T) {
	// ns is gate CI's output count, 1 here. For ni=1 a 1-D layer has
	// na == ni+ns == 2; na == ni+2*ns == 3 is the 2-D case.
	if _, err := NewLSTM("L", 1, 3, false, gatesNA(t, 3)); err == nil {
		t.Fatal("NewLSTM with na = ni + 2*ns: want a 2-D unsupported error, got nil")
	} else if !strings.Contains(err.Error(), "2-D") {
		t.Errorf("NewLSTM with na = ni + 2*ns: error = %q; want it to name the 2-D case", err)
	}
	// Anything that is neither ni+ns nor ni+2*ns is softmax feedback.
	if _, err := NewLSTM("L", 1, 5, false, gatesNA(t, 5)); err == nil {
		t.Fatal("NewLSTM with na = 5 (ni=1, ns=1): want a softmax-feedback error, got nil")
	}
	// And the column-count guard still fires when the gates disagree with na.
	if _, err := NewLSTM("L", 1, 2, false, gatesNA(t, 3)); err == nil {
		t.Fatal("NewLSTM with gates wider than na: want an error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/nn/ -run TestLSTM -v`
Expected: FAIL — `undefined: NewLSTM`.

- [ ] **Step 3: Implement**

Create `internal/nn/lstm.go`:

```go
// This file is a Go translation of src/lstm/lstm.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package nn

import "fmt"

// Gate indices, from LSTM::WeightType in src/lstm/lstm.h. GFS, the fifth gate,
// exists only for 2-D LSTMs and is rejected at construction.
const (
	GateCI  = iota // cell input,  activated with Tanh
	GateGI         // input gate,  activated with Logistic
	GateGF1        // forget gate, activated with Logistic
	GateGO         // output gate, activated with Logistic
	numGates
)

// stateClip is kStateClip from src/lstm/lstm.cpp.
const stateClip = 100.0

// LSTM is NT_LSTM and, with Summary set, NT_LSTM_SUMMARY.
//
// NI is the layer input count, NS the internal state count (= NumOutputs), and
// NA the padded gate-input width, so every gate matrix is NS x (NA+1). The
// column layout of a gate row is
//
//	[0, NI)      the layer input at this timestep
//	[NI, NA)     the recurrent output from the previous timestep
//	[NA]         the bias, against an implicit 1.0
//
// Tesseract also supports a softmax-feedback block between those two and a
// second recurrent block for the 2-D case; neither occurs in a tessdata_best
// Latin model and both are rejected by NewLSTM.
type LSTM struct {
	name    string
	NI      int
	NS      int
	NA      int
	Summary bool
	Gates   [numGates]*Matrix

	// lastState is the cell state as computed at each row's final timestep,
	// captured BEFORE the end-of-row reset zeroes it. Kept only so tests can
	// assert the kStateClip behaviour; it is not part of the forward
	// computation. Reading `state` after the reset would always be zero.
	lastState []float64
}

func NewLSTM(name string, ni, na int, summary bool, g [numGates]*Matrix) (*LSTM, error) {
	for i, m := range g {
		if m == nil {
			return nil, fmt.Errorf("nn: lstm %q gate %d is nil", name, i)
		}
		if m.Inputs != na {
			return nil, fmt.Errorf("nn: lstm %q gate %d has %d input columns, want na=%d", name, i, m.Inputs, na)
		}
		if m.Outputs != g[GateCI].Outputs {
			return nil, fmt.Errorf("nn: lstm %q gate %d has %d outputs, want %d", name, i, m.Outputs, g[GateCI].Outputs)
		}
	}
	ns := g[GateCI].Outputs
	// LSTM::DeSerialize derives is_2d_ as na_ - nf_ == ni_ + 2*ns_. With no
	// softmax feedback nf_ is zero, so a 1-D layer has na == ni + ns exactly.
	if na == ni+2*ns {
		return nil, fmt.Errorf("nn: lstm %q is 2-D (na=%d, ni=%d, ns=%d); 2-D LSTMs and their GFS gate are out of scope for L1b", name, na, ni, ns)
	}
	if na != ni+ns {
		return nil, fmt.Errorf("nn: lstm %q has na=%d but ni+ns=%d; softmax-feedback LSTMs are out of scope for L1b", name, na, ni+ns)
	}
	return &LSTM{name: name, NI: ni, NS: ns, NA: na, Summary: summary, Gates: g}, nil
}

func (l *LSTM) Name() string    { return l.name }
func (l *LSTM) NumOutputs() int { return l.NS }

// LastState exposes the cell state as computed at the last timestep of the last
// row, captured before that row's reset, so tests can assert the kStateClip
// behaviour. It is not used by the forward pass.
func (l *LSTM) LastState() []float64 { return l.lastState }

func (l *LSTM) Forward(in *Tensor) (*Tensor, error) {
	if in.Features != l.NI {
		return nil, fmt.Errorf("nn: lstm %q got %d input features, want %d", l.name, in.Features, l.NI)
	}
	m := in.Map
	outMap := m
	if l.Summary {
		outMap = m.ReduceWidthTo1()
	}
	out := NewTensor(outMap, l.NS)

	// source_ is a NetworkIO, so the recurrent output is narrowed to float32
	// before it is read back into the matvec. That rounding is inside the
	// recurrence and must not be optimised away.
	source := NewTensor(m, l.NA)

	u := make([]float64, l.NA)
	var gate [numGates][]float64
	for i := range gate {
		gate[i] = make([]float64, l.NS)
	}
	state := make([]float64, l.NS)
	output := make([]float64, l.NS)
	l.lastState = make([]float64, l.NS)

	destT := 0
	for y := range m.Height {
		for x := range m.Width {
			t := m.T(y, x)
			source.CopyTimeStepPart(t, 0, l.NI, in, t, 0)
			source.WriteTimeStepPart(t, l.NI, l.NS, output)
			source.ReadTimeStep(t, u)

			l.Gates[GateCI].DotVector(u, gate[GateCI])
			TanhInPlace(gate[GateCI])
			for _, g := range [3]int{GateGI, GateGF1, GateGO} {
				l.Gates[g].DotVector(u, gate[g])
				LogisticInPlace(gate[g])
			}

			for i := range state {
				// Two statements in Tesseract: MultiplyVectorsInPlace then
				// MultiplyAccumulate. The explicit float64 conversions keep Go
				// from fusing either product into the addition.
				keep := float64(state[i] * gate[GateGF1][i])
				add := float64(gate[GateCI][i] * gate[GateGI][i])
				s := keep + add
				if s < -stateClip {
					s = -stateClip
				} else if s > stateClip {
					s = stateClip
				}
				state[i] = s
				output[i] = Tanh(s) * gate[GateGO][i]
			}

			if l.Summary {
				if x == m.Width-1 {
					out.WriteTimeStep(destT, output)
					destT++
				}
			} else {
				out.WriteTimeStep(t, output)
			}

			if x == m.Width-1 {
				// Capture before zeroing: every row is an independent sequence,
				// so the state at the row's last timestep is the only place the
				// kStateClip behaviour is observable.
				copy(l.lastState, state)
				for i := range state {
					state[i] = 0
					output[i] = 0
				}
			}
		}
	}
	return out, nil
}
```

**Note on the state/output loop:** Tesseract computes `h_t` in a separate pass
(`FuncMultiply<HFunc>`) after clipping the whole state vector. Fusing the two
loops as above is equivalent because the clip and the output are both elementwise
and neither reads another element. If Task 14's diff ever implicates this layer,
split them back apart before looking anywhere else.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/nn/ -v`
Expected: PASS, five LSTM tests.

- [ ] **Step 5: Commit**

```bash
git add internal/nn/lstm.go internal/nn/lstm_test.go
git commit -m "feat(nn): add the LSTM cell"
```

---

## Task 11: Build an `nn` graph from a parsed `tessdata` layer tree

The bridge. Every shape assertion this task makes is free validation that the
loader and the runtime agree, and each one turns a class of silent
misalignment into a loud error at load time.

**Files:**
- Create: `internal/recog/build.go`
- Test: `internal/recog/build_test.go`

**Interfaces:**
- Consumes: `tessdata.Recognizer`, `tessdata.Layer`, everything in `internal/nn`.
- Produces:

```go
// Network is a loaded, runnable recognizer.
type Network struct {
	Root        nn.Layer
	InputHeight int      // the Input layer's StaticShape height, 36 for eng
	NumOutputs  int      // the softmax output count, 111 for eng
	NullChar    int      // the CTC blank, from the LSTM trailer
	XScale      int      // the network's x reduction factor, 3 for eng
	Rand        *nn.Rand // seeded from sample_iteration; shared with Normalize
}

func Build(rec *tessdata.Recognizer) (*Network, error)
```

`Rand` is exported because `Normalize` needs the *same* generator instance:
`Copy2DImage` draws from it before `Convolve::Forward` does. Handing `Convolve`
a private copy would reproduce Tesseract only when the scaled height happens to
be exact.

- [ ] **Step 1: Write the failing test**

Create `internal/recog/build_test.go`:

```go
package recog

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func loadModel(t *testing.T) *tessdata.Recognizer {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		t.Fatal("no lstm component")
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	return rec
}

func TestBuildRealModel(t *testing.T) {
	net, err := Build(loadModel(t))
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	if net.InputHeight != 36 {
		t.Errorf("InputHeight = %d; want 36", net.InputHeight)
	}
	if net.NumOutputs != 111 {
		t.Errorf("NumOutputs = %d; want 111", net.NumOutputs)
	}
	if net.NullChar != 110 {
		t.Errorf("NullChar = %d; want 110 (NumOutputs-1)", net.NullChar)
	}
	if net.XScale != 3 {
		t.Errorf("XScale = %d; want 3 (Mp3,3)", net.XScale)
	}
	if net.Root.NumOutputs() != 111 {
		t.Errorf("root NumOutputs() = %d; want 111", net.Root.NumOutputs())
	}
}

// A forward pass over a synthetic input must produce one probability
// distribution per output timestep, of the right width, in the right map.
func TestBuildForwardShapes(t *testing.T) {
	net, err := Build(loadModel(t))
	if err != nil {
		t.Fatalf("Build() error = %v", err)
	}
	const width = 120
	in := nn.NewTensor(nn.StrideMap{Height: 36, Width: width}, 1)
	out, err := net.Root.Forward(in)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	// Maxpool 3,3 reduces 36x120 to 12x40; XYTranspose+SummLSTM collapses the
	// 12 rows to 1; so the output is a 1 x 40 sequence of 111-wide softmax rows.
	if out.Map != (nn.StrideMap{Height: 1, Width: width / 3}) {
		t.Fatalf("output map = %v; want {1 %d}", out.Map, width/3)
	}
	if out.Features != 111 {
		t.Fatalf("output features = %d; want 111", out.Features)
	}
	row := make([]float64, 111)
	for tt := range out.Map.Len() {
		out.ReadTimeStep(tt, row)
		var sum float64
		for _, p := range row {
			sum += p
		}
		if sum < 0.99 || sum > 1.01 {
			t.Fatalf("t=%d: softmax row sums to %v; want ~1", tt, sum)
		}
	}
}

func TestBuildRejectsAnInt8Model(t *testing.T) {
	rec := loadModel(t)
	rec.TrainingFlags |= 1 // TF_INT_MODE
	if _, err := Build(rec); err == nil {
		t.Fatal("Build with TF_INT_MODE set: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/recog/ -v`
Expected: FAIL — `undefined: Build`.

- [ ] **Step 3: Implement**

Create `internal/recog/build.go`:

```go
// This file is a Go translation of the Network::CreateFromFile dispatch in
// src/lstm/network.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// tfIntMode is TF_INT_MODE from src/lstm/lstmrecognizer.h.
const tfIntMode = 1

// Network is a loaded, runnable recognizer graph plus the scalars the decoder
// needs out of the LSTM component's trailer.
type Network struct {
	Root        nn.Layer
	InputHeight int
	NumOutputs  int
	NullChar    int
	XScale      int
	// Rand is the recognizer's TRand. Normalize draws from it first (for rows
	// the scaler left short) and every Convolve continues from there, exactly
	// as Copy2DImage precedes Convolve::Forward in Tesseract.
	Rand *nn.Rand
}

// Build converts a parsed layer tree into a runnable nn graph.
func Build(rec *tessdata.Recognizer) (*Network, error) {
	if rec.TrainingFlags&tfIntMode != 0 {
		return nil, fmt.Errorf("recog: model has TF_INT_MODE set (training_flags=%d); L1b supports float weights from tessdata_best only", rec.TrainingFlags)
	}
	rnd := nn.NewRand(uint64(int64(rec.SampleIteration) * 0x10000001))

	root, err := buildLayer(rec.Root, rnd)
	if err != nil {
		return nil, err
	}

	shape := findInputShape(rec.Root)
	if shape == nil {
		return nil, fmt.Errorf("recog: model has no Input layer")
	}
	if shape.Height <= 0 {
		return nil, fmt.Errorf("recog: input shape height is %d; a fixed height is required", shape.Height)
	}
	if shape.Depth != 1 {
		return nil, fmt.Errorf("recog: input depth %d; only 1-channel grey input is supported", shape.Depth)
	}

	n := &Network{
		Root:        root,
		InputHeight: shape.Height,
		NumOutputs:  root.NumOutputs(),
		// tessdata.Recognizer.NullChar is int32 (L1a Task 4).
		NullChar: int(rec.NullChar),
		XScale:   xScaleFactor(rec.Root),
		Rand:     rnd,
	}
	// null_char_ is the authority for the CTC blank; it is structurally free to
	// differ from NumOutputs-1, so it is read rather than assumed. It must
	// nonetheless be a valid output index.
	if n.NullChar < 0 || n.NullChar >= n.NumOutputs {
		return nil, fmt.Errorf("recog: null_char %d is outside the %d network outputs", n.NullChar, n.NumOutputs)
	}
	return n, nil
}

func buildLayer(l *tessdata.Layer, rnd *nn.Rand) (nn.Layer, error) {
	switch l.Type {
	case tessdata.LayerInput:
		return nn.NewInput(l.Name, l.NumOutputs), nil

	case tessdata.LayerSeries, tessdata.LayerParallel, tessdata.LayerReplicated:
		stack, err := buildStack(l, rnd)
		if err != nil {
			return nil, err
		}
		if l.Type == tessdata.LayerSeries {
			return nn.NewSeries(l.Name, stack)
		}
		return nn.NewParallel(l.Name, stack)

	case tessdata.LayerXReversed, tessdata.LayerYReversed, tessdata.LayerXYTranspose:
		stack, err := buildStack(l, rnd)
		if err != nil {
			return nil, err
		}
		if len(stack) != 1 {
			return nil, fmt.Errorf("recog: %v %q has %d children; want exactly 1", l.Type, l.Name, len(stack))
		}
		kind := map[tessdata.LayerType]nn.ReversalKind{
			tessdata.LayerXReversed:   nn.ReverseX,
			tessdata.LayerYReversed:   nn.ReverseY,
			tessdata.LayerXYTranspose: nn.TransposeXY,
		}[l.Type]
		return nn.NewReversed(l.Name, kind, stack[0]), nil

	case tessdata.LayerConvolve:
		return nn.NewConvolve(l.Name, l.NumInputs, l.HalfX, l.HalfY, rnd), nil

	case tessdata.LayerMaxpool:
		return nn.NewMaxpool(l.Name, l.NumInputs, l.XScale, l.YScale), nil

	case tessdata.LayerReconfig:
		return nn.NewReconfig(l.Name, l.NumInputs, l.XScale, l.YScale), nil

	case tessdata.LayerSoftmax, tessdata.LayerSoftmaxNoCTC, tessdata.LayerTanh,
		tessdata.LayerRelu, tessdata.LayerLinear, tessdata.LayerLogistic,
		tessdata.LayerPosClip, tessdata.LayerSymClip:
		if len(l.Matrices) != 1 {
			return nil, fmt.Errorf("recog: %v %q has %d weight matrices; want 1", l.Type, l.Name, len(l.Matrices))
		}
		m, err := convertMatrix(l.Name, l.Matrices[0])
		if err != nil {
			return nil, err
		}
		// FullyConnected's matrix is no x (ni+1). Both dimensions are checked
		// because a mismatch means the graph parse or the weight load slipped.
		if m.Outputs != l.NumOutputs || m.Inputs != l.NumInputs {
			return nil, fmt.Errorf("recog: %v %q matrix is %dx%d; want %dx%d from the layer header",
				l.Type, l.Name, m.Outputs, m.Inputs+1, l.NumOutputs, l.NumInputs+1)
		}
		act := map[tessdata.LayerType]nn.Activation{
			tessdata.LayerSoftmax:      nn.ActSoftmax,
			tessdata.LayerSoftmaxNoCTC: nn.ActSoftmax,
			tessdata.LayerTanh:         nn.ActTanh,
			tessdata.LayerRelu:         nn.ActRelu,
			tessdata.LayerLinear:       nn.ActLinear,
			tessdata.LayerLogistic:     nn.ActLogistic,
			tessdata.LayerPosClip:      nn.ActPosClip,
			tessdata.LayerSymClip:      nn.ActSymClip,
		}[l.Type]
		return nn.NewFullyConnected(l.Name, act, m), nil

	case tessdata.LayerLSTM, tessdata.LayerLSTMSummary:
		if len(l.Matrices) != 4 {
			return nil, fmt.Errorf("recog: LSTM %q has %d gate matrices; want 4 (a 5th means a 2-D layer, which is out of scope)", l.Name, len(l.Matrices))
		}
		var gates [4]*nn.Matrix
		for i, src := range l.Matrices {
			m, err := convertMatrix(l.Name, src)
			if err != nil {
				return nil, err
			}
			if m.Inputs != l.NA {
				return nil, fmt.Errorf("recog: LSTM %q gate %d has %d input columns; want na=%d", l.Name, i, m.Inputs, l.NA)
			}
			gates[i] = m
		}
		return nn.NewLSTM(l.Name, l.NumInputs, l.NA, l.Type == tessdata.LayerLSTMSummary, gates)

	default:
		return nil, fmt.Errorf("recog: layer type %v (%q) is not supported by the L1b runtime", l.Type, l.Name)
	}
}

func buildStack(l *tessdata.Layer, rnd *nn.Rand) ([]nn.Layer, error) {
	stack := make([]nn.Layer, 0, len(l.Children))
	for i, c := range l.Children {
		sub, err := buildLayer(c, rnd)
		if err != nil {
			return nil, fmt.Errorf("%v %q child %d: %w", l.Type, l.Name, i, err)
		}
		stack = append(stack, sub)
	}
	return stack, nil
}

// convertMatrix reshapes a loader matrix into a runtime one. The loader's Cols
// counts the bias column; the runtime's Inputs does not. The loader's field is
// named Values (L1a Task 3), not W.
func convertMatrix(name string, m tessdata.Matrix) (*nn.Matrix, error) {
	if m.Cols < 1 {
		return nil, fmt.Errorf("recog: layer %q has a matrix with %d columns", name, m.Cols)
	}
	return nn.NewMatrix(m.Rows, m.Cols-1, m.Values)
}

func findInputShape(l *tessdata.Layer) *tessdata.InputShape {
	if l.Shape != nil {
		return l.Shape
	}
	for _, c := range l.Children {
		if s := findInputShape(c); s != nil {
			return s
		}
	}
	return nil
}

// xScaleFactor is Network::XScaleFactor, and the Series case is the ONLY one
// that multiplies.
//
//	src/lstm/network.h:211   Network::XScaleFactor()  -> 1
//	src/lstm/reconfig.cpp    Reconfig::XScaleFactor() -> x_scale_ (Maxpool derives)
//	src/lstm/plumbing.cpp    Plumbing::XScaleFactor() -> stack_[0]->XScaleFactor()
//	src/lstm/series.cpp:91   Series::XScaleFactor()   -> product over the stack
//
// Parallel, Replicated and the three Reversed variants inherit Plumbing's
// version, which takes the FIRST child only. Treating them as products is
// latent for eng — every plumbing node there has exactly one child, so the
// product equals stack[0] — but it is wrong for any Parallel with two or more
// children, and Task 1 Step 2 tells the reader to fall back on this function.
func xScaleFactor(l *tessdata.Layer) int {
	switch l.Type {
	case tessdata.LayerMaxpool, tessdata.LayerReconfig:
		return l.XScale

	case tessdata.LayerSeries:
		f := 1
		for _, c := range l.Children {
			f *= xScaleFactor(c)
		}
		return f

	case tessdata.LayerParallel, tessdata.LayerReplicated, tessdata.LayerParRLLSTM,
		tessdata.LayerParUDLSTM, tessdata.LayerPar2DLSTM,
		tessdata.LayerXReversed, tessdata.LayerYReversed, tessdata.LayerXYTranspose:
		if len(l.Children) == 0 {
			return 1
		}
		return xScaleFactor(l.Children[0])

	default:
		return 1
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/recog/ -v`
Expected: PASS. `TestBuildForwardShapes` takes a few seconds — the `Lfx512`
layer is 1.2M weights and the graph runs unoptimized.

- [ ] **Step 5: Cross-check `num_weights` per layer**

Each layer header carries `num_weights`, which equals the summed element count
of that subtree's matrices **including bias columns**. That is free validation
that the weight load is aligned. Add it as an assertion inside `Build` rather
than as a one-off check:

```go
// in buildLayer, after constructing a FullyConnected or LSTM:
	total := 0
	for _, m := range l.Matrices {
		total += m.Rows * m.Cols
	}
	if total != l.NumWeights {
		return nil, fmt.Errorf("recog: layer %q matrices hold %d weights but the header says %d", l.Name, total, l.NumWeights)
	}
```

Verify the arithmetic by hand once against the dump: `ConvNL` 160 = 16x10,
`Lfx96` 61824 = 4x96x161, `Output` 56943 = 111x513.

- [ ] **Step 6: Commit**

```bash
git add internal/recog/build.go internal/recog/build_test.go
git commit -m "feat(recog): build a runnable graph from a parsed layer tree"
```

---

## Task 12: Leptonica-exact 8bpp grayscale scaling

**This is the largest single fidelity risk in the whole pipeline, and it is
bigger than the activation tables, because it perturbs the *input* rather than
the sixth decimal of an intermediate.**

`ImageData::PreScale` scales the line crop isotropically so its height becomes
exactly `network->NumInputs()` (36 for eng) using Leptonica's `pixScale`. Text
line crops are almost always taller than 36 px, so the factor is below 1 and,
per the research, `pixScaleGeneral` dispatches 8bpp work as:

| factor | routine |
|---|---|
| >= 0.7 | `pixScaleGrayLI` (linear interpolation) |
| 0.02 - 0.7 | `pixScaleAreaMap` |
| < 0.02 | `pixScaleSmooth` |

**That dispatch table was read from Leptonica's `master` on GitHub, not from the
1.87.0 that is installed here. It is unverified.** Do not implement from it.

- [ ] **Step 1: Read the real dispatch, from the installed version**

```bash
mkdir -p /tmp/lept && cd /tmp/lept
brew info leptonica | grep -i 'stable\|1\.87'
curl -fsSL -o leptonica-1.87.0.tar.gz \
  https://github.com/DanBloomberg/leptonica/releases/download/1.87.0/leptonica-1.87.0.tar.gz
tar xf leptonica-1.87.0.tar.gz
sed -n '/pixScaleGeneral/,/^}/p' leptonica-1.87.0/src/scale1.c
```

Record, in a comment at the top of `internal/imaging/scalegray.go`:

1. The exact identity short-circuit — does `pixScale(pix, 1.0, 1.0)` return an
   unmodified copy? Task 17's 36-px-tall corpus depends on it.
2. The exact 8bpp dispatch thresholds and the routines they select.
3. The bodies of `scaleGrayAreaMapLow` and `scaleGrayLILow` from
   `src/scalelow.c`.

**If the thresholds differ from the table above, the table above is wrong;
implement what 1.87.0 does and note the difference in the task report.**

- [ ] **Step 2: Extend the golden generator**

The L0 harness at `testdata/golden/gen/gen.c` already links against the
installed Leptonica and dumps raw bitmaps. Add scaling goldens to it:

```c
// Appended to gen.c's main(), after the existing dumps.
{
    // Scaling goldens for internal/imaging/scalegray.go. Three factors, one on
    // each side of the documented 0.7 dispatch threshold plus the identity.
    static const struct { const char *name; float f; } cases[] = {
        { "scale_identity", 1.0f },
        { "scale_li",       0.80f },
        { "scale_areamap",  0.35f },
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        PIX *s = pixScale(gray, cases[i].f, cases[i].f);
        char p[512];
        snprintf(p, sizeof p, "%s/%s.bin", argv[2], cases[i].name);
        dump(p, s);
        pixDestroy(&s);
    }
}
```

```bash
make goldens
ls -l testdata/golden/scale_*.bin
```

- [ ] **Step 3: Write the failing test**

Create `internal/imaging/scalegray_test.go`:

```go
package imaging

import "testing"

func TestScaleGrayMatchesLeptonica(t *testing.T) {
	src := loadGolden(t, "gray.bin")
	for _, tc := range []struct {
		name   string
		factor float64
	}{
		{"scale_identity", 1.0},
		{"scale_li", 0.80},
		{"scale_areamap", 0.35},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := loadGolden(t, tc.name+".bin")
			got := ScaleGray(src, tc.factor)
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `go test ./internal/imaging/ -run TestScaleGray -v`
Expected: FAIL — `undefined: ScaleGray`.

- [ ] **Step 5: Implement**

Create `internal/imaging/scalegray.go` implementing `ScaleGray(src *Bitmap,
factor float64) *Bitmap` for depth-8 bitmaps. **This step is a transcription, not
a design**: the source is version-pinned and read in Step 1, and inventing an
interpolation kernel instead would be strictly worse. What must come out of it,
in this order:

1. `func ScaleGray(src *Bitmap, factor float64) *Bitmap` — the entry point.
2. The identity short-circuit, spelled exactly as `pixScaleGeneral` spells it
   (Step 1 item 1 records whether that is `scalex == 1.0 && scaley == 1.0` or a
   tolerance).
3. The 8bpp dispatch, with the thresholds Step 1 item 2 recorded, selecting
   between the two routines below. **If 1.87.0 dispatches to a third routine at
   some factor the corpus reaches, transcribe that one too rather than clamping
   into a neighbour.**
4. `scaleGrayLILow` — the linear-interpolation kernel.
5. `scaleGrayAreaMapLow` — the area-map kernel.
6. The output-dimension computation, taken from Leptonica rather than derived:
   `ImageData::PreScale` reads the *actual* `pixGetWidth`/`pixGetHeight` of the
   result, so Cadmus must return real dimensions rather than
   `round(factor*src.Height)`.

Carry the Leptonica attribution header (Leptonica is BSD-2-Clause, not
Apache-2.0 — add a matching notice to `NOTICE` and name the source files).

- [ ] **Step 6: Run the tests and decide**

Run: `go test ./internal/imaging/ -run TestScaleGray -v`

**Acceptance, in order:**

1. **All three exact (`diff == 0`).** Ideal. Continue.
2. **`scale_identity` exact but the resampling cases differ.** Acceptable *for
   now*. Record the measured `diff` percentage and `maxDelta` in the bead, and
   proceed — Task 17's corpus has a 36-px-tall arm that bypasses scaling
   entirely, so Stage 1 can still isolate a forward-pass bug from a scaler bug.
   Re-open this once Stage 1 passes on the 36-px arm and fails on the native arm.
3. **`scale_identity` differs.** Stop. Either `pixScale` does resample at 1.0 —
   in which case Task 17's design assumption is wrong and the corpus must be
   built differently — or the golden loader is misreading the dump. Resolve
   before continuing; nothing downstream is trustworthy otherwise.

```bash
bd update cad-l1b --append-notes "ScaleGray vs Leptonica 1.87.0: identity <exact/not>, LI <n>% differ (max <d> levels), areamap <n>% differ (max <d> levels)."
```

- [ ] **Step 7: Commit**

```bash
git add internal/imaging/scalegray.go internal/imaging/scalegray_test.go testdata/golden/gen/gen.c testdata/golden/scale_*.bin NOTICE
git commit -m "feat(imaging): add Leptonica-compatible 8bpp grayscale scaling"
```

---

## Task 13: Line-image normalization

The full chain from a cropped line image to the network's input tensor:

1. Convert to 8bpp grey.
2. `im_factor = 36 / height`; scale isotropically with `ScaleGray`. **No
   centring, no padding, no fixed width, no mean/standard-deviation
   normalization.** The output width is whatever falls out.
3. Compute a black point and a white point from **one scanline at `y = height/2`**
   of the scaled image: collect local minima into one 256-bucket histogram and
   local maxima into another, over `x` in `[1, width-2]`; `black = mins.ile(0.25)`,
   `white = maxes.ile(0.75)`. Empty histograms default to a single sample at `0`
   and a single sample at `255` respectively — **and `ile` then returns 1 and
   256, not 0 and 255**, because it interpolates past the crossing bucket. That
   is not a bug in the port; see Step 1's `TestNormalizeFlatImageUsesDefaults`.
4. `contrast = (white - black) / 2`, forced to 1 if it is not positive.
5. Per pixel, `value = (pixel - black)/contrast - 1`, computed in **float32**,
   and **not clipped** — pixels outside `[black, black+2*contrast]` land outside
   `[-1, 1]`. Polarity is not inverted: 0 (black ink) is the most negative.
6. The tensor map is `{Height: inputHeight, Width: scaledWidth}` with 1 feature.
   **The map height is the network's, not the scaled image's.**
   `NetworkIO::FromPixes` sets it from `shape.height()` (36, non-zero) and the
   width from `shape.width()` (0 → the actual pix width), and `Copy2DImage` then
   iterates `y < target_height`, filling any row past `pixGetHeight(pix)` with
   `Randomize` — `SignedRand(1.0)` per feature, drawn from the *same* LCG
   `Convolve` uses. `pixScale`'s rounding on an odd source height produces 35 or
   37 often enough that this matters, and swallowing it changes the LCG state
   `Convolve` inherits.

`STATS::ile` is an interpolated percentile over integer buckets, not a plain
quantile, and needs a faithful port.

**Files:**
- Create: `internal/recog/normalize.go`
- Test: `internal/recog/normalize_test.go`

**Interfaces:**
- Consumes: `imaging.Bitmap`, `imaging.ScaleGray`, `nn.Tensor`.
- Produces:

```go
// Normalized is a line image prepared for the network.
type Normalized struct {
	Input       *nn.Tensor
	ScaleFactor float64 // page pixels per network timestep: XScale / imFactor
	Black       float32
	Contrast    float32
}

// rnd is the recognizer's LCG. It is threaded in because Copy2DImage draws from
// it to fill any rows the scaler left short of inputHeight, and Convolve then
// continues from whatever state that leaves. Passing nil is a programming
// error, not a "no randomization" mode.
func Normalize(img image.Image, inputHeight, xScale int, rnd *nn.Rand) (*Normalized, error)
```

- [ ] **Step 1: Write the failing test**

Create `internal/recog/normalize_test.go`:

```go
package recog

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// testRand is the seed Normalize's callers use in these tests. The value is
// irrelevant to every assertion below except that it must be non-nil.
func testRand() *nn.Rand { return nn.NewRand(1) }

// ile is Tesseract's interpolated percentile. These expectations are computed
// by hand from src/ccstruct/statistc.cpp:172-197.
func TestILEInterpolatesAcrossBuckets(t *testing.T) {
	// Four samples: two at 10, two at 20. total=4.
	var s stats
	s.add(10, 2)
	s.add(20, 2)
	// frac 0.25 -> target = clip(1.0, 1, 4) = 1.0. The loop runs while
	// sum < target: index 10 consumes 2, sum=2, index=11, loop ends.
	// result = 0 + 11 - (2-1)/2 = 10.5
	if got := s.ile(0.25); math.Abs(got-10.5) > 1e-12 {
		t.Errorf("ile(0.25) = %v; want 10.5", got)
	}
	// frac 0.75 -> target = 3.0. index 10 gives sum=2 (<3), index 20 gives
	// sum=4, index=21. result = 0 + 21 - (4-3)/2 = 20.5
	if got := s.ile(0.75); math.Abs(got-20.5) > 1e-12 {
		t.Errorf("ile(0.75) = %v; want 20.5", got)
	}
	// An empty histogram returns rangemin.
	var empty stats
	if got := empty.ile(0.5); got != 0 {
		t.Errorf("empty ile(0.5) = %v; want 0", got)
	}
	// frac 0 clips the target up to 1, so it does not return rangemin.
	if got := s.ile(0.0); math.Abs(got-10.5) > 1e-12 {
		t.Errorf("ile(0.0) = %v; want 10.5 (target clips to 1)", got)
	}
}

// A flat image has no local minima or maxima at all, so the defaults kick in:
// a single sample at 0 in `mins` and a single sample at 255 in `maxes`.
//
// ile does NOT then return 0 and 255. With total_count_ == 1 the target clips to
// 1.0, the loop consumes the one populated bucket and leaves `index` one PAST
// it, and the result is `rangemin + index - (sum-target)/buckets[index-1]`:
//
//	mins:  index lands at 1   -> 0 + 1   - 0/1 = 1    -> black = 1
//	maxes: index lands at 256 -> 0 + 256 - 0/1 = 256  -> white = 256
//
// contrast is therefore (256-1)/2 = 127.5, and a mid-grey pixel maps to a
// slightly NEGATIVE value, not a positive one. This was checked by running the
// port of `ile` in Step 5 against these histograms.
func TestNormalizeFlatImageUsesDefaults(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 40, 36))
	for i := range img.Pix {
		img.Pix[i] = 128
	}
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Black != 1 || math.Abs(float64(n.Contrast)-127.5) > 1e-6 {
		t.Fatalf("black=%v contrast=%v; want 1 and 127.5", n.Black, n.Contrast)
	}
	want := float64(float32((128.0-1.0)/127.5 - 1.0))
	if want >= 0 {
		t.Fatalf("expectation %v is not negative; the black point is not 1", want)
	}
	got := make([]float64, 1)
	n.Input.ReadTimeStep(0, got)
	if got[0] != want {
		t.Errorf("pixel value = %v; want %v", got[0], want)
	}
}

// ile's defaults, asserted directly, so a regression here is not mistaken for a
// Normalize bug.
func TestILEEmptyHistogramDefaults(t *testing.T) {
	var mins, maxes stats
	mins.add(0, 1)
	maxes.add(255, 1)
	if got := mins.ile(0.25); got != 1 {
		t.Errorf("mins.ile(0.25) with a single sample at 0 = %v; want 1", got)
	}
	if got := maxes.ile(0.75); got != 256 {
		t.Errorf("maxes.ile(0.75) with a single sample at 255 = %v; want 256", got)
	}
}

// Black ink is the most negative value and white paper the most positive; the
// polarity is not inverted.
func TestNormalizePolarity(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 40, 36))
	for y := range 36 {
		for x := range 40 {
			v := uint8(240)
			if x%4 == 0 {
				v = 10
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	dark := make([]float64, 1)
	light := make([]float64, 1)
	n.Input.ReadTimeStep(n.Input.Map.T(0, 0), dark)
	n.Input.ReadTimeStep(n.Input.Map.T(0, 1), light)
	if dark[0] >= light[0] {
		t.Errorf("dark pixel %v is not below light pixel %v; the polarity is inverted", dark[0], light[0])
	}
}

// A 36-pixel-tall image is not scaled, so the map width is the image width and
// one timestep is XScale page pixels.
func TestNormalizeAtNativeHeightDoesNotScale(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 97, 36))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Input.Map != (nn.StrideMap{Height: 36, Width: 97}) {
		t.Fatalf("map = %v; want 36x97", n.Input.Map)
	}
	if math.Abs(n.ScaleFactor-3) > 1e-9 {
		t.Errorf("ScaleFactor = %v; want 3 (XScale/1.0)", n.ScaleFactor)
	}
}

// A 72-pixel-tall image halves, so one timestep covers 6 page pixels.
func TestNormalizeScaleFactor(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 200, 72))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if math.Abs(n.ScaleFactor-6) > 1e-6 {
		t.Errorf("ScaleFactor = %v; want 6 (3 / (36/72))", n.ScaleFactor)
	}
}

// Tesseract does NOT require the scaled height to equal the network's input
// height. PreScale reports pixGetHeight(pix) as-is, PrepareLSTMInputs rejects
// only `width < min_width || height < min_width`, and Copy2DImage fills any
// missing rows with Randomize. A hard error here would reject real crops whose
// odd source height makes pixScale round to 35 or 37.
func TestNormalizeToleratesAnOffByOneScaledHeight(t *testing.T) {
	// 37 px tall: im_factor = 36/37, and Leptonica's rounding is free to land on
	// 35, 36 or 37. Whatever it lands on, the map must be 36 rows.
	img := image.NewGray(image.Rect(0, 0, 120, 37))
	n, err := Normalize(img, 36, 3, testRand())
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if n.Input.Map.Height != 36 {
		t.Errorf("map height = %d; want 36 (the network's input height, not the scaled image's)", n.Input.Map.Height)
	}
}

// The rejection Tesseract does make: both dimensions must reach min_width,
// which RecognizeLine passes as network->XScaleFactor().
func TestNormalizeRejectsImagesBelowTheMinimum(t *testing.T) {
	if _, err := Normalize(image.NewGray(image.Rect(0, 0, 2, 36)), 36, 3, testRand()); err == nil {
		t.Error("a 2-px-wide line: want an error, got nil")
	}
	if _, err := Normalize(image.NewGray(image.Rect(0, 0, 120, 1)), 36, 3, testRand()); err == nil {
		t.Error("a line that scales to under 3 px tall: want an error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/recog/ -run 'TestILE|TestNormalize' -v`
Expected: FAIL — `undefined: stats`.

- [ ] **Step 3: Implement**

Create `internal/recog/normalize.go`:

```go
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/recog/ -v`
Expected: PASS.

- [ ] **Step 5: Verify `ile` against the C++ once, by hand**

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
sed -n '/^double STATS::ile/,/^}/p' "$SP/tess/src/ccstruct/statistc.cpp"
```

The C++ loop is `for (index = 0; index <= rangemax_ - rangemin_ && sum < target; sum += buckets_[index++]);`
— the increment expression runs *after* the condition, so the Go loop body must
add the bucket and then advance, which is what the code above does. Confirm the
two agree on the boundary `index == 256`. **If they differ, the black and white
points shift and every activation in the network moves.**

- [ ] **Step 6: Commit**

```bash
git add internal/recog/normalize.go internal/recog/normalize_test.go
git commit -m "feat(recog): normalize a line image to the network input height"
```

---

## Task 14: Per-layer activation dumping and the differential protocol

**"The final text is wrong" is not a debuggable signal on a nine-layer network.**
This task builds the instrument that turns it into "layer 4 of 9 diverges", and
it is built *before* the decoder so that it is available the moment Stage 1
produces wrong text.

Tesseract has the dump built in but disabled: `src/lstm/functions.h:27` is
`#define DEBUG_DETAIL 0`, and raising it makes `FullyConnected::Forward` print
`F Output:<name>` followed by the whole activation array, and `LSTM::Forward`
print `Source:`, `State:` and `Output:` for its layer. Raising it also
`#undef _OPENMP`, which is what keeps the interleaving deterministic.

**Files:**
- Create: `cmd/cadmusdump/activations.go`, `cmd/cadmusdump/activations_test.go`
- Create: `testdata/tessdebug/debug-detail.patch`, `testdata/tessdebug/capture.sh`

**Interfaces:**
- Consumes: `recog.Build`, `recog.Normalize`, `nn.Layer`.
- Produces: `cadmusdump -activations <model> <line.png>` writing one block per
  layer to stdout, in the same shape as Tesseract's dump.

- [ ] **Step 1: Add the activation tap to the runtime**

Add to `internal/nn/layer.go`:

```go
// Tap, if non-nil, is called with every layer's output as the graph runs. It
// exists so a debugging harness can diff Cadmus's activations against
// Tesseract's layer by layer; production callers leave it nil.
type Tap func(layer Layer, out *Tensor)

// WithTap wraps a layer so that tap sees its output. Wrapping rather than
// threading a parameter through every Forward keeps the hot path free of a
// nil check per layer per timestep.
type WithTap struct {
	Sub Layer
	Fn  Tap
}

func (l *WithTap) Name() string    { return l.Sub.Name() }
func (l *WithTap) NumOutputs() int { return l.Sub.NumOutputs() }

func (l *WithTap) Forward(in *Tensor) (*Tensor, error) {
	out, err := l.Sub.Forward(in)
	if err != nil {
		return nil, err
	}
	l.Fn(l.Sub, out)
	return out, nil
}
```

And in `internal/recog/build.go`, an option to wrap every constructed layer:

```go
// BuildWithTap is Build, with every layer wrapped so fn observes its output.
func BuildWithTap(rec *tessdata.Recognizer, fn nn.Tap) (*Network, error) {
	n, err := Build(rec)
	if err != nil {
		return nil, err
	}
	n.Root = wrapTap(n.Root, fn)
	return n, nil
}

func wrapTap(l nn.Layer, fn nn.Tap) nn.Layer {
	switch v := l.(type) {
	case *nn.Series:
		for i := range v.Stack {
			v.Stack[i] = wrapTap(v.Stack[i], fn)
		}
	case *nn.Parallel:
		for i := range v.Stack {
			v.Stack[i] = wrapTap(v.Stack[i], fn)
		}
	case *nn.Reversed:
		v.Sub = wrapTap(v.Sub, fn)
	case *nn.Input:
		// The Input layer is the identity, and its Name() is "Input" — exactly
		// the key dumpActivations uses for the pre-network block. Wrapping it
		// would emit two "Output:Input" headers and break Step 6's block-by-
		// block diff protocol, which keys on the header line.
		return l
	}
	return &nn.WithTap{Sub: l, Fn: fn}
}
```

- [ ] **Step 2: Write the failing test**

Create `cmd/cadmusdump/activations_test.go`:

```go
package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestDumpActivations(t *testing.T) {
	model := filepath.Join("..", "..", "testdata", "eng.traineddata")
	line := filepath.Join("..", "..", "testdata", "lines", "h36", "0001.png")
	for _, p := range []string{model, line} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("fixture not present: %v", err)
		}
	}
	var b strings.Builder
	if err := dumpActivations(model, line, &b); err != nil {
		t.Fatalf("dumpActivations() error = %v", err)
	}
	out := b.String()
	// One block per layer, keyed on the serialized layer name, so the blocks
	// line up with Tesseract's own DEBUG_DETAIL output. "Normalized" is the
	// pre-network input; the Input LAYER is not tapped, because its Name() is
	// "Input" and two identical headers would break the diff protocol.
	for _, want := range []string{
		"Output:Normalized", "Output:Convolve", "Output:ConvNL", "Output:Maxpool",
		"Output:Lfys64", "Output:Lfx96", "Output:Lrx96", "Output:Lfx512",
		"Output:Output",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("activation dump missing block %q", want)
		}
	}
	if n := strings.Count(out, "Output:Input\n"); n != 0 {
		t.Errorf("dump contains %d \"Output:Input\" headers; want 0 — the Input layer must not be tapped", n)
	}
	// Every value line must parse as floats. Checking for stray letters cannot
	// work: %g never emits letters for a finite value, so the only thing such a
	// check could ever catch is NaN/Inf, which contain no lowercase letters
	// other than 'a' and 'n'. Parse instead.
	for i, ln := range strings.Split(out, "\n") {
		if ln == "" || strings.HasPrefix(ln, "Output:") {
			continue
		}
		for _, f := range strings.Fields(ln) {
			if _, err := strconv.ParseFloat(f, 64); err != nil {
				t.Fatalf("line %d: field %q is not a float: %v", i+1, f, err)
			}
		}
	}
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `go test ./cmd/cadmusdump/ -run TestDumpActivations -v`
Expected: FAIL — `undefined: dumpActivations` (or a skip if the corpus from
Task 17 does not exist yet; in that case build Task 17 Step 1 first, then
return here — the two tasks are order-independent).

- [ ] **Step 4: Implement**

Create `cmd/cadmusdump/activations.go`:

```go
package main

import (
	"fmt"
	"image"
	_ "image/png"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/recog"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// dumpActivations runs one line image through the network and writes every
// layer's output in the same shape as Tesseract's DEBUG_DETAIL dump: a header
// line "Output:<layer name>", then one line per feature holding that feature's
// value at every timestep, space separated.
//
// Matching that shape exactly is the point — it makes the two dumps diffable
// without a parser on either side.
func dumpActivations(modelPath, imagePath string, w io.Writer) error {
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("reading %s: %w", modelPath, err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		return fmt.Errorf("parsing container: %w", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		return fmt.Errorf("%s has no lstm component", modelPath)
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return fmt.Errorf("parsing recognizer: %w", err)
	}

	f, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("opening %s: %w", imagePath, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return fmt.Errorf("decoding %s: %w", imagePath, err)
	}

	var dumpErr error
	net, err := recog.BuildWithTap(rec, func(l nn.Layer, out *nn.Tensor) {
		if dumpErr != nil {
			return
		}
		dumpErr = writeBlock(w, l.Name(), out)
	})
	if err != nil {
		return fmt.Errorf("building network: %w", err)
	}

	// Normalize consumes the same randomizer Convolve does, and it must run
	// first, exactly as Copy2DImage runs before Convolve::Forward.
	norm, err := recog.Normalize(img, net.InputHeight, net.XScale, net.Rand)
	if err != nil {
		return fmt.Errorf("normalizing: %w", err)
	}
	// Keyed "Normalized", not "Input": the graph's Input layer is also named
	// "Input", and two blocks with the same header make Step 6's diff ambiguous.
	if err := writeBlock(w, "Normalized", norm.Input); err != nil {
		return err
	}
	if _, err := net.Root.Forward(norm.Input); err != nil {
		return fmt.Errorf("forward: %w", err)
	}
	return dumpErr
}

func writeBlock(w io.Writer, name string, x *nn.Tensor) error {
	if _, err := fmt.Fprintf(w, "Output:%s\n", name); err != nil {
		return err
	}
	for feat := range x.Features {
		for t := range x.Map.Len() {
			if _, err := fmt.Fprintf(w, " %g", x.Row(t)[feat]); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintln(w); err != nil {
			return err
		}
	}
	return nil
}
```

Wire it into `main`: `cadmusdump -activations <model> <image>` calls
`dumpActivations`; with no flag the existing `dump` runs unchanged.

- [ ] **Step 5: Build the instrumented Tesseract**

Create `testdata/tessdebug/debug-detail.patch`:

```diff
--- a/src/lstm/functions.h
+++ b/src/lstm/functions.h
@@
-#define DEBUG_DETAIL 0
+#define DEBUG_DETAIL 1
--- a/src/lstm/networkio.cpp
+++ b/src/lstm/networkio.cpp
@@ void NetworkIO::Print(int num) const {
   int num_features = NumFeatures();
   for (int y = 0; y < num_features; ++y) {
     for (int t = 0; t < Width(); ++t) {
-      if (num == 0 || t < num || t + num >= Width()) {
+      if (true) {
--- a/src/lstm/convolve.cpp
+++ b/src/lstm/convolve.cpp
@@ void Convolve::Forward
   } while (dest_index.Increment());
+  tprintf("Output:%s\n", name_.c_str());
+  output->Print(0);
--- a/src/lstm/maxpool.cpp
+++ b/src/lstm/maxpool.cpp
@@ void Maxpool::Forward
   } while (dest_index.Increment());
+  tprintf("Output:%s\n", name_.c_str());
+  output->Print(0);
```

Apply it by hand against the shallow clone — the hunks above are positional
guidance, not a machine-applicable patch, because line numbers move between
Tesseract releases. Then:

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
cd "$SP/tess"
git diff > /Users/christopherdobbyn/work/dobbo-ca/cadmus/testdata/tessdebug/debug-detail.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DDISABLE_ARCHIVE=ON -DDISABLE_CURL=ON \
  -DBUILD_TRAINING_TOOLS=OFF -DGRAPHICS_DISABLED=ON
cmake --build build -j8 --target tesseract
./build/tesseract --version
```

Commit the *generated* diff, not the sketch above, so the next person can
re-apply it exactly.

Create `testdata/tessdebug/capture.sh`:

```bash
#!/usr/bin/env bash
# Captures Tesseract's per-layer activation dump for one line image, using a
# locally built instrumented tesseract (see debug-detail.patch). Manual step;
# never run in CI.
set -euo pipefail
TESS_BIN="${TESS_BIN:?set TESS_BIN to the instrumented tesseract binary}"
TESSDATA="${TESSDATA:?set TESSDATA to the directory holding eng.traineddata}"
IMAGE="${1:?usage: capture.sh <line.png> [out.txt]}"
OUT="${2:-/dev/stdout}"

# --psm 13 (PSM_RAW_LINE) is the ONLY mode besides PSM_SINGLE_WORD that feeds the
# whole image to the LSTM; --psm 7 hands it a baseline-derived, padded per-word
# sub-crop instead (src/ccmain/linerec.cpp:229-250) and the two sides would not
# be looking at the same pixels.
#
# DOTPRODUCT=native forces the reference scalar dot product. Without it the
# arm64 build dispatches to DotProductNEON, whose accumulation order differs.
DOTPRODUCT=native \
  "$TESS_BIN" --tessdata-dir "$TESSDATA" --psm 13 --dpi 300 "$IMAGE" - 2>&1 >/dev/null | tee "$OUT"
```

- [ ] **Step 5b: Confirm what Tesseract actually feeds the network**

The whole `h36` design rests on Tesseract's LSTM input being byte-identical to
the committed PNG. That is a claim about `GetRectImage`, not about the PNG, and
the instrumented build can settle it in one line. Add to the patch, in
`Input::PrepareLSTMInputs` right after the `PreScale` call:

```cpp
tprintf("LSTMInput: %dx%d scale=%g\n", width, height, *image_scale);
```

then:

```bash
./testdata/tessdebug/capture.sh testdata/lines/h36/0001.png /tmp/act/tess.txt
grep LSTMInput /tmp/act/tess.txt
```

**Expected: `height == 36` and `scale == 1`, and `width` equal to the PNG's own
width.** If the height is not 36, `--psm 13` is not doing what this plan claims
and every "byte-identical pixels" statement in Tasks 17 and 18 is void — stop,
record the measured numbers in the bead, and re-derive the corpus design before
writing a decoder against it.

```bash
chmod +x testdata/tessdebug/capture.sh
```

- [ ] **Step 6: Run the first differential comparison**

```bash
export TESS_BIN="$SP/tess/build/tesseract" TESSDATA=testdata
mkdir -p /tmp/act
./testdata/tessdebug/capture.sh testdata/lines/h36/0001.png /tmp/act/tess.txt
go run ./cmd/cadmusdump -activations testdata/eng.traineddata testdata/lines/h36/0001.png > /tmp/act/cadmus.txt
grep -c '^Output:' /tmp/act/tess.txt /tmp/act/cadmus.txt
```

Then compare block by block. A one-off comparison is fine — it is not shipped and
does not need a test. `sed '$d'` drops the trailing `Output:` header that the
range match pulls in; **do not use `head -n -1`, which BSD/macOS `head` rejects
with `illegal line count`.**

```bash
block() {   # block <file> <layer name>
  sed -n "/^Output:$2\$/,/^Output:/p" "$1" | sed -e '1d' -e '$d'
}
block /tmp/act/tess.txt   ConvNL > /tmp/act/a
block /tmp/act/cadmus.txt ConvNL > /tmp/act/b
wc -l /tmp/act/a /tmp/act/b   # must match; if not, the block shapes differ
paste /tmp/act/a /tmp/act/b | awk '{ for (i=1;i<=NF/2;i++) { d=$i-$(i+NF/2); if (d<0) d=-d; if (d>m) m=d } } END { print "max abs delta:", m }'
```

**The debugging protocol, and this is the whole point of the task:**

1. Compare blocks **in graph order**: `Normalized`, `Convolve`, `ConvNL`,
   `Maxpool`, `Lfys64`, `Lfx96`, `Lrx96`, `Lfx512`, `Output`. (Tesseract's own
   first block is named `Input`; that is the same tensor. Cadmus renames it so
   the graph's `Input` layer, which is the identity, does not emit a second
   block with the same header.)
2. Read the max-abs-delta *profile*, not any single number. A healthy profile
   grows slowly and monotonically: roughly 0 at `Normalized` (identical pixels on
   the h36 corpus), ~1e-7 after `ConvNL` (the float32 store), and no more than
   one order of magnitude per LSTM.
3. **The first block whose delta jumps by more than two orders of magnitude
   above its predecessor is the broken layer.** Stop there; do not look at the
   text.
4. Localize within that layer by shrinking the input: a 36 x 9 crop gives a
   3-timestep sequence you can compute by hand.
5. Known-good expectations to check first when a layer is implicated:
   `Normalized` delta above 0 means normalization or scaling, not the runtime —
   go back to Tasks 12 and 13, and check Step 5b's `LSTMInput` height first,
   because a scaled height other than 36 makes `Normalize` draw from the LCG and
   shifts every subsequent `Convolve` draw.
   `Convolve` delta concentrated on x=0, x=W-1,
   y=0 and y=35 means the LCG draw order (Task 7 Step 5). An LSTM delta that
   grows *within* a row and resets at row boundaries means the recurrence;
   one that is uniform means the gate matrices are transposed or misordered.

Record the baseline profile in the bead so later regressions have something to
compare against:

```bash
bd update cad-l1b --append-notes "Activation deltas vs instrumented tesseract 5.5.3 (DOTPRODUCT=native, --psm 13), h36/0001.png: LSTMInput <WxH scale>, Normalized <d>, Convolve <d>, ConvNL <d>, Maxpool <d>, Lfys64 <d>, Lfx96 <d>, Lrx96 <d>, Lfx512 <d>, Output <d>."
```

**Expect divergence, and do not chase zero.** Tesseract's softmax calls the
system `exp`, its dot product may be compiled with FMA contraction, and its
NEON path reassociates. Text equality is the acceptance criterion (Task 18);
this profile exists to localize a *structural* bug, which shows up as orders of
magnitude, not ulps.

- [ ] **Step 7: Commit**

```bash
git add internal/nn/layer.go internal/recog/build.go cmd/cadmusdump testdata/tessdebug
git commit -m "feat(cmd): dump per-layer activations for the Tesseract differential"
```

---

## Task 15: Verify L1a's unicharset parser meets the decoder's needs

**This task creates no file.** L1a Task 5 already ships
`internal/tessdata/unicharset.go`, its tests, and the two traps this plan used to
restate: properties are **hexadecimal**, and unichar id 0's stored token *and*
stored `normed` are both the literal string `NULL`, rewritten to `" "` on load.
Re-creating the parser here would collide with L1a's file and its tests.

L1a's surface differs from the one this plan was drafted against, and the
difference is not cosmetic — it is fields versus methods:

| this plan's draft | L1a's actual API |
|---|---|
| `u.Chars []Unichar` | unexported; use the accessors |
| `u.Len()` | `u.Size()` |
| `u.Chars[id].Text` | `u.Text(id)` |
| `u.Chars[id].Normed` | `u.Normed(id)` |
| `u.Chars[id].Properties` (`int`) | `ch, ok := u.Char(id)`, then `ch.Properties` (`uint32`) |
| `PropDigit` (`int`) | `PropDigit uint32` |

Every later code block in this plan that indexes `Charset.Chars` must be written
against the right-hand column. Those are compile errors, not silent bugs.

**Files:** none.

- [ ] **Step 1: Confirm the parser is present and correct on the real model**

```bash
go test ./internal/tessdata/ -run TestParseUnicharset -v
```

Expected, from L1a's own assertions: 112 entries; ids 0/1/2 are `" "`,
`"Joined"`, `"|Broken|0|1"`; exactly **6** entries whose `Normed` differs from
their `Text` (55 `’`→`'`, 59 `™`→`TM`, 60 `“`→`"`, 70 `—`→`-`,
71 `”`→`"`, 84 `‘`→`'`).

**If L1a's tests do not pin the count of 6**, add that assertion to L1a's test
file. It is the cheapest guard there is against a parser that silently drops the
`normed` field, and Task 18 depends on `Text` and `Normed` being distinguishable.

- [ ] **Step 2: Confirm the accessor the decoder needs**

Task 18 emits the **raw token** (`Text`), not the normalized form, because the
`tesseract` binary prints `id_to_unichar_ext` while `DecodeLabels` prints the
normalized one, and the acceptance oracle is the binary. Confirm both accessors
exist and disagree where they should:

```bash
go doc ./internal/tessdata Unicharset
combine_tessdata -u testdata/eng.traineddata /tmp/eng.
awk 'NR>1' /tmp/eng.lstm-unicharset | awk '{print $1, $NF}' | awk '$1 != $2' | head
```

The `awk` output is a quick eyeball of the six differing entries; it is not
authoritative (short-form lines have no `normed` field and will appear here
spuriously), which is why the test count is.

- [ ] **Step 3: Nothing to commit**

This task produces no diff.

---

## Task 16: Restrict the recoder to single-code entries, in `recog`

**This task creates no file either.** L1a Task 6 already ships
`internal/tessdata/recoder.go` with the wire format, `ComputeCodeRange` and
`SetupDecoder`. It deliberately *supports* multi-code entries, because the loader
has no business refusing to load a valid model.

The single-code restriction is L1b's, and it belongs where the limitation
actually bites: the beam search. Enforce it in `recog.NewRecognizer` (Task 18),
not in the loader.

L1a's API, again methods rather than fields:

| this plan's draft | L1a's actual API |
|---|---|
| `rc.Encode [][]int` | `rc.Encode(unicharID int) []int32` |
| `rc.Decode []int` | `rc.DecodeUnichar(code []int32) (int, bool)` |
| `rc.CodeRange int` | `rc.CodeRange() int` |
| `len(rc.Encode)` | `rc.Size()` |
| — | `rc.MaxCodeLen() int`, `rc.IsValidFirstCode(code int32) bool` |

`MaxCodeLen()` is what makes the restriction a one-liner.

**Files:**
- Modify: `internal/recog/decode.go` (the guard; the file is created in Task 18)
- Modify: `internal/recog/decode_test.go`

- [ ] **Step 1: Confirm L1a's parser on the real model**

```bash
go test ./internal/tessdata/ -run TestParseRecoder -v
```

Expected: 112 entries, `CodeRange() == 111`, `MaxCodeLen() == 1`,
`Encode(UnicharSpace)[0] == 0` ("Space was garbled in recoding"),
`Encode(UnicharJoined)[0] == Encode(UnicharBroken)[0] == 110`, and
`DecodeUnichar([]int32{110})` returning `UnicharBroken` — the higher of the two
colliding ids, because `SetupDecoder` writes ascending.

- [ ] **Step 2: Add the guard where it belongs**

In `NewRecognizer` (Task 18 Step 3), alongside the other cross-component checks:

```go
	// L1b implements only the length-1 recoder fast path: network output index
	// -> code -> unichar id is then a flat near-permutation and the whole
	// partial-code dimension of RecodeBeamSearch (kNumLengths, GetNextCodes,
	// GetFinalCodes) collapses to a no-op. CJK and Indic models need that
	// machinery; see cad-l1-cjk.
	if rc.MaxCodeLen() != 1 {
		return nil, fmt.Errorf("recog: recoder has codes up to %d long; L1b supports single-code recoders only (see cad-l1-cjk)", rc.MaxCodeLen())
	}
```

and a test in `internal/recog/decode_test.go` that a synthetic multi-code
recoder is rejected. Build it with L1a's own `buildRecoder` helper shape rather
than a second copy.

- [ ] **Step 3: File the deferral**

```bash
bd create --title "cad-l1-cjk: support multi-code recoders (CJK, Indic)" \
  --description "internal/recog rejects any model whose recoder MaxCodeLen() > 1. internal/tessdata already loads them. Supporting chi_sim/jpn/kor/Indic needs RecodedCharID prefixes, is_valid_start_, next_codes_ and final_codes_ (L1a deliberately did not build the last two), plus the partial-code beam dimension (kNumLengths, kBeamWidths) in internal/recog/beam.go. Verified: every eng entry has length 1, so nothing is lost today."
```

- [ ] **Step 4: Commit**

Fold this into Task 18's commit — the guard lives in `decode.go`, which does not
exist until then.

---

## Task 17: The acceptance corpus and the `tesseract --psm 13` oracle

The oracle for the whole of L1 is `tesseract --psm 13` on line crops — see the
box at the top of this plan for why it is not `--psm 7`. This task builds the
corpus, captures the oracle output, and commits both so that CI needs neither
`tesseract` nor `text2image`.

**Two arms, and the split is the point:**

- **`h36/`** — crops already exactly 36 pixels tall, produced with Leptonica's
  own `pixScale`. At recognition time `im_factor` is 1.0, so the scaler is an
  identity on both sides and **Cadmus and Tesseract see byte-identical pixels**.
  A Stage-1 failure here is a forward-pass or decode bug, full stop.

  That claim rests on two things, both of which are *verified*, not assumed:
  `--psm 13` making `word_box` the whole image (Task 14 Step 5b measures the
  height Tesseract actually feeds), and `pixScale(pix, 1.0, 1.0)` being an exact
  copy (Task 12 Step 6 acceptance rule 3). If either fails, this arm loses its
  special status and the h36-vs-native split in Task 18 Step 5 rung 5 is
  meaningless — say so in the bead rather than continuing.
- **`native/`** — the same lines at their natural height. A failure here that
  does not reproduce in `h36/` is a scaler fidelity problem (Task 12), not a
  runtime problem.

**Files:**
- Create: `testdata/lines/gen.sh`, `testdata/lines/corpus.txt`
- Create: `testdata/golden/gen/scaleline.c` and its `Makefile` rule
- Create: `testdata/lines/h36/*.png`, `testdata/lines/native/*.png`,
  `testdata/lines/*/[0-9]*.gt.txt`, `testdata/lines/*/[0-9]*.psm13.txt`
- Create: `internal/recog/oracle_test.go` (the shared corpus loader)

- [ ] **Step 1: Write the corpus text**

Create `testdata/lines/corpus.txt`, one line of ground truth per line. Cover the
cases Kleio actually sees, and include at least one line per category:

```
The quick brown fox jumps over the lazy dog.
Invoice No. 2024-00817 dated 14 March 2024
Total due: $1,234.56 (incl. 20% VAT)
Patient: DOBBYN, CHRISTOPHER  DOB 1985-02-11
Prescribed 500mg twice daily for 10 days.
Section 4.2 -- Termination for convenience
See Appendix B, pages 17-23, for the schedule.
Contact: chris@dobbo.ca or +44 20 7946 0958
ACME HOLDINGS LIMITED
Registered office: 1 High Street, London EC1A 1BB
The naive cafe served creme brulee and jalapeno.
"Quoted text," she said, 'with nested quotes.'
Balance b/f 0.00 | Debits 1,500.00 | Credits 250.00
0123456789 OIl1 rn m cl d
Ref: XJ-9921/A rev. 3
```

The last two lines are deliberate: `OIl1`, `rn`/`m` and `cl`/`d` are the
confusion pairs where greedy and beam decoding are most likely to disagree, so
they are the lines that will show Stage 2's value.

- [ ] **Step 2: Write the generator**

Create `testdata/lines/gen.sh`:

```bash
#!/usr/bin/env bash
# Renders the acceptance corpus and captures the tesseract --psm 13 oracle.
# Manual step; never run in CI — the PNGs and the .psm13.txt files are committed.
#
# Requires: text2image and tesseract (Homebrew tesseract ships both),
# ImageMagick 7 (`magick`), and testdata/golden/gen/scaleline built from the L0
# golden harness.
set -euo pipefail
cd "$(dirname "$0")"

FONT="${FONT:-Times New Roman}"
PTSIZE="${PTSIZE:-20}"
TESSDATA="${TESSDATA:-..}"

rm -rf native h36
mkdir -p native h36

i=0
while IFS= read -r text; do
  [ -z "$text" ] && continue
  i=$((i+1))
  id=$(printf '%04d' "$i")
  printf '%s\n' "$text" > "native/$id.gt.txt"
  cp "native/$id.gt.txt" "h36/$id.gt.txt"

  printf '%s' "$text" > /tmp/cadmus-line.txt
  text2image --text=/tmp/cadmus-line.txt --outputbase="/tmp/cadmus-$id" \
    --font="$FONT" --ptsize="$PTSIZE" --margin=8 --leading=0 \
    --xsize=4000 --ysize=120 --unicharset_file=/dev/null >/dev/null 2>&1

  # Trim to the ink and flatten to 8-bit grey.
  magick "/tmp/cadmus-$id.tif" -colorspace Gray -trim +repage -bordercolor white \
    -border 4x4 -depth 8 "native/$id.png"

  # The 36-pixel arm goes through Leptonica's own scaler, so that at
  # recognition time im_factor is exactly 1.0 and no resampling happens on
  # either side.
  ../golden/gen/scaleline "native/$id.png" 36 "h36/$id.png"
done < corpus.txt

# Capture the oracle. --psm 13 is PSM_RAW_LINE, the only mode that feeds the LSTM
# the whole image (src/ccmain/linerec.cpp:229-250); --psm 7 hands it a
# baseline-derived per-word sub-crop instead. DOTPRODUCT=native pins the
# reference scalar dot product.
for arm in native h36; do
  for png in "$arm"/*.png; do
    base="${png%.png}"
    DOTPRODUCT=native tesseract --tessdata-dir "$TESSDATA" --psm 13 --dpi 300 \
      "$png" - 2>/dev/null | sed -e '$ { /^$/d }' > "$base.psm13.txt"
  done
done

echo "corpus: $i lines x 2 arms"
wc -l corpus.txt
```

`scaleline` is a new program in the L0 golden harness, not a four-line addition:
it needs PNG I/O and argument handling. Create
`testdata/golden/gen/scaleline.c`:

```c
/* Scales a grayscale PNG to an exact pixel height using Leptonica's own
 * pixScale, so that the h36 corpus arm goes through the same resampler
 * Tesseract will NOT have to run at recognition time (im_factor == 1.0).
 * Build: make -C testdata/golden/gen scaleline
 * Usage: scaleline <in.png> <height> <out.png>
 */
#include <stdio.h>
#include <stdlib.h>
#include <leptonica/allheaders.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <in.png> <height> <out.png>\n", argv[0]);
        return 2;
    }
    int target = atoi(argv[2]);
    if (target <= 0) {
        fprintf(stderr, "scaleline: bad height %s\n", argv[2]);
        return 2;
    }
    PIX *src = pixRead(argv[1]);
    if (src == NULL) {
        fprintf(stderr, "scaleline: cannot read %s\n", argv[1]);
        return 1;
    }
    PIX *gray = (pixGetDepth(src) == 8) ? pixClone(src) : pixConvertTo8(src, 0);
    float f = (float)target / (float)pixGetHeight(gray);
    PIX *dst = pixScale(gray, f, f);
    if (dst == NULL) {
        fprintf(stderr, "scaleline: pixScale failed\n");
        return 1;
    }
    /* Report what Leptonica actually produced; the caller checks it. */
    fprintf(stderr, "scaleline: %s %dx%d -> %dx%d (factor %g)\n", argv[1],
            pixGetWidth(gray), pixGetHeight(gray),
            pixGetWidth(dst), pixGetHeight(dst), f);
    if (pixWrite(argv[3], dst, IFF_PNG) != 0) {
        fprintf(stderr, "scaleline: cannot write %s\n", argv[3]);
        return 1;
    }
    pixDestroy(&dst);
    pixDestroy(&gray);
    pixDestroy(&src);
    return 0;
}
```

and the `Makefile` rule beside the existing `gen` target — reuse whatever
`CFLAGS`/`LDLIBS` that target already resolves for Leptonica:

```make
scaleline: scaleline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
```

```bash
chmod +x testdata/lines/gen.sh
make -C testdata/golden/gen scaleline
./testdata/lines/gen.sh
ls testdata/lines/h36 | head
```

- [ ] **Step 3: Sanity-check the oracle before trusting it**

```bash
# Every h36 crop must actually be 36 px tall. scaleline prints what Leptonica
# produced; this re-checks the committed file.
for f in testdata/lines/h36/*.png; do magick identify -format '%h ' "$f"; done; echo
# The oracle should mostly agree with ground truth. It will not always; that is
# fine and expected, and the *oracle* is what Cadmus must match, not the truth.
paste -d'|' testdata/lines/h36/0001.gt.txt testdata/lines/h36/0001.psm13.txt
```

**If any h36 crop is not exactly 36 px**, `pixScale`'s rounding did not land on
the target for that source height. Do not paper over it: drop that line from the
corpus or nudge its `PTSIZE`, because Task 18's h36 arm is the only arm where a
decode failure is unambiguously a decode failure.

Count the lines where the oracle differs from ground truth and record it:

```bash
n=0; d=0
for f in testdata/lines/h36/*.gt.txt; do
  n=$((n+1)); cmp -s "$f" "${f%.gt.txt}.psm13.txt" || d=$((d+1))
done
echo "oracle differs from ground truth on $d of $n lines"
bd update cad-l1b --append-notes "Corpus: $n lines, 2 arms. tesseract --psm 13 differs from ground truth on $d/$n h36 lines."
```

**If the oracle differs from ground truth on more than about a third of the
lines, the rendering is bad** — the point size is too small, the trim is too
tight, or `text2image` picked a fallback font. Fix the corpus before writing any
decoder against it; a noisy oracle makes Stage 1 unfalsifiable.

- [ ] **Step 4: Write the shared corpus loader**

Create `internal/recog/oracle_test.go`:

```go
package recog

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// corpusLine is one committed acceptance case.
type corpusLine struct {
	Name   string
	Image  image.Image
	Truth  string // the text that was rendered
	Oracle string // what `tesseract --psm 13` produced
}

// loadCorpus reads one arm of testdata/lines. Arms are "h36" (already 36 px
// tall, so the scaler is an identity on both sides) and "native".
//
// The parameter is testing.TB, not *testing.T, so Task 24's benchmark can call
// this same helper instead of a near-identical copy.
func loadCorpus(t testing.TB, arm string) []corpusLine {
	t.Helper()
	dir := filepath.Join("..", "..", "testdata", "lines", arm)
	pngs, err := filepath.Glob(filepath.Join(dir, "*.png"))
	if err != nil || len(pngs) == 0 {
		t.Skipf("corpus arm %q not present (run ./testdata/lines/gen.sh): %v", arm, err)
	}
	var out []corpusLine
	for _, p := range pngs {
		base := strings.TrimSuffix(p, ".png")
		f, err := os.Open(p)
		if err != nil {
			t.Fatalf("opening %s: %v", p, err)
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("decoding %s: %v", p, err)
		}
		truth, err := os.ReadFile(base + ".gt.txt")
		if err != nil {
			t.Fatalf("reading ground truth for %s: %v", p, err)
		}
		oracle, err := os.ReadFile(base + ".psm13.txt")
		if err != nil {
			t.Fatalf("reading oracle for %s: %v", p, err)
		}
		out = append(out, corpusLine{
			Name:   filepath.Base(base),
			Image:  img,
			Truth:  strings.TrimRight(string(truth), "\n"),
			Oracle: strings.TrimRight(string(oracle), "\n"),
		})
	}
	return out
}

// cer is the Levenshtein character error rate of got against want.
func cer(got, want string) float64 {
	a, b := []rune(want), []rune(got)
	if len(a) == 0 {
		if len(b) == 0 {
			return 0
		}
		return 1
	}
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min(min(curr[j-1]+1, prev[j]+1), prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return float64(prev[len(b)]) / float64(len(a))
}

func TestCorpusIsPresentAndConsistent(t *testing.T) {
	for _, arm := range []string{"h36", "native"} {
		lines := loadCorpus(t, arm)
		if len(lines) < 10 {
			t.Errorf("arm %q has %d lines; want at least 10", arm, len(lines))
		}
		for _, l := range lines {
			if l.Oracle == "" {
				t.Errorf("%s/%s: oracle output is empty", arm, l.Name)
			}
			if arm == "h36" && l.Image.Bounds().Dy() != 36 {
				t.Errorf("h36/%s is %d px tall; want 36", l.Name, l.Image.Bounds().Dy())
			}
		}
	}
}
```

- [ ] **Step 5: Run the loader test**

Run: `go test ./internal/recog/ -run TestCorpus -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add testdata/lines testdata/golden/gen internal/recog/oracle_test.go
git commit -m "test(recog): add the line-crop corpus and the psm 13 oracle"
```

---

## Task 18: Greedy CTC decode — Stage 1

Per-timestep argmax, drop the blank, collapse runs of the same code. That is
`ExtractBestPathAsLabels`'s collapse rule applied to the argmax path instead of
the beam's path:

```
for t in 0..T-1:
    c = argmax over outputs[t]
    if c != nullChar and c != prev:
        emit c at t
    prev = c
```

Do **not** use `LabelsViaSimpleText`, which skips the repeat collapse entirely —
it is only correct for models trained without CTC, which excludes every shipped
`.traineddata`.

Codes map to unichar ids through the recoder, and unichar ids to text through
the unicharset. **Use `Text` (the raw token), not `Normed`.** The two differ for
six eng entries, and the `tesseract` binary emits the raw form
(`id_to_unichar_ext`) while `LSTMRecognizer::DecodeLabels` emits the normalized
one. Matching the binary is what the acceptance oracle measures.

> **L1a API reminder.** `tessdata.Unicharset` and `tessdata.Recoder` expose
> methods, not fields (see Preconditions). Throughout this task and Tasks 19-23:
> `r.Charset.Text(id)`, `r.Charset.Size()`, `r.Recoder.Encode(id)` returning
> `[]int32`, `r.Recoder.DecodeUnichar([]int32{int32(code)})` returning
> `(id, ok)`, `r.Recoder.CodeRange()`, `r.Recoder.Size()`. The code blocks below
> are written against that API; if one of them still reads `Charset.Chars[...]`
> or `Recoder.Decode[...]`, it is a leftover from an earlier draft — fix it, do
> not add a shim.

**Files:**
- Create: `internal/recog/decode.go`
- Test: `internal/recog/decode_test.go`

**Interfaces:**
- Consumes: `Network`, `Normalized`, `tessdata.Recoder`, `tessdata.Unicharset`.
- Produces:

```go
// Recognizer is a loaded model ready to transcribe line images.
type Recognizer struct {
	Net     *Network
	Charset *tessdata.Unicharset
	Recoder *tessdata.Recoder
}

func NewRecognizer(model []byte) (*Recognizer, error)

// Symbol is one decoded character and the timestep run it occupies.
type Symbol struct {
	UnicharID int
	Text      string
	Start     int // first timestep of the run
	End       int // one past the last timestep of the run
}

// GreedyDecode collapses an output tensor to symbols.
func (r *Recognizer) GreedyDecode(out *nn.Tensor) ([]Symbol, error)
```

- [ ] **Step 1: Write the failing test**

Create `internal/recog/decode_test.go`:

```go
package recog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// testing.TB rather than *testing.T, so Task 24's benchmark reuses this helper
// rather than a duplicate named loadRecognizerB.
func loadRecognizer(t testing.TB) *Recognizer {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	r, err := NewRecognizer(raw)
	if err != nil {
		t.Fatalf("NewRecognizer() error = %v", err)
	}
	return r
}

// oneHot builds a synthetic softmax output that puts all the mass on the given
// code at each timestep, so the collapse rule can be tested in isolation.
func oneHot(codes []int, n int) *nn.Tensor {
	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: len(codes)}, n)
	row := make([]float64, n)
	for t, c := range codes {
		for i := range row {
			row[i] = 0.001
		}
		row[c] = 0.9
		out.WriteTimeStep(t, row)
	}
	return out
}

func TestGreedyDecodeCollapsesRepeatsAndDropsBlanks(t *testing.T) {
	r := loadRecognizer(t)
	null := r.Net.NullChar
	// codes: A A blank A blank blank B  ->  A A B
	// (the blank between the two A runs is what keeps them separate)
	codeA := codeFor(t, r, "A")
	codeB := codeFor(t, r, "B")
	syms, err := r.GreedyDecode(oneHot([]int{codeA, codeA, null, codeA, null, null, codeB}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	var got strings.Builder
	for _, s := range syms {
		got.WriteString(s.Text)
	}
	if got.String() != "AAB" {
		t.Fatalf("decoded %q; want \"AAB\"", got.String())
	}
	// The run boundaries must be the argmax runs, not the emission points.
	if syms[0].Start != 0 || syms[0].End != 2 {
		t.Errorf("first symbol run = [%d,%d); want [0,2)", syms[0].Start, syms[0].End)
	}
	if syms[1].Start != 3 || syms[1].End != 4 {
		t.Errorf("second symbol run = [%d,%d); want [3,4)", syms[1].Start, syms[1].End)
	}
	if syms[2].Start != 6 || syms[2].End != 7 {
		t.Errorf("third symbol run = [%d,%d); want [6,7)", syms[2].Start, syms[2].End)
	}
}

func TestGreedyDecodeAllBlank(t *testing.T) {
	r := loadRecognizer(t)
	syms, err := r.GreedyDecode(oneHot([]int{r.Net.NullChar, r.Net.NullChar}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	if len(syms) != 0 {
		t.Errorf("all-blank decode produced %d symbols; want 0", len(syms))
	}
}

// The raw token, not the normalized form: `tesseract` emits U+2019, not "'".
func TestGreedyDecodeEmitsTheRawToken(t *testing.T) {
	r := loadRecognizer(t)
	id := unicharIDFor(t, r, "’")
	if r.Charset.Normed(id) == r.Charset.Text(id) {
		t.Skip("this model does not normalize the right single quote")
	}
	syms, err := r.GreedyDecode(oneHot([]int{codeFor(t, r, "’")}, r.Net.NumOutputs))
	if err != nil {
		t.Fatalf("GreedyDecode() error = %v", err)
	}
	if syms[0].Text != "’" {
		t.Errorf("decoded %q; want the raw token U+2019, not the normed form", syms[0].Text)
	}
}

func unicharIDFor(t testing.TB, r *Recognizer, s string) int {
	t.Helper()
	for id := range r.Charset.Size() {
		if r.Charset.Text(id) == s {
			return id
		}
	}
	t.Fatalf("no unichar %q in the charset", s)
	return -1
}

// codeFor is the network output index for a character. Recoder.Encode is a
// method returning []int32 (L1a Task 6), and L1b has already asserted every
// entry is length 1, so index 0 is the whole code.
func codeFor(t testing.TB, r *Recognizer, s string) int {
	t.Helper()
	codes := r.Recoder.Encode(unicharIDFor(t, r, s))
	if len(codes) != 1 {
		t.Fatalf("unichar %q encodes to %d codes; want 1", s, len(codes))
	}
	return int(codes[0])
}

// The acceptance test. Cadmus's text must match the committed
// `tesseract --psm 13` output on the corpus arm where neither side resamples.
func TestGreedyDecodeMatchesOracleH36(t *testing.T) {
	r := loadRecognizer(t)
	lines := loadCorpus(t, "h36")

	var exact int
	var totalCER float64
	for _, l := range lines {
		got, err := r.RecognizeText(l.Image)
		if err != nil {
			t.Errorf("%s: RecognizeText() error = %v", l.Name, err)
			continue
		}
		if got == l.Oracle {
			exact++
		} else {
			t.Logf("%s\n  oracle: %q\n  cadmus: %q\n  cer:    %.4f", l.Name, l.Oracle, got, cer(got, l.Oracle))
		}
		totalCER += cer(got, l.Oracle)
	}
	meanCER := totalCER / float64(len(lines))
	t.Logf("h36 arm: %d/%d exact, mean CER vs oracle %.4f", exact, len(lines), meanCER)

	// Greedy decoding has no dictionary, so it will not match the oracle
	// everywhere; the bound is what a correct forward pass with no lexicon can
	// be expected to reach on clean rendered text. See Step 5 for what to do
	// when this fails.
	if meanCER > 0.02 {
		t.Errorf("mean CER vs oracle = %.4f; want <= 0.02", meanCER)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/recog/ -run TestGreedy -v`
Expected: FAIL — `undefined: NewRecognizer`.

- [ ] **Step 3: Implement**

Create `internal/recog/decode.go`:

```go
// This file is a Go translation of RecodeBeamSearch::ExtractBestPathAsLabels in
// src/lstm/recodebeam.cpp and LSTMRecognizer::DecodeLabels in
// src/lstm/lstmrecognizer.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"
	"image"
	"strings"

	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// Recognizer is a loaded model ready to transcribe line images.
type Recognizer struct {
	Net     *Network
	Charset *tessdata.Unicharset
	Recoder *tessdata.Recoder
}

// NewRecognizer loads a .traineddata file.
func NewRecognizer(model []byte) (*Recognizer, error) {
	c, err := tessdata.ParseContainer(model)
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm component")
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	net, err := Build(rec)
	if err != nil {
		return nil, err
	}

	// LSTMRecognizer::LoadCharsets reads components 21 and 22 when both are
	// present, which is the case for every model in tessdata, tessdata_best and
	// tessdata_fast. The legacy embedded-charset layout is not supported.
	csEntry, ok := c.Entry(tessdata.TypeLSTMUnicharset)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm-unicharset component; the embedded-charset layout is not supported")
	}
	cs, err := tessdata.ParseUnicharset(csEntry)
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	rcEntry, ok := c.Entry(tessdata.TypeLSTMRecoder)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm-recoder component")
	}
	rc, err := tessdata.ParseRecoder(rcEntry, c.Swapped())
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}

	// Free cross-checks between the four components. Each of these has a
	// distinct failure it catches, and all are cheap.
	if rc.CodeRange() != net.NumOutputs {
		return nil, fmt.Errorf("recog: recoder code range %d does not match the %d network outputs", rc.CodeRange(), net.NumOutputs)
	}
	if rc.Size() != cs.Size() {
		return nil, fmt.Errorf("recog: recoder has %d entries but the unicharset has %d", rc.Size(), cs.Size())
	}
	// L1b implements only the length-1 recoder fast path: network output index
	// -> code -> unichar id is then a flat near-permutation, and the whole
	// partial-code dimension of RecodeBeamSearch (kNumLengths, GetNextCodes,
	// GetFinalCodes) collapses to a no-op. internal/tessdata loads multi-code
	// recoders happily; the restriction is the beam's, so it lives here.
	if rc.MaxCodeLen() != 1 {
		return nil, fmt.Errorf("recog: recoder has codes up to %d long; L1b supports single-code recoders only (see cad-l1-cjk)", rc.MaxCodeLen())
	}
	if space := rc.Encode(tessdata.UnicharSpace); len(space) != 1 || space[0] != tessdata.UnicharSpace {
		return nil, fmt.Errorf("recog: space was garbled in recoding")
	}
	return &Recognizer{Net: net, Charset: cs, Recoder: rc}, nil
}

// Symbol is one decoded character and the timestep run it occupies.
type Symbol struct {
	UnicharID int
	Text      string
	Start     int
	End       int
}

// Forward normalizes a line image and runs it through the network, returning
// the softmax output and the normalization metadata the geometry needs.
func (r *Recognizer) Forward(img image.Image) (*nn.Tensor, *Normalized, error) {
	// Normalize shares the network's randomizer and must run before the graph,
	// exactly as Copy2DImage precedes Convolve::Forward.
	norm, err := Normalize(img, r.Net.InputHeight, r.Net.XScale, r.Net.Rand)
	if err != nil {
		return nil, nil, err
	}
	out, err := r.Net.Root.Forward(norm.Input)
	if err != nil {
		return nil, nil, fmt.Errorf("recog: forward: %w", err)
	}
	if out.Features != r.Net.NumOutputs {
		return nil, nil, fmt.Errorf("recog: network produced %d outputs, want %d", out.Features, r.Net.NumOutputs)
	}
	return out, norm, nil
}

// GreedyDecode is the CTC best-path decode: per-timestep argmax, blanks
// dropped, runs of the same code collapsed to one symbol.
//
// This is deliberately not LabelsViaSimpleText, which omits the run collapse
// and is only correct for models trained without CTC.
func (r *Recognizer) GreedyDecode(out *nn.Tensor) ([]Symbol, error) {
	var syms []Symbol
	prev := -1
	for t := range out.Map.Len() {
		code := argmax(out.Row(t))
		if code != prev && code != r.Net.NullChar {
			id, ok := r.Recoder.DecodeUnichar([]int32{int32(code)})
			if !ok || id < 0 || id >= r.Charset.Size() {
				return nil, fmt.Errorf("recog: code %d at t=%d does not decode to a unichar id inside the charset (got %d, ok=%v)", code, t, id, ok)
			}
			syms = append(syms, Symbol{UnicharID: id, Text: r.Charset.Text(id), Start: t, End: t + 1})
		} else if code == prev && len(syms) > 0 && code != r.Net.NullChar {
			syms[len(syms)-1].End = t + 1
		}
		prev = code
	}
	return syms, nil
}

// RecognizeText is the whole Stage 1 pipeline: image in, text out.
func (r *Recognizer) RecognizeText(img image.Image) (string, error) {
	out, _, err := r.Forward(img)
	if err != nil {
		return "", err
	}
	syms, err := r.GreedyDecode(out)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, s := range syms {
		b.WriteString(s.Text)
	}
	return b.String(), nil
}

func argmax(row []float32) int {
	best, bestI := row[0], 0
	for i, v := range row[1:] {
		if v > best {
			best, bestI = v, i+1
		}
	}
	return bestI
}
```

- [ ] **Step 4: Run the tests**

```bash
go test ./internal/recog/ -run TestGreedy -v
```

- [ ] **Step 5: When the acceptance test fails — the debugging ladder**

This is the moment the plan exists for. Work the ladder in order and **do not
skip a rung**:

1. **Is the text empty or pure noise?** The forward pass is broken, not the
   decoder. Go to Task 14 Step 6 and read the activation delta profile.
2. **Is the text roughly right but shifted or truncated?** The stride map is
   wrong. Compare `out.Map.Width` against `floor(imageWidth/3)` and confirm the
   `XYTranspose`/`SummLSTM` block collapses to `Height: 1`.
3. **Is the text right but the wrong characters throughout, consistently?**
   The recoder or the charset is misaligned. Print
   `r.Recoder.DecodeUnichar([]int32{int32(argmax)})` for the first ten timesteps and check the ids
   against `combine_tessdata -u`'s unicharset file by hand.
4. **Is it right except for the confusion pairs (`rn`/`m`, `cl`/`d`, `l`/`1`)
   and the occasional word?** That is exactly what Stage 1 is expected to look
   like — greedy decoding has no lexicon. Log the mean CER and move on; Stage 2
   is what closes that gap.
5. **Does `h36` pass but `native` fail?** The scaler. Return to Task 12 with the
   measurement.

Record the outcome either way:

```bash
bd update cad-l1b --append-notes "Stage 1 (greedy, no dict) vs tesseract --psm 13: h36 <n>/<N> exact, mean CER <x>; native <n>/<N> exact, mean CER <x>."
```

**On the 0.02 bound:** it is a plan-imposed target, not a measured Tesseract
figure — no published greedy-versus-beam accuracy delta for `eng` exists, and
the research explicitly declined to invent one. If the measured mean CER lands
between 0.02 and 0.05 with the failures concentrated in confusion pairs and
dictionary words, **raise the bound to the measured value, record the reason in
the bead, and proceed to Stage 2**; that is the signal that the forward pass is
right and only the lexicon is missing. If it is above 0.05, or the failures are
spread evenly across all characters, the forward pass is wrong — go back to
rung 1.

- [ ] **Step 6: Commit**

```bash
git add internal/recog/decode.go internal/recog/decode_test.go
git commit -m "feat(recog): add greedy CTC decoding"
```

---

## Task 19: Per-character certainty, rating and boundaries

Kleio's validation gate reads per-word confidence with bounding boxes, so this
arithmetic is a requirement, not a nicety. It is a four-stage pipeline and every
stage has a constant that matters.

**Stage 1 — probability to certainty** (`NetworkIO::ProbToCertainty`):

```
kMinCertainty = -20
kMinProb      = exp(-20)
certainty(p)  = p > kMinProb ? log(p) : kMinCertainty
```

Every node's certainty is `ProbToCertainty(p) + kCertOffset` with
`kCertOffset = -0.085`, and a non-dictionary node is then multiplied by
`kDictRatio = 2.25`. **That scaling survives into the reported confidence** —
a non-dictionary word's certainty is 2.25x worse than its raw log probability.
Greedy decoding has no dictionary at all, so every character takes the 2.25.

**Stage 2 — nodes to per-character values** (`ExtractPathAsUnicharIds`):

- `certs[i]` is the **minimum** certainty over the run `[preceding blanks .. end
  of the character's duplicate run]`, initialised to 0.
- `ratings[i]` is the **negated sum** of certainties over the same span, a
  positive cost.
- The blanks *preceding* a character are charged to that character.
- Trailing blanks after the last character fold back onto it.

**Stage 3 — character boundaries** (`calculateCharBoundaries`): for `n >= 1`
characters there are `n+1` boundaries; `bounds[0] = 0`, `bounds[n] = width`, and
`bounds[i] = ends[i-1] + (starts[i] - ends[i-1]) / 2` — the floor of the midpoint
of the blank gap.

**`n == 0` is the exception, and it is Tesseract's, not a port artefact.** The
C++ pushes `0`, runs a loop over `ends` that does not execute, then
`pop_back()` — which removes the `0` it just pushed — and finally pushes
`maxWidth`. So an empty path yields the single-element `[width]`, not `[0]` and
not `[0, width]`. `TestCharBoundariesEdgeCases` pins that.

**Files:**
- Create: `internal/recog/certainty.go`
- Test: `internal/recog/certainty_test.go`

**Interfaces:**
- Produces:

```go
const (
	MinCertainty   = -20.0  // kMinCertainty
	CertOffset     = -0.085 // kCertOffset
	DictRatio      = 2.25   // kDictRatio
	CertaintyScale = 7.0    // kCertaintyScale
)

func ProbToCertainty(p float64) float64

// Scored is a Symbol with its CTC certainty and rating attached.
type Scored struct {
	Symbol
	Certainty float64 // min over the character's span, already offset and scaled
	Rating    float64 // -sum over the same span
}

// ScoreSymbols attaches certainties and ratings, charging each character for
// the blanks that precede it. dictRatio is 2.25 for the greedy path.
//
// There is deliberately no nullChar parameter: the span rule reads the argmax
// probability at every timestep in the span, which is what ExtractPathAsUnicharIds
// does with the chosen node's own probability, and under greedy decoding the
// chosen node IS the argmax at every timestep — blanks included. Taking a
// nullChar and not using it would be a lie about the algorithm and `unparam`
// would flag it.
func ScoreSymbols(out *nn.Tensor, syms []Symbol, dictRatio float64) []Scored

// CharBoundaries returns len(syms)+1 timestep boundaries.
func CharBoundaries(syms []Scored, width int) []int
```

- [ ] **Step 1: Write the failing test**

Create `internal/recog/certainty_test.go`:

```go
package recog

import (
	"math"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

func TestProbToCertaintyFloor(t *testing.T) {
	if got := ProbToCertainty(1.0); got != 0 {
		t.Errorf("ProbToCertainty(1) = %v; want 0", got)
	}
	if got, want := ProbToCertainty(0.5), math.Log(0.5); got != want {
		t.Errorf("ProbToCertainty(0.5) = %v; want %v", got, want)
	}
	if got := ProbToCertainty(0); got != MinCertainty {
		t.Errorf("ProbToCertainty(0) = %v; want %v", got, MinCertainty)
	}
	if got := ProbToCertainty(math.Exp(MinCertainty) / 2); got != MinCertainty {
		t.Errorf("ProbToCertainty below kMinProb = %v; want the floor %v", got, MinCertainty)
	}
}

// probs builds a synthetic output tensor from an explicit per-timestep
// probability for the winning code, so the certainty arithmetic is checkable
// by hand.
func probs(codes []int, p []float64, n int) *nn.Tensor {
	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: len(codes)}, n)
	row := make([]float64, n)
	for t, c := range codes {
		rest := (1 - p[t]) / float64(n-1)
		for i := range row {
			row[i] = rest
		}
		row[c] = p[t]
		out.WriteTimeStep(t, row)
	}
	return out
}

// The blanks before a character are charged to that character, and its
// certainty is the minimum across the whole span.
//
// Every expectation narrows its probability through float32 first. probs()
// writes via nn.Tensor.WriteTimeStep, which stores float32, and ScoreSymbols
// reads float64(row[...]) back — so the value that reaches ProbToCertainty is
// float64(float32(0.9)) = 0.899999976…, whose log differs from log(0.9) by
// ~2.6e-8, four orders of magnitude above any sane tolerance. Task 9's
// equivalent test already does this; the pattern is not optional.
func TestScoreSymbolsChargesPrecedingBlanks(t *testing.T) {
	const null = 9
	// t0 blank p=0.5, t1 'A' p=0.9, t2 'A' p=0.8, t3 blank p=0.99
	out := probs([]int{null, 1, 1, null}, []float64{0.5, 0.9, 0.8, 0.99}, 10)
	syms := []Symbol{{UnicharID: 1, Text: "A", Start: 1, End: 3}}
	scored := ScoreSymbols(out, syms, 1.0)
	if len(scored) != 1 {
		t.Fatalf("got %d scored symbols; want 1", len(scored))
	}
	cert := func(p float64) float64 { return ProbToCertainty(float64(float32(p))) + CertOffset }
	// Span is t0..t3: the leading blank, both 'A' steps, and the trailing blank
	// folded back onto the last character.
	wantCert := math.Min(math.Min(cert(0.5), cert(0.9)), math.Min(cert(0.8), cert(0.99)))
	if math.Abs(scored[0].Certainty-wantCert) > 1e-12 {
		t.Errorf("Certainty = %v; want %v (the minimum over the whole span)", scored[0].Certainty, wantCert)
	}
	wantRating := -(cert(0.5) + cert(0.9) + cert(0.8) + cert(0.99))
	if math.Abs(scored[0].Rating-wantRating) > 1e-12 {
		t.Errorf("Rating = %v; want %v (minus the sum over the span)", scored[0].Rating, wantRating)
	}
}

// The dict ratio multiplies the certainty of every non-dictionary node, and it
// survives into the reported number.
func TestScoreSymbolsAppliesDictRatio(t *testing.T) {
	out := probs([]int{1}, []float64{0.9}, 10)
	plain := ScoreSymbols(out, []Symbol{{UnicharID: 1, Start: 0, End: 1}}, 1.0)
	scaled := ScoreSymbols(out, []Symbol{{UnicharID: 1, Start: 0, End: 1}}, DictRatio)
	if math.Abs(scaled[0].Certainty-plain[0].Certainty*DictRatio) > 1e-12 {
		t.Errorf("scaled certainty = %v; want %v", scaled[0].Certainty, plain[0].Certainty*DictRatio)
	}
}

// n characters give n+1 boundaries; interior boundaries are the floor of the
// midpoint of the blank gap.
func TestCharBoundaries(t *testing.T) {
	syms := []Scored{
		{Symbol: Symbol{Start: 1, End: 3}},
		{Symbol: Symbol{Start: 6, End: 7}},
	}
	got := CharBoundaries(syms, 10)
	want := []int{0, 4, 10} // 3 + (6-3)/2 = 4
	if len(got) != len(want) {
		t.Fatalf("CharBoundaries() = %v; want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("CharBoundaries() = %v; want %v", got, want)
		}
	}
}

// The edge case the research flagged as reasoned-through but never executed:
// a path with no characters at all, and a path that starts and ends on blanks.
func TestCharBoundariesEdgeCases(t *testing.T) {
	if got := CharBoundaries(nil, 10); len(got) != 1 || got[0] != 10 {
		t.Errorf("CharBoundaries(nil, 10) = %v; want [10]", got)
	}
	syms := []Scored{{Symbol: Symbol{Start: 4, End: 5}}}
	got := CharBoundaries(syms, 10)
	if len(got) != 2 || got[0] != 0 || got[1] != 10 {
		t.Errorf("single symbol boundaries = %v; want [0 10]", got)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/recog/ -run 'TestProbTo|TestScoreSymbols|TestCharBound' -v`
Expected: FAIL — `undefined: ProbToCertainty`.

- [ ] **Step 3: Implement**

Create `internal/recog/certainty.go`:

```go
// This file is a Go translation of NetworkIO::ProbToCertainty in
// src/lstm/networkio.cpp and RecodeBeamSearch::ExtractPathAsUnicharIds and
// ::calculateCharBoundaries in src/lstm/recodebeam.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"math"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

const (
	// MinCertainty is kMinCertainty from src/lstm/networkio.cpp.
	MinCertainty = -20.0
	// CertOffset is kCertOffset from src/lstm/lstmrecognizer.cpp, added to
	// every node's certainty.
	CertOffset = -0.085
	// DictRatio is kDictRatio. Tesseract multiplies the certainty of every
	// non-dictionary hypothesis by it, so a word the lexicon does not know is
	// reported 2.25 times worse than its raw log probability. Greedy decoding
	// has no lexicon, so every character takes it.
	DictRatio = 2.25
	// CertaintyScale is kCertaintyScale from src/ccmain/linerec.cpp, applied
	// once at export.
	CertaintyScale = 7.0
)

var minProb = math.Exp(MinCertainty)

// ProbToCertainty is NetworkIO::ProbToCertainty: the log probability, floored.
func ProbToCertainty(p float64) float64 {
	if p > minProb {
		return math.Log(p)
	}
	return MinCertainty
}

// Scored is a Symbol with its CTC certainty and rating attached.
type Scored struct {
	Symbol
	Certainty float64
	Rating    float64
}

// ScoreSymbols attaches per-character certainties and ratings.
//
// The span charged to a character runs from the end of the previous character
// through the end of this one, so the blanks *between* two characters are
// charged to the later one — that is what ExtractPathAsUnicharIds does, and it
// is why a character preceded by a long uncertain gap reports low confidence.
// Trailing blanks after the last character fold back onto it.
func ScoreSymbols(out *nn.Tensor, syms []Symbol, dictRatio float64) []Scored {
	width := out.Map.Len()
	cert := func(t int) float64 {
		row := out.Row(t)
		p := float64(row[argmax(row)])
		return (ProbToCertainty(p) + CertOffset) * dictRatio
	}

	scored := make([]Scored, 0, len(syms))
	prevEnd := 0
	for i, s := range syms {
		end := s.End
		if i == len(syms)-1 {
			// Fold the trailing blanks onto the last character.
			end = width
		}
		c := 0.0
		r := 0.0
		for t := prevEnd; t < end; t++ {
			v := cert(t)
			if v < c {
				c = v
			}
			r -= v
		}
		scored = append(scored, Scored{Symbol: s, Certainty: c, Rating: r})
		prevEnd = s.End
	}
	return scored
}

// CharBoundaries is calculateCharBoundaries: for n >= 1 characters it returns
// n+1 timestep boundaries, where an interior boundary is the floor of the
// midpoint of the blank gap between two characters, and character i occupies
// [bounds[i], bounds[i+1]).
//
// For n == 0 it returns the single element []int{width}, matching the C++'s
// push(0) / empty loop / pop_back / push(maxWidth) sequence. That is not the
// general rule with n substituted; it is a separate case.
func CharBoundaries(syms []Scored, width int) []int {
	bounds := make([]int, 0, len(syms)+1)
	bounds = append(bounds, 0)
	for i := 1; i < len(syms); i++ {
		end := syms[i-1].End
		bounds = append(bounds, end+(syms[i].Start-end)/2)
	}
	if len(syms) == 0 {
		return []int{width}
	}
	return append(bounds, width)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/recog/ -v`
Expected: PASS.

- [ ] **Step 5: Verify the span rule against the source**

`ExtractPathAsUnicharIds` is intricate, and the plan's "charge preceding blanks
to the following character" summary is a paraphrase. Read it before trusting the
Go:

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
sed -n '565,640p' "$SP/tess/src/lstm/recodebeam.cpp"
```

Check three things: the `certainty` accumulator is initialised to `0.0` and not
to `+inf`; `rating -= cert` (a *positive* cost from negative certainties); and
the two `UNICHAR_SPACE` special cases.

**Both space special cases are deliberately absent from the Go above, and their
gates are opposite — do not "fix" one into the other.** Verbatim from
`ExtractPathAsUnicharIds`:

```cpp
// (1) before the run loop — moves the space's accumulated nulls onto the
//     PREVIOUS character. Requires a dictionary permuter.
if (unichar_id == UNICHAR_SPACE && !certs->empty() &&
    best_nodes[t]->permuter != NO_PERM) { … }

// (2) inside the run loop — makes a space FORGET the preceding nulls and take
//     its own certainty. Requires exactly NO_PERM.
if (cert < certainty || (unichar_id == UNICHAR_SPACE &&
                         best_nodes[t - 1]->permuter == NO_PERM)) {
  certainty = cert;
}
```

Case (2) looks like it should fire on the no-dictionary path, and it does not.
`NO_PERM` is stamped on exactly one push in the whole search —
`ContinueUnichar`'s `PushInitialDawgIfBetter` for a space — and that push is
inside `if (dict_ != nullptr && …)`. With no dictionary loaded no node ever
carries `NO_PERM`, every non-dawg node is `TOP_CHOICE_PERM`, and neither case
fires. So the plan's conclusion (both belong to Task 23) is right even though
its original one-line reason was not: it is `dict_ != nullptr` that gates them,
not the sign of the `permuter` comparison.

Implementing case (2) in the greedy path would therefore *diverge* from
Tesseract, not converge on it. Both cases are listed in Task 23 Step 1's
transcription checklist instead.

- [ ] **Step 6: Commit**

```bash
git add internal/recog/certainty.go internal/recog/certainty_test.go
git commit -m "feat(recog): add per-character certainty, rating and boundaries"
```

---

## Task 20: Words — segmentation, confidence and bounding boxes

The deliverable Kleio actually consumes. Words are cut at `UNICHAR_SPACE`, and
in the dictionary path also at a `start_of_word` node and at every character in
a non-space-delimited script.

Word confidence, end to end:

```
word certainty = min over the word's characters of certs[i]
word certainty = min(word certainty, space certainty)   // the flanking spaces
confidence %   = clip(100 + 5 * CertaintyScale * certainty, 0, 100)
               = clip(100 + 35 * certainty, 0, 100)
```

Word geometry:

```
scale = XScale / imFactor                                   // page px per timestep
left  = floor(bounds[wordStart] * scale) + lineBox.Min.X
right = ceil (bounds[wordEnd]   * scale) + lineBox.Min.X
top, bottom = the line strip's own extent
```

**Y is not derived from the network at all.** Every word inherits the full line
strip's vertical extent, exactly as `InitializeWord` does. Tightening it needs
connected-component ink from `internal/imaging`, which is L2's problem.

**Files:**
- Create: `internal/recog/words.go`
- Test: `internal/recog/words_test.go`

**Interfaces:**
- Produces:

```go
// Word is one recognized word with its geometry and confidence.
type Word struct {
	Text       string
	Bounds     image.Rectangle
	Confidence float64 // 0-100
}

// Line is one recognized line.
type Line struct {
	Text       string
	Words      []Word
	Bounds     image.Rectangle
	Confidence float64
}

// Recognize is the full pipeline: line image in, Line out.
func (r *Recognizer) Recognize(img image.Image) (Line, error)
```

- [ ] **Step 1: Write the failing test**

Create `internal/recog/words_test.go`:

```go
package recog

import (
	"image"
	"math"
	"strings"
	"testing"
)

func TestConfidencePercent(t *testing.T) {
	// A dictionary-clean word with every timestep at p >= 0.99 has certainty
	// about -0.095 and reports about 96.7%.
	if got := ConfidencePercent(-0.095); math.Abs(got-96.675) > 0.01 {
		t.Errorf("ConfidencePercent(-0.095) = %v; want ~96.68", got)
	}
	// Certainty 0 is a perfect 100, and anything below -100/35 clips to 0.
	if got := ConfidencePercent(0); got != 100 {
		t.Errorf("ConfidencePercent(0) = %v; want 100", got)
	}
	if got := ConfidencePercent(-10); got != 0 {
		t.Errorf("ConfidencePercent(-10) = %v; want 0", got)
	}
}

func TestSplitWordsAtSpaces(t *testing.T) {
	syms := []Scored{
		{Symbol: Symbol{Text: "h", UnicharID: 3}, Certainty: -0.2},
		{Symbol: Symbol{Text: "i", UnicharID: 4}, Certainty: -0.1},
		{Symbol: Symbol{Text: " ", UnicharID: 0}, Certainty: -0.9},
		{Symbol: Symbol{Text: "t", UnicharID: 5}, Certainty: -0.3},
		{Symbol: Symbol{Text: "o", UnicharID: 6}, Certainty: -0.4},
	}
	groups := splitWords(syms)
	if len(groups) != 2 {
		t.Fatalf("splitWords returned %d groups; want 2", len(groups))
	}
	if groups[0].start != 0 || groups[0].end != 2 {
		t.Errorf("first group = [%d,%d); want [0,2)", groups[0].start, groups[0].end)
	}
	if groups[1].start != 3 || groups[1].end != 5 {
		t.Errorf("second group = [%d,%d); want [3,5)", groups[1].start, groups[1].end)
	}
	// The space's certainty bounds both neighbours.
	if math.Abs(groups[0].spaceCert-(-0.9)) > 1e-12 {
		t.Errorf("first group space certainty = %v; want -0.9", groups[0].spaceCert)
	}
}

func TestWordBoundsScaleFromTimesteps(t *testing.T) {
	// bounds [0, 4, 10] with scale 6 and a line box starting at x=100:
	// word 0 spans timesteps [0,4) -> [100, 100+ceil(24)] = [100, 124]
	got := wordBounds([]int{0, 4, 10}, 0, 1, 6.0, image.Rect(100, 50, 400, 90))
	want := image.Rect(100, 50, 124, 90)
	if got != want {
		t.Errorf("wordBounds = %v; want %v", got, want)
	}
}

func TestRecognizeProducesWordsWithGeometry(t *testing.T) {
	r := loadRecognizer(t)
	lines := loadCorpus(t, "h36")
	for _, l := range lines {
		line, err := r.Recognize(l.Image)
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		if line.Text != l.Oracle {
			continue // text accuracy is Task 18's assertion, not this one
		}
		if len(line.Words) == 0 && line.Text != "" {
			t.Errorf("%s: text %q produced no words", l.Name, line.Text)
		}
		joined := make([]string, len(line.Words))
		for i, w := range line.Words {
			joined[i] = w.Text
			if w.Bounds.Empty() {
				t.Errorf("%s: word %q has an empty bounding box", l.Name, w.Text)
			}
			if !w.Bounds.In(line.Bounds) {
				t.Errorf("%s: word %q bounds %v escape the line bounds %v", l.Name, w.Text, w.Bounds, line.Bounds)
			}
			if w.Confidence < 0 || w.Confidence > 100 {
				t.Errorf("%s: word %q confidence %v out of range", l.Name, w.Text, w.Confidence)
			}
		}
		if got := strings.Join(joined, " "); got != strings.Join(strings.Fields(line.Text), " ") {
			t.Errorf("%s: words %q do not reconstruct the line %q", l.Name, got, line.Text)
		}
		// Words must be left to right and non-overlapping.
		for i := 1; i < len(line.Words); i++ {
			if line.Words[i].Bounds.Min.X < line.Words[i-1].Bounds.Max.X {
				t.Errorf("%s: word %d overlaps word %d", l.Name, i, i-1)
			}
		}
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/recog/ -run 'TestConfidence|TestSplitWords|TestWordBounds|TestRecognize' -v`
Expected: FAIL — `undefined: ConfidencePercent`.

- [ ] **Step 3: Implement**

Create `internal/recog/words.go`:

```go
// This file is a Go translation of RecodeBeamSearch::ExtractBestPathAsWords and
// ::InitializeWord in src/lstm/recodebeam.cpp, Tesseract::SearchWords in
// src/ccmain/linerec.cpp, and LTRResultIterator::Confidence in
// src/ccmain/ltrresultiterator.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"image"
	"math"
	"strings"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// Word is one recognized word with its geometry and confidence.
type Word struct {
	Text       string
	Bounds     image.Rectangle
	Confidence float64
}

// Line is one recognized line.
type Line struct {
	Text       string
	Words      []Word
	Bounds     image.Rectangle
	Confidence float64
}

// ConfidencePercent is LTRResultIterator::Confidence composed with
// Tesseract::SearchWords' kCertaintyScale: clip(100 + 5*7*certainty, 0, 100).
func ConfidencePercent(certainty float64) float64 {
	c := 100 + 5*CertaintyScale*certainty
	if c < 0 {
		return 0
	}
	if c > 100 {
		return 100
	}
	return c
}

// wordGroup is a run of symbols between spaces.
type wordGroup struct {
	start, end int
	spaceCert  float64
}

// splitWords cuts at UNICHAR_SPACE. Tesseract additionally cuts at a
// start_of_word node and, under TOP_CHOICE_PERM, at every character of a
// non-space-delimited script; the first only exists in the dictionary path
// (Task 23) and the second cannot occur while the recoder is restricted to
// Latin single-code models.
func splitWords(syms []Scored) []wordGroup {
	var groups []wordGroup
	i := 0
	prevSpaceCert := 0.0
	for i < len(syms) {
		if syms[i].UnicharID == tessdata.UnicharSpace {
			prevSpaceCert = syms[i].Certainty
			i++
			continue
		}
		start := i
		for i < len(syms) && syms[i].UnicharID != tessdata.UnicharSpace {
			i++
		}
		// The word's space certainty is the lesser of the spaces flanking it.
		spaceCert := prevSpaceCert
		if i < len(syms) && syms[i].Certainty < spaceCert {
			spaceCert = syms[i].Certainty
		}
		groups = append(groups, wordGroup{start: start, end: i, spaceCert: spaceCert})
	}
	return groups
}

// wordBounds converts a timestep span to page coordinates. Y is the line
// strip's own extent: InitializeWord gives every word the full line height,
// and nothing in the network output constrains it further.
func wordBounds(bounds []int, start, end int, scale float64, lineBox image.Rectangle) image.Rectangle {
	left := lineBox.Min.X + int(math.Floor(float64(bounds[start])*scale))
	right := lineBox.Min.X + int(math.Ceil(float64(bounds[end])*scale))
	if right <= left {
		right = left + 1
	}
	return image.Rect(left, lineBox.Min.Y, right, lineBox.Max.Y)
}

// Recognize transcribes one cropped line image.
func (r *Recognizer) Recognize(img image.Image) (Line, error) {
	out, norm, err := r.Forward(img)
	if err != nil {
		return Line{}, err
	}
	syms, err := r.GreedyDecode(out)
	if err != nil {
		return Line{}, err
	}
	// The greedy path has no lexicon, so every character takes the dict ratio.
	scored := ScoreSymbols(out, syms, DictRatio)
	bounds := CharBoundaries(scored, out.Map.Len())

	lineBox := img.Bounds()
	line := Line{Bounds: lineBox, Confidence: 100}

	var text strings.Builder
	for _, s := range scored {
		text.WriteString(s.Text)
	}
	line.Text = text.String()

	for _, g := range splitWords(scored) {
		cert := 0.0
		var wt strings.Builder
		for i := g.start; i < g.end; i++ {
			wt.WriteString(scored[i].Text)
			if scored[i].Certainty < cert {
				cert = scored[i].Certainty
			}
		}
		if g.spaceCert < cert {
			cert = g.spaceCert
		}
		line.Words = append(line.Words, Word{
			Text:       wt.String(),
			Bounds:     wordBounds(bounds, g.start, g.end, norm.ScaleFactor, lineBox),
			Confidence: ConfidencePercent(cert),
		})
		if c := ConfidencePercent(cert); c < line.Confidence {
			line.Confidence = c
		}
	}
	if len(line.Words) == 0 {
		line.Confidence = 0
	}
	return line, nil
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/recog/ -v`
Expected: PASS.

- [ ] **Step 5: Sanity-check the boxes visually, once**

Numbers that are in range can still be systematically shifted. Render the boxes
onto the line crop and look at one:

Add this as a throwaway test — it is deleted before the commit, so it needs no
`//go:build ignore` and no separate module:

```go
// internal/recog/boxes_eyeball_test.go — DELETE after looking at the output.
package recog

import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"testing"
)

func TestEyeballWordBoxes(t *testing.T) {
	r := loadRecognizer(t)
	// 0003 is the invoice line: digits, punctuation, a currency symbol.
	src := loadCorpus(t, "h36")[2]
	line, err := r.Recognize(src.Image)
	if err != nil {
		t.Fatalf("Recognize() error = %v", err)
	}
	b := src.Image.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, src.Image, b.Min, draw.Src)
	red := color.RGBA{R: 255, A: 255}
	for _, w := range line.Words {
		for x := w.Bounds.Min.X; x < w.Bounds.Max.X; x++ {
			out.Set(x, w.Bounds.Min.Y, red)
			out.Set(x, w.Bounds.Max.Y-1, red)
		}
		for y := w.Bounds.Min.Y; y < w.Bounds.Max.Y; y++ {
			out.Set(w.Bounds.Min.X, y, red)
			out.Set(w.Bounds.Max.X-1, y, red)
		}
		t.Logf("%-20q %v conf %.1f", w.Text, w.Bounds, w.Confidence)
	}
	f, err := os.Create("/tmp/cadmus-boxes.png")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, out); err != nil {
		t.Fatal(err)
	}
	t.Log("wrote /tmp/cadmus-boxes.png")
}
```

```bash
go test ./internal/recog/ -run TestEyeballWordBoxes -v
open /tmp/cadmus-boxes.png
```

**Each box should sit over its word with at most a few pixels of slack on either
side.** A uniform horizontal offset means `ScaleFactor` is wrong (check `XScale`
via Task 1 Step 2); boxes roughly half the right width mean the `bounds` indices
are off by the `CharBoundaries` midpoint rule. Delete the file before committing.

```bash
bd update cad-l1b --append-notes "Word boxes eyeballed on h36/0003.png: <aligned / offset by N px>."
```

- [ ] **Step 6: Commit**

```bash
git add internal/recog/words.go internal/recog/words_test.go
git commit -m "feat(recog): segment words with confidence and bounding boxes"
```

**Stage 1 is now complete: line image in, text out, with per-word confidence and
boxes.** Do not start Task 21 until Task 18's acceptance test passes at the
bound agreed in its Step 5.

---

## Task 21: Extend L1a's DAWG reader with edge-level accessors

**The wire format and the file already exist.** L1a Task 7 ships
`internal/tessdata/dawg.go`: magic 42, `unicharset_size`, `num_edges`, then
`num_edges` 64-bit `EDGE_RECORD`s, total `10 + 8*num_edges`; the
`flag_start_bit = CeilLog2(unicharset_size)` bit-length trap; and word-dawg
`Contains`/`HasPrefix`. It deliberately stops there, and its own comment names
this plan as the owner of what comes next:

> *"L1a's `Dawg` carries no `DawgType` because nothing in L1a needs one … The
> pattern-dawg machinery (`DawgPosition`, `Dict::LetterIsOkay`) is L2b's
> `internal/dict`, and that is where the type tag belongs."*

So this task adds four accessors to the existing file and nothing else. The
type tag goes in `internal/dict` (Task 22), where the punctuation/number
semantics that need it live.

**Files:**
- Modify: `internal/tessdata/dawg.go`
- Modify: `internal/tessdata/dawg_test.go`

**Interfaces:**
- Consumes: L1a's `Dawg` and its unexported `edges`, `letterMask`,
  `flagStartBit`, `nextNodeStartBit`.
- Produces, added to `tessdata.Dawg`:

```go
// NoEdge is Tesseract's NO_EDGE.
const NoEdge = int64(-1)

// PatternUnicharID is Dawg::kPatternUnicharID, the wildcard inside the
// punctuation and number dawgs meaning "a word/digit run goes here". It
// collides numerically with UnicharSpace, deliberately.
const PatternUnicharID = 0

func (d *Dawg) UnicharID(edge int64) int
func (d *Dawg) NextNode(edge int64) int64
func (d *Dawg) EndOfWord(edge int64) bool
func (d *Dawg) EdgeChar(node int64, unicharID int, wordEnd bool) int64
```

- [ ] **Step 1: Write the failing test**

Append to `internal/tessdata/dawg_test.go`. L1a's test file already has a
synthetic dawg builder and `loadRealDawgs`; **reuse both rather than declaring
new ones**, and if the builder's name is not `buildSyntheticDawg`, use whatever
L1a called it — do not add a second builder:

```bash
go doc -all ./internal/tessdata 2>/dev/null | grep -i dawg
grep -n 'func build.*[Dd]awg\|func loadRealDawgs' internal/tessdata/dawg_test.go
```

```go
func TestDawgEdgeAccessors(t *testing.T) {
	// L1a's synthetic fixture spells the single word [1, 2] as two edges:
	//   edge 0: node 0, unichar 1, last, not end-of-word, next_node 1
	//   edge 1: node 1, unichar 2, last, end-of-word,     next_node 0
	d := buildSyntheticDawg(t, 112)

	e := d.EdgeChar(0, 1, false)
	if e != 0 {
		t.Fatalf("EdgeChar(0, 1, false) = %d; want edge 0", e)
	}
	if got := d.UnicharID(e); got != 1 {
		t.Errorf("UnicharID(0) = %d; want 1", got)
	}
	if got := d.NextNode(e); got != 1 {
		t.Errorf("NextNode(0) = %d; want 1", got)
	}
	if d.EndOfWord(e) {
		t.Error("edge 0 is flagged end-of-word; it should not be")
	}

	e2 := d.EdgeChar(1, 2, true)
	if e2 != 1 {
		t.Fatalf("EdgeChar(1, 2, true) = %d; want edge 1", e2)
	}
	if !d.EndOfWord(e2) {
		t.Error("edge 1 is not flagged end-of-word")
	}
	if d.EdgeChar(0, 99, false) != NoEdge {
		t.Error("EdgeChar for an absent id did not return NoEdge")
	}
	if d.EdgeChar(-1, 1, false) != NoEdge || d.EdgeChar(int64(d.NumEdges()), 1, false) != NoEdge {
		t.Error("EdgeChar with an out-of-range node did not return NoEdge")
	}
}

// SquishedDawg::edge_char_of guards its linear scan with edge_occupied(edge),
// which is `edges_[edge] != next_node_mask_`. An empty slot has unichar_id 0 —
// the same value as kPatternUnicharID — so without the guard a punctuation-dawg
// wildcard probe can match a hole and NextNode then returns a huge index.
func TestEdgeCharSkipsAnEmptySlot(t *testing.T) {
	d := buildSyntheticDawg(t, 112)
	d.setEdgeForTest(0, d.emptySlotForTest()) // test-only helper, see Step 3
	if got := d.EdgeChar(0, PatternUnicharID, false); got != NoEdge {
		t.Errorf("EdgeChar over an empty slot = %d; want NoEdge", got)
	}
}

// The structural assumptions EdgeChar's linear scan relies on, asserted on the
// shipped model rather than assumed. If any of these fails on a future model,
// EdgeChar must grow the binary search and the direction check that
// edge_char_of and read_squished_dawg have.
func TestRealDawgsSatisfyTheLinearScanPreconditions(t *testing.T) {
	punc, word, number, charset := loadRealDawgs(t)
	for name, d := range map[string]*Dawg{"punc": punc, "word": word, "number": number} {
		var empty, backward int
		for e := range int64(d.NumEdges()) {
			if d.isEmptySlotForTest(e) {
				empty++
				continue
			}
			if !d.isForwardForTest(e) {
				backward++
			}
			if d.NextNode(e) >= int64(d.NumEdges()) {
				t.Fatalf("%s: edge %d has next_node %d, past the end", name, e, d.NextNode(e))
			}
			if d.UnicharID(e) >= charset.Size() {
				t.Fatalf("%s: edge %d has unichar id %d, past the charset", name, e, d.UnicharID(e))
			}
		}
		if empty != 0 || backward != 0 {
			t.Errorf("%s: %d empty slots, %d backward edges; the linear scan assumes zero of each", name, empty, backward)
		}
		// Node 0's run must have no duplicate unichar ids, which is what makes a
		// linear first-match equivalent to edge_char_of's binary search there.
		seen := map[int]bool{}
		for e := int64(0); e < int64(d.NumEdges()); e++ {
			id := d.UnicharID(e)
			if seen[id] {
				t.Errorf("%s: root has a duplicate unichar id %d; linear scan and binary search can disagree", name, id)
			}
			seen[id] = true
			if d.lastEdge(e) {
				t.Logf("%s: root has %d edges", name, e+1)
				break
			}
		}
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/tessdata/ -run 'TestDawgEdge|TestEdgeChar|TestRealDawgs' -v`
Expected: FAIL — `d.EdgeChar undefined`.

- [ ] **Step 3: Implement**

Append to `internal/tessdata/dawg.go`:

```go
// Edge flag bits, from Dawg::init in src/dict/dawg.cpp. They sit immediately
// above the unichar id, at flagStartBit.
const (
	markerFlag   = 1 // last edge of this node's run
	directionBit = 2 // 1 for a backward edge; a written file has none
	werdEndFlag  = 4 // this edge terminates a word
)

// NoEdge is Tesseract's NO_EDGE.
const NoEdge = int64(-1)

// PatternUnicharID is Dawg::kPatternUnicharID: inside a punctuation or number
// dawg, unichar id 0 on an edge is a wildcard, not the space character. The
// collision with UnicharSpace is deliberate in Tesseract.
const PatternUnicharID = 0

func (d *Dawg) UnicharID(edge int64) int { return int(d.edges[edge] & d.letterMask) }

func (d *Dawg) NextNode(edge int64) int64 {
	return int64(d.edges[edge] >> d.nextNodeStartBit)
}

func (d *Dawg) EndOfWord(edge int64) bool {
	return d.edges[edge]&(werdEndFlag<<d.flagStartBit) != 0
}

func (d *Dawg) lastEdge(edge int64) bool {
	return d.edges[edge]&(markerFlag<<d.flagStartBit) != 0
}

// emptySlot is edge_occupied's sentinel: an edge record equal to
// next_node_mask_ is a hole left by the squishing pass, not an edge.
func (d *Dawg) emptySlot() uint64 { return ^uint64(0) << d.nextNodeStartBit }

// EdgeChar is SquishedDawg::edge_char_of: the edge leaving node on unicharID,
// or NoEdge.
//
// Two divergences from the C++, both deliberate and both bounded by
// TestRealDawgsSatisfyTheLinearScanPreconditions:
//
//   - Node 0 is scanned linearly rather than binary-searched. The two agree
//     whenever the root's unichar ids are distinct, which the test asserts on
//     every shipped dawg; eng's roots are 8, 67 and 40 edges.
//   - The direction bit is not tested, because edge_char_of does not test it
//     either — read_squished_dawg's validation loop does, at load time, and a
//     written file contains only forward edges.
//
// The empty-slot guard IS reproduced, because edge_char_of has it
// (`if (edge != NO_EDGE && edge_occupied(edge))`) and an empty slot's unichar id
// is 0, which is exactly the wildcard a punctuation-dawg probe asks for.
func (d *Dawg) EdgeChar(node int64, unicharID int, wordEnd bool) int64 {
	if node < 0 || node >= int64(len(d.edges)) {
		return NoEdge
	}
	if d.edges[node] == d.emptySlot() {
		return NoEdge
	}
	for e := node; e < int64(len(d.edges)); e++ {
		if d.UnicharID(e) == unicharID && (!wordEnd || d.EndOfWord(e)) {
			return e
		}
		if d.lastEdge(e) {
			break
		}
	}
	return NoEdge
}
```

`NextNode` needs no mask: `nextNodeStartBit` is `flagStartBit + 3`, so a plain
right shift already drops the id and the three flags. **Do not cache a
"node 0 edge count" field** — nothing reads it, and a written-but-never-read
field is dead code.

The three `*ForTest` helpers the tests use go in `dawg_test.go`, not the
production file:

```go
func (d *Dawg) emptySlotForTest() uint64       { return d.emptySlot() }
func (d *Dawg) setEdgeForTest(i int, v uint64) { d.edges[i] = v }
func (d *Dawg) isEmptySlotForTest(e int64) bool { return d.edges[e] == d.emptySlot() }
func (d *Dawg) isForwardForTest(e int64) bool {
	return d.edges[e]&(directionBit<<d.flagStartBit) == 0
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/tessdata/ -v`
Expected: PASS. `TestRealDawgsSatisfyTheLinearScanPreconditions` walks 461,848
edges for the word dawg alone; it takes a second, and it logs the three root
sizes (expect 8, 67 and 40).

- [ ] **Step 5: Cross-check the edge counts with the oracle**

```bash
combine_tessdata -u testdata/eng.traineddata /tmp/eng.
for f in lstm-punc-dawg lstm-word-dawg lstm-number-dawg; do
  bytes=$(wc -c < "/tmp/eng.$f")
  echo "$f: $bytes bytes -> $(( (bytes - 10) / 8 )) edges"
done
```

Expected: 539, 461848, 591 — the same numbers L1a Task 7 asserts. If the
component sizes differ, the model changed; update both plans' tests to the new
figures rather than relaxing either.

- [ ] **Step 6: Commit**

```bash
git add internal/tessdata/dawg.go internal/tessdata/dawg_test.go
git commit -m "feat(tessdata): add dawg edge accessors for the lexicon walk"
```

---

## Task 22: The dictionary position machinery

The beam does not walk a dawg directly. Each hypothesis carries a *set* of
`DawgPosition`s — the active lexicon frontier — and extends it one unichar at a
time through `Dict::def_letter_is_okay`, which returns a permuter (or `NoPerm`
if the letter kills every position).

Three dawgs are loaded, in this order, each with a permuter value:

| component | type | permuter |
|---|---|---|
| 18 `lstm-punc-dawg` | punctuation | `PuncPerm` = 1 |
| 19 `lstm-word-dawg` | word | `SystemDawgPerm` = 8 |
| 20 `lstm-number-dawg` | number | `NumberPerm` = 6 |

The punctuation dawg **wraps** the word and number dawgs: `kDawgSuccessors`
allows punctuation → word and punctuation → number, and word/number → punctuation.
`kPatternUnicharID = 0` inside the punctuation dawg is the wildcard meaning "a
word goes here", and it collides numerically with `UnicharSpace` deliberately.
In the number dawg every digit is folded to the pattern id before lookup.

**Files:**
- Create: `internal/dict/dict.go`
- Test: `internal/dict/dict_test.go`

**Interfaces:**
- Produces:

```go
// DawgType lives here, not in internal/tessdata: it exists only to drive
// char_for_dawg's digit folding and the successor table, both of which are this
// package's concern. There is deliberately NO DawgPattern value — see Step 1.
type DawgType int

const (
	DawgPunctuation DawgType = iota
	DawgWord
	DawgNumber
)

type PermuterType int
const (
	NoPerm         PermuterType = 0
	PuncPerm       PermuterType = 1
	NumberPerm     PermuterType = 6
	SystemDawgPerm PermuterType = 8
)

// Position is Tesseract's DawgPosition. A dawgIndex or puncIndex of -1 means
// "not in that dawg"; a ref of NoEdge means "at the start".
type Position struct {
	DawgIndex  int
	DawgRef    int64
	PuncIndex  int
	PuncRef    int64
	BackToPunc bool
}

type Dict struct{ /* dawgs []*tessdata.Dawg, types []DawgType, charset *tessdata.Unicharset */ }

func New(charset *tessdata.Unicharset, punc, word, number *tessdata.Dawg) *Dict

// Start returns the initial position set for a word.
func (d *Dict) Start() []Position

// LetterIsOkay extends active by one unichar.
//
// prev is the permuter carried in from the PREVIOUS letter of the same word.
// def_letter_is_okay's `dawg_args->permuter` is an in/out parameter and its
// final block only overwrites it under three conditions; dropping that carry
// is why "(the)" would report PuncPerm at the closing paren instead of
// SystemDawgPerm. Callers pass NoPerm for the first letter.
//
// It returns the permuter after the carry rule, the updated position set, and
// whether the word may validly end here.
func (d *Dict) LetterIsOkay(prev PermuterType, active []Position, unicharID int, wordEnd bool) (PermuterType, []Position, bool)
```

- [ ] **Step 1: Read the source before writing any of it**

`def_letter_is_okay` is 120 lines of branching over a five-field position record
and it is the one function in this plan most likely to be transcribed subtly
wrong. Read it, and the two helpers it leans on, before implementing:

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
sed -n '/int Dict::def_letter_is_okay/,/^}/p'  "$SP/tess/src/dict/dict.cpp"
sed -n '/void Dict::default_dawgs/,/^}/p'      "$SP/tess/src/dict/dict.cpp"
grep -n 'GetStartingNode\|char_for_dawg\|kDawgSuccessors' "$SP/tess/src/dict/dict.h"
```

Take notes on seven branches specifically, because the Go below encodes them and
**the C++ is the specification**:

1. `dawg == nil` (in the punctuation dawg, no core dawg chosen yet) — the
   pattern-transition branch, then the plain punctuation-extension branch.
2. `punc != nil && dawg.EndOfWord(pos.DawgRef)` — returning to punctuation.
3. `pos.BackToPunc` — `continue`, no further extension.
4. The `wordEnd && punc != nil && !punc.EndOfWord(pos.PuncRef)` rejection.
5. `GetStartingNode(dawg, ref)`: `ref == NoEdge` gives node 0, otherwise
   `dawg.NextNode(ref)`.
6. **The final permuter-carry block**, which the Go below reproduces verbatim:

   ```cpp
   if (dawg_args->permuter == NO_PERM || curr_perm == NO_PERM ||
       (curr_perm != PUNC_PERM && dawg_args->permuter != COMPOUND_PERM)) {
     dawg_args->permuter = curr_perm;
   }
   return dawg_args->permuter;   // NOT curr_perm
   ```

   `dawg_args->permuter` is in/out and survives across letters. When
   `curr_perm == PUNC_PERM` the *previous* permuter is kept, which is how
   `"(the)"` still reports `SYSTEM_DAWG_PERM` at the closing paren. Task 23
   branches on `best_nodes[index]->permuter` for word splitting and passes it to
   `FakeWordFromRatings`, so losing this carry surfaces as wrong word boundaries
   and wrong confidence, not as a lexicon miss.

   `COMPOUND_PERM` never occurs on the LSTM path — nothing sets it — so the
   `!= COMPOUND_PERM` conjunct is always true and can be dropped, but say so in
   a comment rather than silently omitting it.
7. **The `DAWG_TYPE_PATTERN` branch**, immediately after the `back_to_punc`
   check:

   ```cpp
   if (dawg->type() == DAWG_TYPE_PATTERN) {
     ProcessPatternEdges(dawg, pos, unichar_id, word_end, dawg_args, &curr_perm);
     continue;
   }
   ```

   **L1b does not implement it, and therefore does not declare a
   `DawgPattern`.** Pattern dawgs come from `user-patterns`, loaded through a
   separate API; no `.traineddata` LSTM component is one. Confirm that before
   relying on it:

   ```bash
   combine_tessdata -d testdata/eng.traineddata
   ```

   Expect components 18/19/20 (`lstm-punc-dawg`, `lstm-word-dawg`,
   `lstm-number-dawg`) and nothing pattern-shaped. A declared-but-unhandled enum
   value would be worse than no value at all: `New` takes exactly three dawgs
   and assigns their types itself, so an unhandled type is unreachable by
   construction.

**If the Go below disagrees with the source on any of these, the source wins.**

- [ ] **Step 2: Write the failing test**

Create `internal/dict/dict_test.go`:

```go
package dict

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func loadDict(t *testing.T) (*Dict, *tessdata.Unicharset) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	csEntry, _ := c.Entry(tessdata.TypeLSTMUnicharset)
	cs, err := tessdata.ParseUnicharset(csEntry)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	// tessdata.ParseDawg takes no type tag (L1a Task 7); Dict assigns the types.
	load := func(typ tessdata.Type) *tessdata.Dawg {
		e, ok := c.Entry(typ)
		if !ok {
			t.Fatalf("no %v component", typ)
		}
		d, err := tessdata.ParseDawg(e, c.Swapped())
		if err != nil {
			t.Fatalf("ParseDawg(%v) error = %v", typ, err)
		}
		return d
	}
	return New(cs,
		load(tessdata.TypeLSTMPuncDawg),
		load(tessdata.TypeLSTMSystemDawg),
		load(tessdata.TypeLSTMNumberDawg)), cs
}

func idsOf(t *testing.T, cs *tessdata.Unicharset, s string) []int {
	t.Helper()
	out := make([]int, 0, len(s))
	for _, r := range s {
		found := -1
		for id := range cs.Size() {
			if cs.Text(id) == string(r) {
				found = id
				break
			}
		}
		if found < 0 {
			t.Fatalf("no unichar for %q", r)
		}
		out = append(out, found)
	}
	return out
}

// walk feeds a whole string through LetterIsOkay and returns the final
// permuter and whether the last position was a valid word end.
//
// The permuter is threaded from letter to letter — that is what makes it an
// in/out parameter in the C++, and it is the whole point of the carry rule.
// Seeding with NoPerm reproduces DawgArgs' initial state.
func walk(t *testing.T, d *Dict, cs *tessdata.Unicharset, s string) (PermuterType, bool) {
	t.Helper()
	active := d.Start()
	ids := idsOf(t, cs, s)
	perm := NoPerm
	var validEnd bool
	for i, id := range ids {
		perm, active, validEnd = d.LetterIsOkay(perm, active, id, i == len(ids)-1)
		if perm == NoPerm {
			return NoPerm, false
		}
	}
	return perm, validEnd
}

func TestLetterIsOkayAcceptsDictionaryWords(t *testing.T) {
	d, cs := loadDict(t)
	for _, w := range []string{"the", "and", "zebra", "hello"} {
		perm, validEnd := walk(t, d, cs, w)
		if perm == NoPerm {
			t.Errorf("%q was rejected outright", w)
			continue
		}
		if !validEnd {
			t.Errorf("%q did not end on a valid word boundary", w)
		}
		if perm != SystemDawgPerm {
			t.Errorf("%q permuter = %d; want SystemDawgPerm (%d)", w, perm, SystemDawgPerm)
		}
	}
}

func TestLetterIsOkayRejectsNonWords(t *testing.T) {
	d, cs := loadDict(t)
	for _, w := range []string{"qqqqq", "zzzxq"} {
		if perm, _ := walk(t, d, cs, w); perm != NoPerm {
			t.Errorf("%q was accepted with permuter %d; want NoPerm", w, perm)
		}
	}
}

// A dictionary word wrapped in punctuation must reach the word dawg through the
// punctuation dawg's kPatternUnicharID transition.
func TestLetterIsOkayHandlesPunctuationWrapping(t *testing.T) {
	d, cs := loadDict(t)
	perm, validEnd := walk(t, d, cs, "(the)")
	if perm == NoPerm {
		t.Fatal("\"(the)\" was rejected; the punctuation dawg does not wrap the word dawg")
	}
	if !validEnd {
		t.Error("\"(the)\" did not end on a valid word boundary")
	}
	// The carry rule is what keeps this SystemDawgPerm rather than PuncPerm:
	// the closing paren's own curr_perm is PUNC_PERM, and the final block
	// therefore preserves the permuter the core word established.
	if perm != SystemDawgPerm {
		t.Errorf("\"(the)\" permuter = %d; want SystemDawgPerm (%d) — the permuter carry was dropped", perm, SystemDawgPerm)
	}
}

// A prefix of a real word is extendable but is not a valid end.
func TestLetterIsOkayPrefixIsNotAValidEnd(t *testing.T) {
	d, cs := loadDict(t)
	active := d.Start()
	ids := idsOf(t, cs, "zebr")
	perm := NoPerm
	var validEnd bool
	for _, id := range ids {
		perm, active, validEnd = d.LetterIsOkay(perm, active, id, false)
		if perm == NoPerm {
			t.Fatal("\"zebr\" was rejected; it is a prefix of \"zebra\"")
		}
	}
	if validEnd {
		t.Error("\"zebr\" reported a valid word end")
	}
	if len(active) == 0 {
		t.Error("\"zebr\" left no active positions to extend")
	}
}

// Digits are folded to kPatternUnicharID inside the number dawg.
func TestLetterIsOkayAcceptsNumbers(t *testing.T) {
	d, cs := loadDict(t)
	if perm, _ := walk(t, d, cs, "2024"); perm == NoPerm {
		t.Error("\"2024\" was rejected; digits must fold to the pattern id in the number dawg")
	}
}

// The pattern id must never be accepted as a literal letter.
func TestLetterIsOkayRejectsThePatternID(t *testing.T) {
	d, _ := loadDict(t)
	if perm, _, _ := d.LetterIsOkay(NoPerm, d.Start(), tessdata.PatternUnicharID, false); perm != NoPerm {
		t.Errorf("kPatternUnicharID was accepted with permuter %d; want NoPerm", perm)
	}
	// And a non-NoPerm carry must not rescue it: the wildcard rejection is
	// unconditional and happens before the carry block.
	if perm, _, _ := d.LetterIsOkay(SystemDawgPerm, d.Start(), tessdata.PatternUnicharID, false); perm != NoPerm {
		t.Errorf("kPatternUnicharID with a carried permuter = %d; want NoPerm", perm)
	}
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `go test ./internal/dict/ -run TestLetterIsOkay -v`
Expected: FAIL — `undefined: New`.

- [ ] **Step 4: Implement**

Create `internal/dict/dict.go`:

```go
// This file is a Go translation of Dict::def_letter_is_okay,
// Dict::default_dawgs, Dict::GetStartingNode and Dict::char_for_dawg in
// src/dict/dict.cpp and src/dict/dict.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package dict

import "github.com/dobbo-ca/cadmus/internal/tessdata"

// DawgType is which lexicon a dawg is, which decides how its unichar ids are
// folded before lookup and which other dawgs may follow it. It lives here
// rather than in internal/tessdata because nothing in the loader needs it.
//
// DAWG_TYPE_PATTERN has no value here on purpose. def_letter_is_okay has a
// branch for it (ProcessPatternEdges), pattern dawgs come from `user-patterns`
// through a separate API, and no .traineddata LSTM component is one — so
// declaring the value without the branch would advertise support that does not
// exist. New() takes exactly three dawgs and assigns these types itself, so an
// unhandled type is unreachable by construction.
type DawgType int

const (
	DawgPunctuation DawgType = iota
	DawgWord
	DawgNumber
)

// PermuterType is Tesseract's PermuterType, restricted to the values the LSTM
// path can produce. Larger is better; def_letter_is_okay takes the maximum over
// the surviving positions and then applies the carry rule in LetterIsOkay.
type PermuterType int

const (
	NoPerm         PermuterType = 0
	PuncPerm       PermuterType = 1
	NumberPerm     PermuterType = 6
	SystemDawgPerm PermuterType = 8
	// TopChoicePerm is what the beam stamps on a non-dictionary hypothesis.
	TopChoicePerm PermuterType = 2
)

// Position is DawgPosition: a point in the lexicon frontier. DawgIndex or
// PuncIndex of -1 means "not in that dawg", and a ref of NoEdge means "at the
// start of it".
type Position struct {
	DawgIndex  int
	DawgRef    int64
	PuncIndex  int
	PuncRef    int64
	BackToPunc bool
}

// Dict is the LSTM lexicon: the punctuation dawg wrapping the word and number
// dawgs, exactly as Dict::LoadLSTM assembles it.
type Dict struct {
	charset *tessdata.Unicharset
	dawgs   []*tessdata.Dawg // index 0 punctuation, 1 word, 2 number
	types   []DawgType
	perms   []PermuterType
}

const (
	puncIdx   = 0
	wordIdx   = 1
	numberIdx = 2
)

func New(charset *tessdata.Unicharset, punc, word, number *tessdata.Dawg) *Dict {
	return &Dict{
		charset: charset,
		dawgs:   []*tessdata.Dawg{punc, word, number},
		types:   []DawgType{DawgPunctuation, DawgWord, DawgNumber},
		perms:   []PermuterType{PuncPerm, SystemDawgPerm, NumberPerm},
	}
}

// Start is Dict::default_dawgs. The punctuation dawg seeds a position on its
// own; the word and number dawgs are subsumed by it whenever the punctuation
// dawg has a kPatternUnicharID edge out of its root, which every stock eng
// model does.
func (d *Dict) Start() []Position {
	punc := d.dawgs[puncIdx]
	puncAvailable := punc != nil && punc.EdgeChar(0, tessdata.PatternUnicharID, true) != tessdata.NoEdge

	var out []Position
	for i, dw := range d.dawgs {
		if dw == nil {
			continue
		}
		if i == puncIdx {
			out = append(out, Position{DawgIndex: -1, DawgRef: tessdata.NoEdge, PuncIndex: i, PuncRef: tessdata.NoEdge})
			continue
		}
		// Word and number are successors of punctuation, so they are only
		// seeded independently when punctuation cannot reach them.
		if !puncAvailable {
			out = append(out, Position{DawgIndex: i, DawgRef: tessdata.NoEdge, PuncIndex: -1, PuncRef: tessdata.NoEdge})
		}
	}
	return out
}

// LetterIsOkay is Dict::def_letter_is_okay.
//
// prev is the caller's carried permuter — def_letter_is_okay's in/out
// `dawg_args->permuter`. Pass NoPerm for the first letter of a word and the
// previous return value thereafter.
func (d *Dict) LetterIsOkay(prev PermuterType, active []Position, unicharID int, wordEnd bool) (PermuterType, []Position, bool) {
	// A word may never contain the pattern wildcard; accepting it would let a
	// pattern dawg match arbitrary text. This is before the carry block in the
	// C++ too, and it sets dawg_args->permuter = NO_PERM outright.
	if unicharID == tessdata.PatternUnicharID || unicharID < 0 || unicharID >= d.charset.Size() {
		return NoPerm, nil, false
	}

	curr := NoPerm
	var updated []Position
	validEnd := false

	add := func(p Position) {
		for _, q := range updated {
			if q == p {
				return
			}
		}
		updated = append(updated, p)
	}

	for _, pos := range active {
		var punc, dw *tessdata.Dawg
		if pos.PuncIndex >= 0 {
			punc = d.dawgs[pos.PuncIndex]
		}
		if pos.DawgIndex >= 0 {
			dw = d.dawgs[pos.DawgIndex]
		}
		if punc == nil && dw == nil {
			continue
		}

		if dw == nil {
			// In the punctuation dawg with no core dawg chosen yet.
			puncNode := startingNode(punc, pos.PuncRef)
			if trans := punc.EdgeChar(puncNode, tessdata.PatternUnicharID, wordEnd); trans != tessdata.NoEdge {
				for _, si := range d.successors(pos.PuncIndex) {
					sd := d.dawgs[si]
					ch := d.charForDawg(unicharID, si)
					if e := sd.EdgeChar(0, ch, wordEnd); e != tessdata.NoEdge {
						add(Position{DawgIndex: si, DawgRef: e, PuncIndex: pos.PuncIndex, PuncRef: trans})
						if d.perms[si] > curr {
							curr = d.perms[si]
						}
						if sd.EndOfWord(e) && punc.EndOfWord(trans) {
							validEnd = true
						}
					}
				}
			}
			if e := punc.EdgeChar(puncNode, unicharID, wordEnd); e != tessdata.NoEdge {
				add(Position{DawgIndex: -1, DawgRef: tessdata.NoEdge, PuncIndex: pos.PuncIndex, PuncRef: e})
				if PuncPerm > curr {
					curr = PuncPerm
				}
				if punc.EndOfWord(e) {
					validEnd = true
				}
			}
			continue
		}

		if punc != nil && pos.DawgRef != tessdata.NoEdge && dw.EndOfWord(pos.DawgRef) {
			// The core word can end here; see whether punctuation continues.
			puncNode := startingNode(punc, pos.PuncRef)
			if e := punc.EdgeChar(puncNode, unicharID, wordEnd); e != tessdata.NoEdge {
				add(Position{DawgIndex: pos.DawgIndex, DawgRef: pos.DawgRef, PuncIndex: pos.PuncIndex, PuncRef: e, BackToPunc: true})
				if d.perms[pos.DawgIndex] > curr {
					curr = d.perms[pos.DawgIndex]
				}
				if punc.EndOfWord(e) {
					validEnd = true
				}
			}
		}

		if pos.BackToPunc {
			continue
		}

		// DAWG_TYPE_PATTERN would be handled here, before the edge lookup.
		// L1b declares no such type; see Step 1 branch 7.

		node := startingNode(dw, pos.DawgRef)
		e := tessdata.NoEdge
		if node != tessdata.NoEdge {
			e = dw.EdgeChar(node, d.charForDawg(unicharID, pos.DawgIndex), wordEnd)
		}
		if e == tessdata.NoEdge {
			continue
		}
		if wordEnd && punc != nil && pos.PuncRef != tessdata.NoEdge && !punc.EndOfWord(pos.PuncRef) {
			// The punctuation constraint is not satisfied at the end of a word.
			continue
		}
		if d.perms[pos.DawgIndex] > curr {
			curr = d.perms[pos.DawgIndex]
		}
		if dw.EndOfWord(e) && (punc == nil || pos.PuncRef == tessdata.NoEdge || punc.EndOfWord(pos.PuncRef)) {
			validEnd = true
		}
		add(Position{DawgIndex: pos.DawgIndex, DawgRef: e, PuncIndex: pos.PuncIndex, PuncRef: pos.PuncRef})
	}

	// The permuter carry, verbatim from the tail of def_letter_is_okay:
	//
	//	if (dawg_args->permuter == NO_PERM || curr_perm == NO_PERM ||
	//	    (curr_perm != PUNC_PERM && dawg_args->permuter != COMPOUND_PERM)) {
	//	  dawg_args->permuter = curr_perm;
	//	}
	//	return dawg_args->permuter;
	//
	// COMPOUND_PERM is unreachable on the LSTM path — nothing assigns it — so
	// that conjunct is always true and is omitted. What remains is the case that
	// matters: when this letter's own best permuter is PUNC_PERM and a core word
	// was already established, the OLD permuter is preserved. That is why
	// "(the)" reports SystemDawgPerm at the closing paren rather than PuncPerm,
	// and Task 23 branches on exactly that value.
	out := prev
	if prev == NoPerm || curr == NoPerm || curr != PuncPerm {
		out = curr
	}
	if out == NoPerm {
		return NoPerm, nil, false
	}
	return out, updated, validEnd
}

// successors is kDawgSuccessors: punctuation may be followed by the word and
// number dawgs, and those two only by punctuation.
func (d *Dict) successors(index int) []int {
	if index == puncIdx {
		var out []int
		for _, i := range []int{wordIdx, numberIdx} {
			if d.dawgs[i] != nil {
				out = append(out, i)
			}
		}
		return out
	}
	return nil
}

// charForDawg is Dict::char_for_dawg: inside the number dawg every digit folds
// to the pattern wildcard, so "2024" and "1999" share one path. The type comes
// from this package's own table, keyed on the dawg's index, because
// tessdata.Dawg carries no type tag.
func (d *Dict) charForDawg(unicharID, dawgIndex int) int {
	if dawgIndex < 0 || dawgIndex >= len(d.types) || d.types[dawgIndex] != DawgNumber {
		return unicharID
	}
	ch, ok := d.charset.Char(unicharID)
	if ok && ch.Properties&tessdata.PropDigit != 0 {
		return tessdata.PatternUnicharID
	}
	return unicharID
}

// startingNode is Dict::GetStartingNode: a position with no edge yet starts at
// the root, otherwise it continues from the edge's successor.
func startingNode(dw *tessdata.Dawg, ref int64) int64 {
	if ref == tessdata.NoEdge {
		return 0
	}
	return dw.NextNode(ref)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `go test ./internal/dict/ -v`
Expected: PASS.

**If `TestLetterIsOkayHandlesPunctuationWrapping` fails but the plain word tests
pass**, the pattern-transition branch is wrong — re-read branch 1 from Step 1.
**If it accepts `"(the)"` but reports `PuncPerm` instead of `SystemDawgPerm`**,
the permuter carry (branch 6) was dropped or `curr` is being returned in place of
the carried value.
**If prefixes are accepted as valid ends**, the `EndOfWord` checks are being
applied to the wrong ref.

- [ ] **Step 6: Commit**

```bash
git add internal/dict/dict.go internal/dict/dict_test.go
git commit -m "feat(dict): add the dawg position machinery"
```

---

## Task 23: RecodeBeamSearch — Stage 2

The dictionary is a **weight, not a constraint**. There are always two parallel
worlds, `isDawg=0` and `isDawg=1`, competing on score; a dead dictionary prefix
simply fails to extend the dawg beam while the non-dawg beam carries on. The
entire dictionary influence is one multiplication: a non-dictionary node's
certainty is scaled by `kDictRatio = 2.25`, and since certainties are negative,
that makes it 2.25 times worse.

Because L1b restricts the recoder to single-code entries (Task 16), the beam's
partial-code dimension collapses: `kNumLengths` is effectively 1 and there are
`2 * NC_COUNT = 6` heaps per timestep, each of width `kBeamWidths[0] = 5`, plus
one `bestInitialDawg` slot per continuation.

`NodeContinuation` exists so score merging is safe:

- `NcAnything` — the node used only its own score; anything may follow.
- `NcOnlyDup` — the node added a neighbour's probability without a preceding
  stand-alone, so it **must** be followed by a stand-alone duplicate of itself.
- `NcNoDup` — the node combined scores after a stand-alone, so it may **not** be
  followed by a duplicate of itself.

The score merging is why Tesseract's certainties beat greedy's: at a class
transition a node may claim `P(code) + P(blank)`, and in the top-2 X→Y case also
`P(prev)`. Their worked example turns a `log(0.55)` certainty into `log(0.95)` —
through the `100 + 35*c` export formula, the difference between reporting 79%
and 98% for the same word.

**Files:**
- Create: `internal/recog/beam.go`, `internal/recog/beam_test.go`
- Modify: `internal/recog/decode.go` — add `Recognizer.Dict` and the three dawg
  loads to `NewRecognizer`, and rewire `Recognize` from `GreedyDecode` +
  `ScoreSymbols` to `BeamDecode` (Step 4 spells both out)
- Modify: `internal/recog/decode_test.go` — `loadRecognizer` now returns a
  recognizer with a non-nil `Dict`

**Interfaces:**
- Produces:

```go

// BeamDecode is RecodeBeamSearch::Decode followed by ExtractBestPaths and
// ExtractPathAsUnicharIds. It returns the same Scored slice the greedy path
// produces, so Task 20's word segmentation is reused unchanged.
func (r *Recognizer) BeamDecode(out *nn.Tensor) ([]Scored, error)
```

- [ ] **Step 1: Transcribe the search from the source, line by line**

`ContinueContext` is the most intricate function in this plan and the research
that described it read it once. **Before writing Go, read these and check the
plan's description against them:**

```bash
SP=/private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad
sed -n '37,81p'    "$SP/tess/src/lstm/recodebeam.h"    # the NodeContinuation derivation
sed -n '85,140p'   "$SP/tess/src/lstm/recodebeam.cpp"  # Decode / DecodeStep
sed -n '888,1010p' "$SP/tess/src/lstm/recodebeam.cpp"  # ContinueContext
sed -n '1090,1180p' "$SP/tess/src/lstm/recodebeam.cpp" # ContinueDawg / ContinueUnichar
sed -n '200,260p'  "$SP/tess/src/lstm/recodebeam.cpp"  # ExtractBestPaths
```

Write the transcription into `beam.go` as a comment block first, then the code
under it. **Where this plan and the source disagree, the source wins**; note
every such correction in the task report, because those are the places the
research was wrong and the next reader needs to know.

Confirm in particular:

- `ComputeTopN` forces `topNFlags[nullChar] = TN_TOP2` unconditionally.
- `DecodeStep` sweeps `TN_TOP2`, then `TN_TOPN`, then `TN_ALSO_RAN`, stopping as
  soon as any `NC_ANYTHING` beam is non-empty.
- The min-certainty floor `kMinCertainty` is skipped for `nullChar`.
- `ContinueDawg` is gated on `cert > worstDictCert`, with
  `worstDictCert = kWorstDictCertainty / kCertaintyScale = -25/7`.
- `ExtractBestPaths` skips `NC_ONLY_DUP` as a line ending, and a dawg
  hypothesis may only finish the line if its last real node has `endOfWord` set
  or is a space.
- **`ContinueUnichar` stamps `NO_PERM` on exactly one push** — the
  `PushInitialDawgIfBetter` for `UNICHAR_SPACE`, inside `if (dict_ != nullptr …)`
  — and it does *not* multiply that node's certainty by `dict_ratio`. Every other
  non-dawg push is `TOP_CHOICE_PERM` at `cert * dict_ratio`.
- **Both `UNICHAR_SPACE` special cases in `ExtractPathAsUnicharIds` belong
  here**, and their gates are opposite (see Task 19 Step 5):
  case (1), before the run loop, moves a space's accumulated nulls onto the
  *previous* character and requires `permuter != NO_PERM`; case (2), inside the
  run loop, makes the space *forget* the preceding nulls and requires
  `permuter == NO_PERM`. Both are unreachable with `Dict == nil`, which is why
  Task 19 omits them and why `TestBeamAgreesWithGreedyOnUnambiguousInput` runs
  with the dictionary off.
- `LetterIsOkay`'s permuter is threaded per hypothesis, not recomputed: each
  `RecodeNode` carries the permuter its predecessor reached, and `continueDawg`
  passes it in as `prev` (Task 22).

- [ ] **Step 2: Write the failing test**

Create `internal/recog/beam_test.go`:

```go
package recog

import (
	"strings"
	"testing"

	"github.com/dobbo-ca/cadmus/internal/nn"
)

// With no dictionary and unambiguous per-timestep distributions, the beam must
// agree with greedy exactly — that is the invariant that says the search
// machinery is not corrupting an already-correct path.
func TestBeamAgreesWithGreedyOnUnambiguousInput(t *testing.T) {
	r := loadRecognizer(t)
	r.Dict = nil
	for _, l := range loadCorpus(t, "h36") {
		out, _, err := r.Forward(l.Image)
		if err != nil {
			t.Fatalf("%s: Forward() error = %v", l.Name, err)
		}
		greedy, err := r.GreedyDecode(out)
		if err != nil {
			t.Fatalf("%s: GreedyDecode() error = %v", l.Name, err)
		}
		beam, err := r.BeamDecode(out)
		if err != nil {
			t.Fatalf("%s: BeamDecode() error = %v", l.Name, err)
		}
		var g, b strings.Builder
		for _, s := range greedy {
			g.WriteString(s.Text)
		}
		for _, s := range beam {
			b.WriteString(s.Text)
		}
		if g.String() != b.String() {
			t.Errorf("%s: dictionary-free beam %q differs from greedy %q", l.Name, b.String(), g.String())
		}
	}
}

// The score merge is the beam's main contribution to confidence: a class
// transition where the probability is split between a code and the blank must
// report a better certainty than greedy's per-timestep minimum.
//
// TWO TIMESTEPS, not one. The merged push in ContinueContext's final_codes loop
// carries continuation NC_ONLY_DUP — "must be followed by a stand-alone
// duplicate of itself" — and ExtractBestPaths refuses NC_ONLY_DUP as a line
// ending. On a width-1 input the merged node therefore cannot be the answer and
// the unmerged NC_ANYTHING node wins, so a width-1 test measures the opposite of
// what it claims to.
//
// The assertion is a BRACKET, not an equality. Which continuation survives to
// the end of a two-step path depends on the heap ordering, and pinning an exact
// value here without measuring it first is how a plan-invented number gets
// "fixed" by loosening the tolerance. The bracket is tight enough to fail
// loudly if the merge is missing and loose enough not to lie.
func TestBeamMergesTransitionScores(t *testing.T) {
	r := loadRecognizer(t)
	r.Dict = nil
	n := r.Net.NumOutputs
	null := r.Net.NullChar
	code := codeFor(t, r, "A")

	out := nn.NewTensor(nn.StrideMap{Height: 1, Width: 2}, n)
	row := make([]float64, n)

	// t=0: the mass is split 0.55 / 0.40 between 'A' and the blank.
	for i := range row {
		row[i] = 0.05 / float64(n-2)
	}
	row[code] = 0.55
	row[null] = 0.40
	out.WriteTimeStep(0, row)

	// t=1: an unambiguous blank, so the path is a single "A" and there is a
	// neighbour for the NC_ONLY_DUP node to resolve against.
	for i := range row {
		row[i] = 0.01 / float64(n-1)
	}
	row[null] = 0.99
	out.WriteTimeStep(1, row)

	beam, err := r.BeamDecode(out)
	if err != nil {
		t.Fatalf("BeamDecode() error = %v", err)
	}
	if len(beam) != 1 || beam[0].Text != "A" {
		t.Fatalf("beam decoded %v; want a single \"A\"", beam)
	}

	cert := func(p float64) float64 { return (ProbToCertainty(float64(float32(p))) + CertOffset) * DictRatio }
	unmerged := cert(0.55) // what greedy's per-timestep minimum would give
	merged := cert(0.95)   // log(0.55 + 0.40), the best the merge can reach
	if beam[0].Certainty <= unmerged+1e-9 {
		t.Errorf("beam certainty %v is no better than the per-timestep minimum %v; the P(code)+P(blank) merge is missing",
			beam[0].Certainty, unmerged)
	}
	if beam[0].Certainty > merged+1e-9 {
		t.Errorf("beam certainty %v is better than log(0.55+0.40) = %v; the merge is double-counting a timestep",
			beam[0].Certainty, merged)
	}
	t.Logf("merged certainty %v, bracket (%v, %v]", beam[0].Certainty, unmerged, merged)
}

// With the dictionary on, a dictionary word must not be scaled by the dict
// ratio and so must report a better confidence than the same characters would
// as a non-word.
func TestBeamPrefersTheDictionaryPath(t *testing.T) {
	r := loadRecognizer(t)
	if r.Dict == nil {
		t.Skip("model has no dawg components")
	}
	lines := loadCorpus(t, "h36")
	var found bool
	for _, l := range lines {
		if !strings.Contains(l.Oracle, "the ") {
			continue
		}
		found = true
		withDict, err := r.Recognize(l.Image)
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		saved := r.Dict
		r.Dict = nil
		withoutDict, err := r.Recognize(l.Image)
		r.Dict = saved
		if err != nil {
			t.Fatalf("%s: Recognize() error = %v", l.Name, err)
		}
		if withDict.Confidence < withoutDict.Confidence {
			t.Errorf("%s: dictionary confidence %.2f is below the dictionary-free %.2f",
				l.Name, withDict.Confidence, withoutDict.Confidence)
		}
	}
	if !found {
		t.Skip("no corpus line contains a common dictionary word")
	}
}

// The acceptance test for Stage 2: beam plus dictionary must match the oracle
// at least as well as greedy did, on both arms.
func TestBeamMatchesOracle(t *testing.T) {
	r := loadRecognizer(t)
	for _, arm := range []string{"h36", "native"} {
		lines := loadCorpus(t, arm)
		var exact int
		var total float64
		for _, l := range lines {
			line, err := r.Recognize(l.Image)
			if err != nil {
				t.Errorf("%s/%s: Recognize() error = %v", arm, l.Name, err)
				continue
			}
			if line.Text == l.Oracle {
				exact++
			} else {
				t.Logf("%s/%s\n  oracle: %q\n  cadmus: %q\n  cer:    %.4f", arm, l.Name, l.Oracle, line.Text, cer(line.Text, l.Oracle))
			}
			total += cer(line.Text, l.Oracle)
		}
		mean := total / float64(len(lines))
		t.Logf("%s arm: %d/%d exact, mean CER vs oracle %.4f", arm, exact, len(lines), mean)
		if mean > 0.01 {
			t.Errorf("%s arm: mean CER vs oracle = %.4f; want <= 0.01", arm, mean)
		}
	}
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `go test ./internal/recog/ -run TestBeam -v`
Expected: FAIL — `undefined: BeamDecode`.

- [ ] **Step 4: Implement**

Create `internal/recog/beam.go` following the transcription from Step 1. The
structure, with the constants pinned:

```go
// This file is a Go translation of src/lstm/recodebeam.cpp and
// src/lstm/recodebeam.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

const (
	// beamWidth is kBeamWidths[0]: the heap size for completed characters.
	// Tesseract's wider entries apply only to partial multi-code sequences,
	// which L1b's single-code recoder restriction rules out.
	beamWidth = 5
	// worstDictCert is kWorstDictCertainty / kCertaintyScale = -25/7.
	worstDictCert = -25.0 / CertaintyScale
)

// nodeContinuation constrains what may follow a beam node, so that the score
// merges below cannot double-count a timestep's probability.
type nodeContinuation int

const (
	// ncAnything: the node used only its own score.
	ncAnything nodeContinuation = iota
	// ncOnlyDup: the node absorbed a neighbour's probability without a
	// preceding stand-alone, so it must be followed by a duplicate of itself.
	ncOnlyDup
	// ncNoDup: the node combined scores after a stand-alone, so it may not be
	// followed by a duplicate of itself.
	ncNoDup
	ncCount
)

// topNState classifies an output index at one timestep.
type topNState int

const (
	tnTop2 topNState = iota
	tnTopN
	tnAlsoRan
	tnCount
)

// node is RecodeNode. dawgs is the lexicon frontier carried by this hypothesis.
type node struct {
	code      int
	unicharID int // -1 for the blank and for partial codes
	permuter  dict.PermuterType
	startOfDawg bool
	startOfWord bool
	endOfWord   bool
	duplicate   bool
	certainty float64
	score     float64
	prev      *node
	dawgs     []dict.Position
	codeHash  uint64
}
```

with these methods, each transcribed from its C++ counterpart:

- `computeTopN(row []float32, n int)` — fills `topNFlags`, `topCode`,
  `secondCode`; always forces `topNFlags[nullChar] = tnTop2`.
- `decodeStep(row, t)` — the three-tier sweep, then the `bestInitialDawgs` push.
- `continueContext(prev *node, isDawg bool, cont nodeContinuation, row []float32, flag topNState)`
  — the duplicate pushes, the `ncOnlyDup` early return, then the loop over
  every code with the `ContinueUnichar` and score-merge branches.
- `continueUnichar(...)` — the two-world split: `continueDawg` when `isDawg` and
  `cert > worstDictCert`, and the non-dawg push at `cert * DictRatio`.
- `continueDawg(...)` — `dict.LetterIsOkay`, the space special case, and the
  `bestInitialDawgs` slot.
- `pushIfBetter(heap *[]*node, n *node)` — dedup on
  `(code, codeHash, permuter, startOfDawg)`, keeping the best score, then evict
  the worst when the heap exceeds `beamWidth`.
- `codeHashOf(code int, dup bool, prev *node) uint64` — the rolling base-`CodeRange`
  hash that **skips duplicates and blanks**, so it hashes the CTC-collapsed
  label sequence.
- `extractBestPath() []*node` — the backtrack, skipping `ncOnlyDup` as a line
  ending and requiring a dawg hypothesis to end on a complete word or a space.
- `BeamDecode(out *nn.Tensor) ([]Scored, error)` — run the timesteps, backtrack,
  then reuse Task 19's span rule over the resulting node path to produce
  `Scored` values, so Task 20's word segmentation needs no change.

Add the `Dict` field to `Recognizer` and load the three dawg components in
`NewRecognizer` when they are present:

```go
type Recognizer struct {
	Net     *Network
	Charset *tessdata.Unicharset
	Recoder *tessdata.Recoder
	// Dict is the lexicon. Nil disables the dictionary beams; Recognize still
	// works, at greedy-equivalent accuracy.
	Dict *dict.Dict
}
```

and switch `Recognize` from `GreedyDecode` + `ScoreSymbols` to `BeamDecode`.

- [ ] **Step 5: Run the tests**

```bash
go test ./internal/recog/ -run TestBeam -v
```

**If `TestBeamAgreesWithGreedyOnUnambiguousInput` fails, stop and fix the beam
before looking at anything else.** With no dictionary and confident inputs the
two must agree; a disagreement means the search itself is wrong, and every other
beam test is then measuring noise.

**If only `TestBeamMatchesOracle` fails**, log the per-line diffs it prints and
classify them: word-level substitutions that are dictionary words (the dawg
weighting is too weak or the punctuation wrapping is wrong), space insertions or
deletions (the `ContinueDawg` space special case), or the same confusion pairs
greedy got wrong (the beam is not being consulted at all — check that `Dict` is
non-nil and that the dawg beams are populated).

- [ ] **Step 6: Commit**

```bash
git add internal/recog/beam.go internal/recog/beam_test.go
git commit -m "feat(recog): add the dictionary-weighted CTC beam search"
```

---

## Task 24: Wire-up, benchmark, and close

**Files:**
- Modify: `internal/recog/decode.go` — add the unexported `lexicon` field and the
  `SetLexicon` method
- Create: `internal/recog/bench_test.go`

- [ ] **Step 1: Add the lexicon toggle and a benchmark**

The design spec's public API has `WithLexicon(enabled bool)` — an *option* on the
not-yet-existing module root. L1b's recognizer is already constructed by the time
the toggle is useful, so the L1b-level spelling is the setter `SetLexicon`, and
L2 wraps it as `WithLexicon`. **The two names are not interchangeable; this task
implements `SetLexicon` and nothing else.**

Add an unexported `lexicon *dict.Dict` field to `Recognizer`, set by
`NewRecognizer` alongside `Dict`, so the toggle can restore what was loaded:

```go
// SetLexicon enables or disables dictionary weighting. Disabling it makes the
// beam search behave like greedy decoding with better certainties, because the
// score merge stays but the dawg beams are never populated.
//
// L2's WithLexicon(bool) option calls this.
func (r *Recognizer) SetLexicon(enabled bool) {
	if enabled {
		r.Dict = r.lexicon
		return
	}
	r.Dict = nil
}
```

Create `internal/recog/bench_test.go`:

```go
package recog

import "testing"

func BenchmarkRecognizeLine(b *testing.B) {
	r := loadRecognizer(b)
	lines := loadCorpus(b, "h36")
	b.ResetTimer()
	for range b.N {
		for _, l := range lines {
			if _, err := r.Recognize(l.Image); err != nil {
				b.Fatalf("Recognize() error = %v", err)
			}
		}
	}
}
```

**There are no `…B` variants.** `loadCorpus` (Task 17) and `loadRecognizer`
(Task 18) already take `testing.TB`, precisely so this benchmark can call them.
If either still takes `*testing.T`, widen it there — do not add a second helper
under a different name, which is what a `…B` suffix is.

- [ ] **Step 2: Measure**

```bash
CGO_ENABLED=0 go test ./internal/recog/ -run '^$' -bench BenchmarkRecognizeLine -benchtime 3x
```

Record the per-line time. The spec's risk table already accepts that pure-Go CPU
inference is slow and names the optimization path (int8 weights, then SIMD via
Go assembly); this number is the baseline that decides whether that work is
needed.

```bash
bd update cad-l1b --append-notes "Baseline: <N> ms per line crop, single core, unoptimized. Corpus of <M> lines."
```

- [ ] **Step 3: Full verification**

```bash
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=0 go test ./...
golangci-lint run
[ -s go.sum ] && { echo "go.sum is non-empty; a dependency was added"; exit 1; } || echo "zero dependencies: ok"
```

All must pass. `go test ./...` must also pass with the fixtures removed — verify
the skips work:

```bash
mv testdata/eng.traineddata /tmp/ && CGO_ENABLED=0 go test ./... ; mv /tmp/eng.traineddata testdata/
```

Expected: PASS, with skips logged. **A failure here means a test hard-depends on
a fixture and CI will break.**

- [ ] **Step 4: File the follow-ups**

```bash
bd create --title "cad-l1-int8: support tessdata_fast int8 weights" \
  --description "L1b is float-only. int8 needs: WeightMatrix scales_ loading, NetworkIO int8 storage with the x128/x127 quantization in SetPixel and WriteTimeStep, IntSimdMatrix-equivalent matvec, and the ResizeFloat exception that forces softmax outputs back to float."
bd create --title "cad-l1-2d-lstm: support 2-D LSTMs and nested softmax layers" \
  --description "internal/nn/lstm.go rejects na == ni + 2*ns (the GFS gate) and na != ni + ns (softmax feedback). No tessdata_best Latin model uses either. Needed for scripts whose networks use Lfys/Lbx 2-D variants."
bd create --title "cad-l1-tight-boxes: intersect word boxes with connected-component ink" \
  --description "Word boxes inherit the full line strip's vertical extent, exactly as Tesseract's InitializeWord does. internal/imaging/conncomp.go can tighten them; decide whether that is wanted before Kleio's spatial confidence grid consumes them."
```

- [ ] **Step 5: Close the bead**

```bash
bd update cad-l1b --append-notes "L1b complete. Tensor runtime (StrideMap, Tensor, Matrix, transcribed activation tables, Convolve, Maxpool, Reconfig, FullyConnected, LSTM, Series/Parallel/Reversed/XYTranspose), line normalization to 36px with Leptonica-compatible scaling, greedy CTC decode, dictionary-weighted beam search, per-word confidence and bounding boxes. Acceptance vs tesseract --psm 13: h36 mean CER <x>, native mean CER <x>."
bd close cad-l1b
```

- [ ] **Step 6: Commit**

```bash
git add internal/recog
git commit -m "feat(recog): add the lexicon toggle and a recognition benchmark"
```

---

## Self-Review

**Spec coverage.** The design spec's L1 scope is "network-graph deserialization
into the tensor runtime, LSTM forward pass, and CTC beam-search decode with dawg
lexicon weighting", verified by "for a corpus of line crops, Go output text must
match `tesseract --psm 7` output; per-character confidence within tolerance". **The
oracle is corrected to `--psm 13` here** — see the box at the top of this plan;
`--psm 7` does not feed the LSTM the committed crop, so it cannot be an oracle
for what Cadmus computes. Update the spec to match.
Deserialization is the spike's and **L1a's**, not this plan's. Tasks 1, 15 and 16
are verification-only because L1a already ships the layer geometry, the
unicharset and the recoder; Task 21 only adds edge accessors to L1a's dawg
reader. This plan covers the tensor runtime (Tasks 2-11), the forward pass
(Tasks 12-14), and both decodes (Tasks 17-20 and 21-23), with the corpus and the
oracle in Task 17 and the acceptance assertions in Tasks 18 and 23. §8's
requirement — "geolocated per-word confidence: `Word.Bounds` plus
`Word.Confidence`, exactly the information Kleio currently parses out of
`tesseract ... tsv`" — is Tasks 19 and 20.

**L1a is a hard blocking dependency and it has not run.** The Preconditions
section states the gate, the failure signature, and the exact L1a API this plan
consumes; nothing here re-implements any of it. The two plans were drafted
against different names for `Unicharset`, `Recoder` and `Dawg`, and L1a's win —
that reconciliation is a table in Preconditions, not an exercise for the reader.

**Deliberately out of scope, with beads filed in Task 24:** int8 weights, 2-D
LSTMs and nested-softmax LSTMs, multi-code (CJK/Indic) recoders, tight word
boxes, and auto-inversion (`tessedit_do_invert`, which reruns the whole forward
pass on an inverted image and keeps the better result — it means the `tesseract`
CLI's default output is not always a single forward pass, and on a clean corpus
it never fires).

**Two-stage sequencing.** Stage 1 (greedy, no dictionary) is Tasks 17-20 and
ends with a stated acceptance gate; Task 21 must not start until it passes. Task
14 builds the per-layer activation diff *before* the decoder exists, so it is
available the moment Stage 1 produces wrong text, and Task 18 Step 5 is the
explicit ladder from "the text is wrong" to a single layer.

**Placeholder scan.** Every task has real code and real assertions. Four
deliberate exceptions, all of which are pointers to a source file plus what to
do with what is found, not deferred work:

1. **Task 12 Step 1 and Step 5** — the Leptonica 1.87.0 scaling dispatch and the
   two low-level routines. The research read `master`, not 1.87.0, and said so;
   the step names the tarball, the functions, and a three-way acceptance rule
   with a defined fallback (the `h36` corpus arm) so a mismatch does not block.
   This is the one place in the plan where a substantial routine ships as
   "transcribe what you read"; transcribing from a version-pinned source is the
   *point*, because inventing an interpolation kernel would be worse.
2. **Task 22 Step 1** — `def_letter_is_okay`. The Go is written out in full, but
   the step lists seven specific branches to check against the source — including
   the permuter carry and the `DAWG_TYPE_PATTERN` branch this plan deliberately
   does not implement — and states that the source wins.
3. **Task 23 Step 1** — `ContinueContext` and friends. The plan gives the method
   list, the constants, and the invariants, and requires the transcription to be
   checked line by line, because the research read it once and this is the most
   intricate function in the port.
4. **Task 14 Step 5** — the instrumented-Tesseract patch is applied by hand and
   then *regenerated* with `git diff` before it is committed, because line
   numbers move between releases. The committed artefact is machine-applicable
   even though the sketch in the plan is not.

**Type consistency.** `nn.StrideMap` (Task 2) is consumed by `Tensor` (3) and
every layer (6-10). `nn.Tensor` is the single currency between layers.
`nn.Matrix.Inputs` excludes the bias column while `tessdata.Matrix.Cols`
includes it; `convertMatrix` (Task 11) is the one place that conversion happens
and it is asserted against the layer header. `recog.Symbol` (18) is embedded in
`recog.Scored` (19) and consumed by `splitWords` and `wordBounds` (20).
`Scored` is also what `BeamDecode` returns (23), which is why Task 20's word
segmentation needs no change when Stage 2 lands. `tessdata.NoEdge` and
`tessdata.PatternUnicharID` are defined in Task 21 and consumed in Task 22;
`dict.DawgType` is defined in Task 22, not in the loader, because only the
lexicon walk needs it. `recog.DictRatio` is defined in Task 19 and used by both
decoders. `nn.Rand` is created once in `Build` (Task 11), exported as
`Network.Rand`, and consumed by `Normalize` (13) *before* every `Convolve` (7) —
that order is load-bearing, not incidental.

**Signature changes this revision makes, and why:**
`Normalize` gained a `*nn.Rand`; `ScoreSymbols` lost its unused `nullChar`;
`Dict.LetterIsOkay` gained a leading `prev PermuterType`; `loadCorpus` and
`loadRecognizer` take `testing.TB`. Each is forced by a Tesseract behaviour the
earlier draft had wrong, and each is named in the task that introduces it.

**The constants that decide whether the confidence numbers are right**, all
pinned with their source: `kMinCertainty = -20`, `kCertOffset = -0.085`,
`kDictRatio = 2.25`, `kCertaintyScale = 7`, `kWorstDictCertainty = -25`,
`kStateClip = 100`, `kMaxSoftmaxActivation = 86`, `kTableSize = 4096`,
`kScaleFactor = 256`, `kBeamWidths[0] = 5`, `kDawgMagicNumber = 42`,
`kMaxCodeLen = 9`.

**Numeric bounds, and which are measured versus imposed.** Measured or derived:
the 1.5e-6 activation-table interpolation bound (Task 4, from
`(h²/8)·max|f''|`); the 4096 table sizes; the eng shapes (36 input height, 111
outputs, null char 110, 112 unicharset entries, 461848 word-dawg edges). Imposed
by this plan and explicitly flagged as such: the 0.02 Stage-1 mean-CER bound
(Task 18 Step 5, with instructions to raise it to the measured value if the
failures are concentrated in confusion pairs and dictionary words) and the 0.01
Stage-2 bound (Task 23). **No greedy-versus-beam accuracy delta is asserted
anywhere**, because none is published and the research declined to invent one;
Task 18's measurement is how the project finds out.
