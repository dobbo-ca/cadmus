# Cadmus L1a Implementation Plan — Complete Model Loading

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `eng.traineddata` from `tessdata_best` into complete in-memory
structures — every weight *value*, the unicharset, the recoder, and the three
DAWG lexicons — and make a human able to look at the result and see that the
numbers are real.

**Architecture:** L1a extends the packages the `cad-spike` already proved. It does
not re-create them. `internal/tessdata` gains float-weight payload loading on top
of the existing `ParseNetwork` walk, a retained recognizer trailer, three new
component parsers (`unicharset.go`, `recoder.go`, `dawg.go`), and a `LoadModel`
aggregate that enforces the cross-component invariants that bind the four
components together (`Softmax.no == recoder.code_range == null_char + 1`, and
`len(recoder) == len(unicharset)`). `cmd/cadmusdump` becomes the human-facing
sanity check: per-matrix weight statistics, the unicharset table, the recoder
map, and DAWG summaries.

**Tech Stack:** Go 1.26, standard library only. Test-only oracles: `tesseract` /
`combine_tessdata` (via Homebrew) for cross-checking component inventories and the
recognizer trailer. Never linked, never shipped, never required in CI.

## Global Constraints

- **Go 1.26.** `go.mod` declares `go 1.26`.
- **Module path:** `github.com/dobbo-ca/cadmus`
- **Zero third-party dependencies.** `go.mod`'s `require` block stays empty and
  `go.sum` stays empty for the life of this plan. CI enforces the `go.sum` check
  already. If a task seems to need a dependency, stop and raise it.
- **No cgo.** All packages build with `CGO_ENABLED=0`.
- **Apache-2.0.** Every file that is a translation of Tesseract source carries the
  attribution header described in `internal/doc.go`, naming the Tesseract source
  files it derives from. Files that are original work carry no such header.
- **Oracles are test-only.** `tesseract` and `combine_tessdata` may be invoked by
  humans and by the steps of this plan, never by Go tests. `go test ./...` must
  pass on a machine with neither installed — every fixture-dependent test uses
  `t.Skipf`, never `t.Fatal`.
- **FLOAT weights only, from `tessdata_best`.** The int8-quantized path
  (`kInt8Flag` on a weight matrix, `TF_INT_MODE` in the recognizer trailer) is a
  later bead. L1a must reject an int8 model **loudly**, not silently mis-parse it.
- **Scope wall.** L1a loads and validates. No forward pass, no arithmetic on
  weights beyond min/max/mean summary statistics, no decoding. If a task starts
  to need a tensor op, it has left this plan.

---

## File Structure

| File | Responsibility | New / Extended |
|---|---|---|
| `testdata/fetch.sh` | Fetch the pinned `tessdata_best` model and verify its checksum | Extended (Task 1) |
| `internal/tessdata/tfile.go` | Endian-aware primitive reader | Extended: `Int16`, `Uint64`, `Float32` (Task 2) |
| `internal/tessdata/network.go` | Layer graph **with weight values** and type-specific fields | Extended (Tasks 3, 4) |
| `internal/tessdata/unicharset.go` | `lstm-unicharset` text parser | New (Task 5) |
| `internal/tessdata/recoder.go` | `lstm-recoder` binary parser | New (Task 6) |
| `internal/tessdata/dawg.go` | `SquishedDawg` reader for the three `lstm-*-dawg` components | New (Task 7) |
| `internal/tessdata/model.go` | `LoadModel`: the whole file plus cross-component invariants | New (Task 8) |
| `cmd/cadmusdump/main.go` | Human sanity check: weight stats, unicharset, recoder, dawgs | Extended (Task 9) |

**Task dependencies:** Task 1 must run first (every later test reads the new
fixture). Task 2 must precede Tasks 4 and 7. **Task 5 must precede Tasks 6 and
7**: Task 6's `setupDecoder` uses `UnicharSpace`, and Task 7's `loadRealDawgs`
test helper calls `ParseUnicharset` — both are compile-time dependencies even
though the three parsers are logically independent. Tasks 3/4 are independent of
5–7 and the two chains (3→4 and 5→6, 5→7) can be worked concurrently. Task 8
needs all of 3–7. Task 9 needs 8.

---

## Facts established by research, and where they came from

Every number below was read out of the real `tessdata_best` 4.1.0 `eng.traineddata`
during planning, not inferred. Tasks assert against them. If an assertion fires,
the parser is wrong — do not relax the assertion.

| Fact | Value |
|---|---|
| Model | `tessdata_best` tag `4.1.0`, 15,400,601 bytes, sha256 `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba` |
| Version component | `4.00.00alpha:eng:synth20170629:[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]` |
| `network_str_` in the trailer | `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]` (49 bytes) |
| Weight-matrix `mode` byte | `132` on **every** matrix = `kDoubleFlag\|kAdamFlag`, `kInt8Flag` clear |
| Layer training-state byte | `2` (`TS_TEMP_DISABLE`) on **every** layer — so no `updates_`/`dw_sq_sum_` arrays |
| `network_flags` | `192` on **every** layer = `NF_LAYER_SPECIFIC_LR\|NF_ADAM` |
| `empty_` padding element | `0.0` in every matrix (NOT asserted on — nothing in the format requires it) |
| Trailer | `training_flags=64`, `training_iteration=814100`, `sample_iteration=814136`, `null_char=110`, `adam_beta=0.999`, `learning_rate=0.001`, `momentum=0.5`, then 0 bytes left |
| unicharset | 112 entries; scripts only `{Common, Latin}`; longest line 82 bytes; two line shapes (4 fields ×1, 8 fields ×111) |
| recoder | 112 entries, **all length 1**, `code_range = 111`, space → code `0` |
| dawgs | punc 539 edges (4322 B), word 461,848 edges (3,694,794 B), number 591 edges (4738 B); all edges forward; `flag_start_bit = 7` |
| word-dawg node 0 | 67 edges, sorted by unichar id, **no duplicate ids** |

The parsed tree, with the weight statistics Task 9 must be able to print:

```
Series "Series" ni=36 no=111 weights=1461007
  Input "Input" ni=36 no=1 weights=0
      shape batch=1 height=36 width=0 depth=1 loss=0
  Series "ConvSeries" ni=1 no=16 weights=160
    Convolve "Convolve" ni=1 no=9 weights=0
        half_x=1 half_y=1
    Tanh "ConvNL" ni=9 no=16 weights=160
        [0]     16x10   min=-0.683293 max=0.878813  mean=-0.030511
  Maxpool "Maxpool" ni=16 no=16 weights=0
      x_scale=3 y_scale=3
  XYTranspose "XYTransLSTM" ni=16 no=64 weights=20736
    SummLSTM "Lfys64" ni=16 no=64 na=80 weights=20736
        [0] CI  64x81   min=-2.899579 max=2.975000  mean=-0.009705
        [1] GI  64x81   min=-3.044665 max=3.522794  mean=-0.021487
        [2] GF1 64x81   min=-4.070291 max=3.969492  mean=0.029684
        [3] GO  64x81   min=-3.381452 max=2.538097  mean=0.004132
  LSTM "Lfx96" ni=64 no=96 na=160 weights=61824
      [0] CI  96x161  min=-3.978138 max=3.016736  mean=-0.002328
      [1] GI  96x161  min=-4.359385 max=4.154920  mean=-0.000619
      [2] GF1 96x161  min=-3.729903 max=3.143611  mean=0.001740
      [3] GO  96x161  min=-2.998609 max=3.504283  mean=-0.006121
  RTLReversed "RevLSTM" ni=96 no=96 weights=74112
    LSTM "Lrx96" ni=96 no=96 na=192 weights=74112
        [0] CI  96x193  min=-4.031739 max=3.743301  mean=-0.006396
        [1] GI  96x193  min=-3.593063 max=3.653696  mean=-0.000674
        [2] GF1 96x193  min=-3.411325 max=3.682367  mean=0.014321
        [3] GO  96x193  min=-3.787791 max=3.537246  mean=-0.015312
  LSTM "Lfx512" ni=96 no=512 na=608 weights=1247232
      [0] CI  512x609 min=-5.091887 max=5.650279  mean=-0.002124
      [1] GI  512x609 min=-8.193624 max=9.140669  mean=-0.003454
      [2] GF1 512x609 min=-6.082829 max=4.703187  mean=-0.001162
      [3] GO  512x609 min=-5.406325 max=6.143987  mean=-0.004736
  Softmax "Output" ni=512 no=111 weights=56943
      [0]     111x513 min=-29.279424 max=35.155323 mean=0.051364
```

**Note the topology change.** The spike parsed Homebrew's *int8* model
(`Lfys48 … Lfx192`, 385,807 weights). `tessdata_best` is a different, larger
network (`Lfys64 … Lfx512`, 1,461,007 weights). Task 1 documents that this is
expected, not a regression.

---

## Task 1: Pin `testdata/fetch.sh` to `tessdata_best` and re-baseline the spike output

The committed fixture today is Homebrew's `eng.traineddata`, which is int8
(`mode` byte 133, `training_flags` 65). L1 implements the float path only, so the
fixture must change and the wrong model must fail loudly rather than quietly.

**Files:**
- Modify: `testdata/fetch.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: `testdata/eng.traineddata` == `tessdata_best` 4.1.0 eng, 15,400,601
  bytes, sha256 `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba`.

- [ ] **Step 0: Create the tracking bead**

`cad-l1a` does not exist yet; the epic `cad-l1` and the loose bead `cad-jgq`
("Record the LSTMRecognizer trailer format", which Task 4 completes) do. Create
it before starting so the Final step has something to close:

```bash
bd create --id cad-l1a --parent cad-l1 -t task -p 0 \
  "L1a: complete model loading (weights, unicharset, recoder, dawgs)"
bd show cad-l1a
```

If `bd create --id` refuses the explicit id, create it without `--id`, record the
id it hands back, and substitute that id everywhere this plan says `cad-l1a`.

- [ ] **Step 1: Record the current baseline before changing anything**

```bash
cd /Users/christopherdobbyn/work/dobbo-ca/cadmus
shasum -a 256 testdata/eng.traineddata
go run ./cmd/cadmusdump testdata/eng.traineddata > /tmp/cadmus-baseline-homebrew.txt
head -20 /tmp/cadmus-baseline-homebrew.txt
```

Record whatever sha256 you get — it is a before/after marker, not an assertion.
(It was `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` during
planning, but that is only whatever the Homebrew `tesseract` formula last shipped
on this machine and it will differ elsewhere or after a `brew upgrade`.) What
*is* expected is the tree: `Lfys48 / Lfx96 / Lrx96 / Lfx192`, `weights=385807`.
Keep the file; Step 5 diffs against it.

- [ ] **Step 2: Rewrite `testdata/fetch.sh`**

Replace the whole file with:

```bash
#!/usr/bin/env bash
# Fetches the model fixture used by tests. Not run in CI; fixture-dependent
# tests t.Skipf when it is absent. Run this once locally before working on
# internal/tessdata.
#
# The fixture is ALWAYS tessdata_best. Cadmus L1 implements the FLOAT weight
# path only. Homebrew's share/tessdata/eng.traineddata is the int8-quantized
# tessdata build (weight-matrix mode byte 133 with kInt8Flag set,
# training_flags 65 with TF_INT_MODE set) and internal/tessdata rejects it by
# design. Do not reintroduce a Homebrew fallback here.
set -euo pipefail
cd "$(dirname "$0")"

URL="https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/eng.traineddata"
WANT_BYTES=15400601
WANT_SHA256="8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba"

# Download to a temporary file and only move it into place once it verifies.
# Writing straight to eng.traineddata would truncate a working fixture before
# any check ran, so a partial or corrupt download would leave the repo with a
# broken 15 MB file that tests still find (and therefore do not skip).
tmp="$(mktemp "${TMPDIR:-/tmp}/eng.traineddata.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

curl -fsSL -o "$tmp" "$URL"

got_bytes=$(wc -c < "$tmp" | tr -d ' ')
got_sha256=$(shasum -a 256 "$tmp" | cut -d' ' -f1)

if [ "$got_bytes" != "$WANT_BYTES" ] || [ "$got_sha256" != "$WANT_SHA256" ]; then
  echo "fetch.sh: downloaded file does not match pinned tessdata_best 4.1.0" >&2
  echo "  want ${WANT_BYTES} bytes sha256 ${WANT_SHA256}" >&2
  echo "  got  ${got_bytes} bytes sha256 ${got_sha256}" >&2
  echo "  any existing testdata/eng.traineddata was left untouched" >&2
  exit 1
fi

mv "$tmp" eng.traineddata
trap - EXIT

echo "fetched tessdata_best 4.1.0 eng.traineddata (${got_bytes} bytes, sha256 verified)"
```

- [ ] **Step 3: Fetch and verify**

```bash
./testdata/fetch.sh
```

Expected output: `fetched tessdata_best 4.1.0 eng.traineddata (15400601 bytes, sha256 verified)`.

**If the checksum fails:** the `4.1.0` tag is immutable, so a mismatch means the
download was corrupted or intercepted — retry once, and if it fails again stop and
report rather than editing `WANT_SHA256`. The script verifies in a temp file and
only `mv`s on success, so a failed run leaves whatever fixture was already there
intact; confirm that with `shasum -a 256 testdata/eng.traineddata` before
reporting. Do **not** switch the URL to `raw/main`; `main` is a moving target and
this pin is what makes the fixture reproducible.

- [ ] **Step 4: Confirm the fixture is not committed**

```bash
git check-ignore -v testdata/eng.traineddata
git status --porcelain testdata/
```

Expected: `check-ignore` reports the `.gitignore` line `*.traineddata`, and
`git status` shows no untracked `eng.traineddata`. A 15 MB binary must never enter
the repo; the fetch script plus the pinned checksum is the reproducibility
mechanism.

- [ ] **Step 5: Re-run the existing suite and diff the tree**

```bash
CGO_ENABLED=0 go test ./... 2>&1 | tail -20
go run ./cmd/cadmusdump testdata/eng.traineddata > /tmp/cadmus-baseline-best.txt
diff /tmp/cadmus-baseline-homebrew.txt /tmp/cadmus-baseline-best.txt
```

Expected: **all tests still pass.** The existing assertions in
`internal/tessdata/network_test.go` and `cmd/cadmusdump/main_test.go` are
structural (root is `Series`, at least one Convolve, at least one LSTM, output
contains `lstm` / `lstm-unicharset` / `Series`) — there are **no committed
layer-tree goldens to regenerate**. Verified by reading both test files during
planning.

The `diff` must show the expected topology change and nothing else:
`Lfys48 → Lfys64`, `Lfx192 → Lfx512`, `weights=385807 → weights=1461007`, `lstm`
component `401636 → 11689099` bytes, matrices losing their `(int8)` suffix, and
the `version:` line changing. `no=111` at the root and at `Output` is unchanged, as
are `lstm-unicharset` (6360 B) and `lstm-recoder` (1012 B).

**If any test fails**, that is information about the parser, not about the model —
fix the parser before proceeding.

- [ ] **Step 6: Cross-check against the Tesseract oracle**

```bash
combine_tessdata -d testdata/eng.traineddata
combine_tessdata -l testdata/eng.traineddata
```

`-l` must report `int_mode=0`, `recoding=1`, `null_char=110`,
`iteration=814100`, `sample_iteration=814136`, `learning_rate=0.001`,
`momentum=0.5`, `adam_beta=0.999`. Those are the values Tasks 4 and 8 assert on.
If `combine_tessdata` is not installed, skip this step and note it — the same
values are re-derived from the bytes in Task 4.

- [ ] **Step 7: Commit**

```bash
git add testdata/fetch.sh
git commit -m "chore(testdata): pin the fixture to tessdata_best 4.1.0

L1 implements the float weight path only; Homebrew's eng.traineddata is
int8-quantized. Pinned by tag and sha256 so the fixture is reproducible.
The model topology changes with it: Lfys48/Lfx192 -> Lfys64/Lfx512,
385807 -> 1461007 weights."
```

---

## Task 2: Reader primitives L1a needs

Three primitives the spike did not need: `Int16` (DAWG magic), `Uint64` (DAWG edge
records), `Float32` (the recognizer trailer's `adam_beta_`, `learning_rate_`,
`momentum_`, which are `float` and not `TFloat` in
`src/lstm/lstmrecognizer.h`).

**Files:**
- Modify: `internal/tessdata/tfile.go`
- Modify: `internal/tessdata/tfile_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces:

```go
func (r *Reader) Int16() (int16, error)
func (r *Reader) Uint64() (uint64, error)
func (r *Reader) Float32() (float32, error)
```

- [ ] **Step 1: Write the failing tests**

Append to `internal/tessdata/tfile_test.go`:

```go
func TestReaderInt16(t *testing.T) {
	// kDawgMagicNumber == 42, written as int16.
	r := NewReader([]byte{0x2a, 0x00, 0xff, 0xff})
	if got, err := r.Int16(); err != nil || got != 42 {
		t.Fatalf("Int16() = %d, %v; want 42, nil", got, err)
	}
	if got, err := r.Int16(); err != nil || got != -1 {
		t.Fatalf("Int16() = %d, %v; want -1, nil", got, err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
}

func TestReaderInt16Swapped(t *testing.T) {
	r := NewReader([]byte{0x00, 0x2a})
	r.SetSwap(true)
	if got, err := r.Int16(); err != nil || got != 42 {
		t.Fatalf("Int16() with swap = %d, %v; want 42, nil", got, err)
	}
}

func TestReaderInt16Truncated(t *testing.T) {
	r := NewReader([]byte{0x2a})
	if _, err := r.Int16(); err == nil {
		t.Fatal("Int16() on 1-byte input: want error, got nil")
	}
}

func TestReaderUint64(t *testing.T) {
	// A DAWG edge record: letter=9, eow set (bit 9), next_node=1222,
	// with flag_start_bit=7 => raw 0x0000000000131a09.
	r := NewReader([]byte{0x09, 0x1a, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00})
	got, err := r.Uint64()
	if err != nil || got != 0x131a09 {
		t.Fatalf("Uint64() = %#x, %v; want 0x131a09, nil", got, err)
	}
}

func TestReaderUint64Swapped(t *testing.T) {
	r := NewReader([]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x1a, 0x09})
	r.SetSwap(true)
	got, err := r.Uint64()
	if err != nil || got != 0x131a09 {
		t.Fatalf("Uint64() with swap = %#x, %v; want 0x131a09, nil", got, err)
	}
}

func TestReaderFloat32(t *testing.T) {
	// learning_rate_ in eng.traineddata is float32 0.001 == 0x3a83126f.
	r := NewReader([]byte{0x6f, 0x12, 0x83, 0x3a})
	got, err := r.Float32()
	if err != nil || got != float32(0.001) {
		t.Fatalf("Float32() = %v, %v; want 0.001, nil", got, err)
	}
}

func TestReaderFloat32Swapped(t *testing.T) {
	r := NewReader([]byte{0x3a, 0x83, 0x12, 0x6f})
	r.SetSwap(true)
	got, err := r.Float32()
	if err != nil || got != float32(0.001) {
		t.Fatalf("Float32() with swap = %v, %v; want 0.001, nil", got, err)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run 'TestReaderInt16|TestReaderUint64|TestReaderFloat32' -v`
Expected: FAIL — `r.Int16 undefined`, `r.Uint64 undefined`, `r.Float32 undefined`.

- [ ] **Step 3: Implement the primitives**

In `internal/tessdata/tfile.go`, add after `Uint32`/`Int32`:

```go
// Int16 reads a 2-byte signed integer. Only the DAWG header uses one
// (kDawgMagicNumber, src/dict/dawg.cpp).
func (r *Reader) Int16() (int16, error) {
	b, err := r.Bytes(2)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint16(b)
	if r.swap {
		v = v>>8 | v<<8
	}
	return int16(v), nil
}

// Float32 reads a 4-byte IEEE-754 value. Tesseract writes `float` rather than
// `TFloat` for the recognizer's adam_beta_, learning_rate_ and momentum_
// (src/lstm/lstmrecognizer.h) and for Plumbing::learning_rates_.
func (r *Reader) Float32() (float32, error) {
	b, err := r.Bytes(4)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(b)
	if r.swap {
		v = bits32Reverse(v)
	}
	return math.Float32frombits(v), nil
}

// Uint64 reads an 8-byte unsigned integer. DAWG edge records are uint64.
func (r *Reader) Uint64() (uint64, error) {
	b, err := r.Bytes(8)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(b)
	if r.swap {
		v = bits64Reverse(v)
	}
	return v, nil
}
```

and replace the existing `Int64` body so the two do not duplicate the swap logic:

```go
func (r *Reader) Int64() (int64, error) {
	v, err := r.Uint64()
	return int64(v), err
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -v`
Expected: PASS, including the pre-existing `TestReader*` and container/network tests.

- [ ] **Step 5: Commit**

```bash
git add internal/tessdata/tfile.go internal/tessdata/tfile_test.go
git commit -m "feat(tessdata): add Int16, Uint64 and Float32 reader primitives"
```

---

## Task 3: Load float weight VALUES

Replace the shape-only matrix skip with a real payload read, and make every path
that is not "modern float64, not a training dump" a loud error.

The full float weight-matrix body, from `WeightMatrix::DeSerialize`
(`src/lstm/weightmatrix.cpp:280`) and `GENERIC_2D_ARRAY<T>::DeSerialize`
(`src/ccstruct/matrix.h:197`), with `training == false`:

```
uint8   mode                    ; kInt8Flag=1, kAdamFlag=4, kDoubleFlag=128
int32   dim1                    ; outputs;  rejected above UINT16_MAX
int32   dim2                    ; inputs+recurrent+1 bias; rejected above UINT16_MAX
f64     empty_                  ; ONE padding element, BEFORE the payload
f64     array_[dim1*dim2]
```

…and nothing else. No scales, no size prefix, no trailing count.

Three traps, all confirmed against the real model:

1. **`kAdamFlag` is set (mode == 132) but no Adam arrays follow.** Whether
   `updates_`/`dw_sq_sum_` are present is decided by the *layer's* training-state
   byte, not by the mode byte. Every layer in `tessdata_best` has state `2`
   (`TS_TEMP_DISABLE`), because `LSTMTrainer::SaveRecognitionDump` flips it before
   writing. Keying off `kAdamFlag` desynchronises the stream.
2. **`empty_` sits between the dimensions and the payload.** The existing
   `skip2DArray` already accounts for it; the value reader must too.
3. **`dim1`/`dim2` are `int32_t` checked only against `UINT16_MAX`.** Negatives
   pass Tesseract's check and misbehave in `Resize`. Reject `< 0` explicitly.

**Files:**
- Modify: `internal/tessdata/network.go`
- Modify: `internal/tessdata/network_test.go`

**Interfaces:**
- Consumes: `Reader` (Task 2).
- Produces:

```go
// Matrix replaces MatrixShape.
type Matrix struct {
	Rows, Cols int
	Values     []float64
}

func (m *Matrix) At(row, col int) float64
func (m *Matrix) Stats() (min, max, mean float64)

// InputShape is Tesseract's StaticShape, the 4-D tensor description an Input
// layer carries (src/lstm/static_shape.h). Width 0 means "determined at
// runtime".
type InputShape struct {
	Batch, Height, Width, Depth, LossType int
}

type Layer struct {
	Type       LayerType
	Name       string
	NumInputs  int
	NumOutputs int
	NumWeights int
	Flags      int32
	Matrices   []Matrix
	Children   []*Layer

	// Type-specific fields, zero/nil unless the layer type sets them.
	HalfX, HalfY int         // Convolve:          half_x_, half_y_
	XScale       int         // Maxpool, Reconfig: x_scale_
	YScale       int         // Maxpool, Reconfig: y_scale_
	Shape        *InputShape // Input only
	NA           int         // LSTM family:       na_
}
```

The type-specific scalars are captured rather than skipped because they are the
model, not decoration: `Shape.Height` is the 36-px normalisation height,
`XScale` is the timestep→pixel reduction factor, `HalfX`/`HalfY` are the
convolution window, and `NA` is the gate matrices' column budget. They are
in scope for "complete model loading" and every one of them is consumed by L1b.

**These names are chosen to match L1b, and they supersede L1b Task 1.** The L1b
plan (`docs/superpowers/plans/2026-07-27-cadmus-l1b-forward-decode.md`, Task 1
"Retain the layer geometry the runtime needs") declares the *same* fields on the
*same* struct with `type InputShape struct{…}` / `Shape *InputShape` / `NA int`.
An earlier draft of this task used `InputShape [5]int32` and `NumAll int`, which
cannot coexist with L1b's `InputShape` type in one package. This task now uses
L1b's spelling. Consequences to honour when L1b is executed:

- L1b Task 1's premise ("`network.go` currently … throws them all away") is false
  once this task lands. **Do not re-implement it.** Reduce L1b Task 1 to a
  verification step: run `TestParseNetworkRetainsLayerGeometry` (whose assertions
  are a subset of `TestParseNetworkRealModelWeights` below) and confirm it passes
  without touching `network.go`, or drop it as redundant.
- L1b Task 1's test helper is named `parseRealModel`; this task's is
  `loadRealNetwork` (Task 4 renames it to `loadRealRecognizer` with a
  `loadRealNetwork` shim). L1b must use the existing helper rather than declaring
  a second one in the same package.

- [ ] **Step 1: Write the failing tests**

Append to `internal/tessdata/network_test.go`:

```go
// Add BOTH to the existing import block. network_test.go currently imports only
// os, path/filepath, strings and testing; the helpers below need encoding/binary
// as well as math.
import (
	"encoding/binary"
	"math"
)

// buildLayerHeader emits the common Network header: type tag, training state,
// backprop flag, flags, ni, no, num_weights, name.
func buildLayerHeader(typ LayerType, training int8, flags, ni, no, numWeights int32, name string) []byte {
	b := []byte{byte(typ), byte(training), 0}
	b = binary.LittleEndian.AppendUint32(b, uint32(flags))
	b = binary.LittleEndian.AppendUint32(b, uint32(ni))
	b = binary.LittleEndian.AppendUint32(b, uint32(no))
	b = binary.LittleEndian.AppendUint32(b, uint32(numWeights))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(name)))
	return append(b, name...)
}

// buildFloatMatrix emits a modern float64 WeightMatrix: mode byte, dim1, dim2,
// the lone empty_ element, then dim1*dim2 values in row-major order.
func buildFloatMatrix(mode uint8, rows, cols int, values []float64) []byte {
	b := []byte{mode}
	b = binary.LittleEndian.AppendUint32(b, uint32(rows))
	b = binary.LittleEndian.AppendUint32(b, uint32(cols))
	b = binary.LittleEndian.AppendUint64(b, math.Float64bits(0)) // empty_
	for _, v := range values {
		b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
	}
	return b
}

func TestParseMatrixReadsValuesRowMajor(t *testing.T) {
	// A 2x3 Tanh layer: 6 weights, num_weights must equal 6.
	vals := []float64{1, 2, 3, 4, 5, 6}
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, vals)...)

	r := NewReader(data)
	l, err := parseLayer(r)
	if err != nil {
		t.Fatalf("parseLayer() error = %v", err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
	if len(l.Matrices) != 1 {
		t.Fatalf("len(Matrices) = %d; want 1", len(l.Matrices))
	}
	m := l.Matrices[0]
	if m.Rows != 2 || m.Cols != 3 {
		t.Fatalf("matrix = %dx%d; want 2x3", m.Rows, m.Cols)
	}
	// Row-major: At(row, col) == Values[row*Cols+col].
	for row := range 2 {
		for col := range 3 {
			want := vals[row*3+col]
			if got := m.At(row, col); got != want {
				t.Errorf("At(%d,%d) = %v; want %v", row, col, got, want)
			}
		}
	}
	min, max, mean := m.Stats()
	if min != 1 || max != 6 || mean != 3.5 {
		t.Errorf("Stats() = %v, %v, %v; want 1, 6, 3.5", min, max, mean)
	}
}

func TestParseMatrixRejectsInt8Mode(t *testing.T) {
	// mode 133 == kDoubleFlag|kAdamFlag|kInt8Flag: Homebrew's tessdata build.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(133, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	_, err := parseLayer(NewReader(data))
	if err == nil {
		t.Fatal("parseLayer() with kInt8Flag: want error, got nil")
	}
	if !strings.Contains(err.Error(), "int8") {
		t.Errorf("error %q does not name int8 as the cause", err)
	}
}

func TestParseMatrixRejectsLegacyFormat(t *testing.T) {
	// mode 4 has kDoubleFlag clear => WeightMatrix::DeSerializeOld, float32.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(4, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	if _, err := parseLayer(NewReader(data)); err == nil {
		t.Fatal("parseLayer() with kDoubleFlag clear: want error, got nil")
	}
}

func TestParseLayerRejectsTrainingDump(t *testing.T) {
	// Training state 1 == TS_ENABLED: the matrices carry updates_ (and, with
	// kAdamFlag, dw_sq_sum_) arrays this parser deliberately does not read.
	data := buildLayerHeader(LayerTanh, 1, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	_, err := parseLayer(NewReader(data))
	if err == nil {
		t.Fatal("parseLayer() with TS_ENABLED: want error, got nil")
	}
	// Match the flag NAME, not the word "training": the trailer field names
	// training_flags and training_iteration also contain it, so "training"
	// alone would be satisfied by an unrelated failure.
	if !strings.Contains(err.Error(), "TS_ENABLED") {
		t.Errorf("error %q does not name the training state as the cause", err)
	}
}

func TestParseLayerRejectsNumWeightsMismatch(t *testing.T) {
	// num_weights claims 99 but the 2x3 matrix holds 6 elements.
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 99, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	if _, err := parseLayer(NewReader(data)); err == nil {
		t.Fatal("parseLayer() with a bad num_weights: want error, got nil")
	}
}

func TestParseMatrixRejectsNegativeDimension(t *testing.T) {
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	m := []byte{132}
	m = binary.LittleEndian.AppendUint32(m, uint32(int32(-1)))
	m = binary.LittleEndian.AppendUint32(m, 3)
	if _, err := parseLayer(NewReader(append(data, m...))); err == nil {
		t.Fatal("parseLayer() with dim1 = -1: want error, got nil")
	}
}

// The real model: every weight value present, in the shapes and ranges
// measured from tessdata_best 4.1.0 eng.traineddata during planning.
func TestParseNetworkRealModelWeights(t *testing.T) {
	root := loadRealNetwork(t)

	byName := map[string]*Layer{}
	var walk func(*Layer)
	walk = func(l *Layer) {
		byName[l.Name] = l
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(root)

	for _, tc := range []struct {
		name          string
		matrices      int
		rows, cols    int
		na            int
	}{
		{"ConvNL", 1, 16, 10, 0},
		{"Lfys64", 4, 64, 81, 80},
		{"Lfx96", 4, 96, 161, 160},
		{"Lrx96", 4, 96, 193, 192},
		{"Lfx512", 4, 512, 609, 608},
		{"Output", 1, 111, 513, 0},
	} {
		l, ok := byName[tc.name]
		if !ok {
			t.Errorf("layer %q missing from the parsed tree", tc.name)
			continue
		}
		if len(l.Matrices) != tc.matrices {
			t.Errorf("%s: len(Matrices) = %d; want %d", tc.name, len(l.Matrices), tc.matrices)
			continue
		}
		if l.NA != tc.na {
			t.Errorf("%s: NA = %d; want %d", tc.name, l.NA, tc.na)
		}
		for i, m := range l.Matrices {
			if m.Rows != tc.rows || m.Cols != tc.cols {
				t.Errorf("%s matrix %d = %dx%d; want %dx%d", tc.name, i, m.Rows, m.Cols, tc.rows, tc.cols)
			}
			if len(m.Values) != tc.rows*tc.cols {
				t.Errorf("%s matrix %d: len(Values) = %d; want %d", tc.name, i, len(m.Values), tc.rows*tc.cols)
			}
			min, max, mean := m.Stats()
			if math.IsNaN(min) || math.IsInf(min, 0) || math.IsNaN(max) || math.IsInf(max, 0) {
				t.Errorf("%s matrix %d: non-finite bounds min=%v max=%v", tc.name, i, min, max)
			}
			// Every matrix in this model straddles zero and stays well inside
			// +/-100. A matrix of zeros, or one full of huge values, means the
			// payload read is misaligned.
			if !(min < 0 && max > 0) {
				t.Errorf("%s matrix %d: range [%v, %v] does not straddle zero", tc.name, i, min, max)
			}
			if min < -100 || max > 100 {
				t.Errorf("%s matrix %d: range [%v, %v] outside the plausible +/-100", tc.name, i, min, max)
			}
			if math.Abs(mean) > 1 {
				t.Errorf("%s matrix %d: mean %v is implausibly large", tc.name, i, mean)
			}
		}
	}

	// Type-specific scalars.
	wantShape := InputShape{Batch: 1, Height: 36, Width: 0, Depth: 1, LossType: 0}
	if in := byName["Input"]; in == nil {
		t.Error("Input layer missing")
	} else if in.Shape == nil {
		t.Error("Input.Shape = nil; want the StaticShape")
	} else if *in.Shape != wantShape {
		t.Errorf("Input.Shape = %+v; want %+v", *in.Shape, wantShape)
	}
	if c := byName["Convolve"]; c == nil {
		t.Error("Convolve layer missing")
	} else if c.HalfX != 1 || c.HalfY != 1 {
		t.Errorf("Convolve half = %d,%d; want 1,1", c.HalfX, c.HalfY)
	}
	if mp := byName["Maxpool"]; mp == nil {
		t.Error("Maxpool layer missing")
	} else if mp.XScale != 3 || mp.YScale != 3 {
		t.Errorf("Maxpool scale = %d,%d; want 3,3", mp.XScale, mp.YScale)
	}

	// Spot-check one exact statistic against the value measured during
	// planning. Softmax "Output" has the widest range in the model; if the
	// payload were shifted by even one float64 this would not match.
	out := byName["Output"]
	min, max, mean := out.Matrices[0].Stats()
	assertClose(t, "Output min", min, -29.279424)
	assertClose(t, "Output max", max, 35.155323)
	assertClose(t, "Output mean", mean, 0.051364)
}

func assertClose(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 5e-6 {
		t.Errorf("%s = %.6f; want %.6f", what, got, want)
	}
}
```

Also refactor the existing `TestParseNetworkRealModel` so its fixture loading
lives in one helper both tests use — add to `network_test.go`:

```go
// loadRealNetwork parses the LSTM component of the fetched fixture, skipping
// the test when the fixture has not been fetched.
func loadRealNetwork(t *testing.T) *Layer {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(TypeLSTM)
	if !ok {
		t.Fatal("eng.traineddata has no lstm component")
	}
	root, err := ParseNetwork(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseNetwork() error = %v", err)
	}
	return root
}
```

and replace the body of `TestParseNetworkRealModel`'s first 20 lines with
`root := loadRealNetwork(t)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run 'TestParseMatrix|TestParseLayer|TestParseNetworkRealModelWeights' -v`
Expected: FAIL — `undefined: buildLayerHeader` / `l.Matrices[0].At undefined` /
`l.NA undefined` / `undefined: InputShape`.

- [ ] **Step 3: Implement value loading**

In `internal/tessdata/network.go`:

**(a)** Replace the `MatrixShape` type with:

```go
// Matrix is one deserialized weight matrix.
//
// Values is row-major with row stride Cols: element (row, col) is
// Values[row*Cols+col]. GENERIC_2D_ARRAY's index() is
// `column*dim2_ + row`, and its "column" is Tesseract's *output* index — so
// despite the header comment in src/ccstruct/matrix.h calling the storage
// column-major, the effective address math is dim1 contiguous runs of dim2.
//
// Rows is the output count (dim1). Cols is the input count plus ONE trailing
// bias column (dim2): MatrixDotVectorInternal computes the dot product over
// dim2-1 elements and then adds w[i][dim2-1] against an implicit 1.0
// (src/lstm/weightmatrix.cpp:99).
type Matrix struct {
	Rows, Cols int
	Values     []float64
}

// At returns the weight connecting input col to output row.
func (m *Matrix) At(row, col int) float64 { return m.Values[row*m.Cols+col] }

// Stats returns the minimum, maximum and arithmetic mean of the matrix's
// values. An empty matrix returns zeroes. Used by cadmusdump and by the tests
// that check the payload is real numbers rather than misaligned garbage.
func (m *Matrix) Stats() (min, max, mean float64) {
	if len(m.Values) == 0 {
		return 0, 0, 0
	}
	min, max = m.Values[0], m.Values[0]
	var sum float64
	for _, v := range m.Values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}
	return min, max, sum / float64(len(m.Values))
}
```

**(b)** Add the `InputShape` type and extend `Layer` with `Matrices []Matrix`
and the type-specific fields listed in the Interfaces block above.

**(c)** Replace the mode-byte constants block:

```go
const (
	// Flags on WeightMatrix::DeSerialize's leading mode byte
	// (src/lstm/weightmatrix.cpp:229).
	modeInt8Flag   uint8 = 1
	modeAdamFlag   uint8 = 4
	modeDoubleFlag uint8 = 128
)
```

`modeAdamFlag` is declared for documentation and is deliberately *not* consulted:
whether Adam arrays follow is decided by the layer's training state, and this
parser rejects training dumps outright. Add that as a comment so a later reader
does not "fix" it.

**(d)** Replace `parseMatrixShape` and delete `parseMatrixShapeOld`,
`skip2DArray`, `skipInt32s` and `maxScaleCount` — all four become unused:

```go
// parseMatrix mirrors WeightMatrix::DeSerialize for the only configuration
// cadmus supports: modern format (kDoubleFlag), float64 weights (kInt8Flag
// clear), not a training dump. Every other configuration is an error, loudly.
func parseMatrix(r *Reader) (Matrix, error) {
	mode, err := r.Uint8()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading matrix mode: %w", err)
	}
	if mode&modeDoubleFlag == 0 {
		return Matrix{}, fmt.Errorf("matrix mode %d has kDoubleFlag clear, selecting WeightMatrix::DeSerializeOld (float32 weights); cadmus supports only the modern float64 format", mode)
	}
	if mode&modeInt8Flag != 0 {
		return Matrix{}, fmt.Errorf("matrix mode %d has kInt8Flag set: this is an int8-quantized model. Cadmus L1 implements the float weight path only; run ./testdata/fetch.sh to get the tessdata_best model", mode)
	}
	return readFloat64Matrix(r)
}

// readFloat64Matrix mirrors GENERIC_2D_ARRAY<double>::DeSerialize +
// DeSerializeSize (src/ccstruct/matrix.h:197, :567):
//
//	int32   dim1
//	int32   dim2
//	f64     empty_             ; ONE padding element, before the payload
//	f64     array_[dim1*dim2]
//
// empty_ is read and discarded. It is 0.0 in every matrix of tessdata_best
// eng, but nothing in the format requires that, so its value is not asserted.
func readFloat64Matrix(r *Reader) (Matrix, error) {
	dim1, err := r.Int32()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading dim1: %w", err)
	}
	dim2, err := r.Int32()
	if err != nil {
		return Matrix{}, fmt.Errorf("reading dim2: %w", err)
	}
	// Tesseract checks only the UINT16_MAX upper bound; a negative size passes
	// and then misbehaves inside Resize(). Reject it here.
	if dim1 < 0 || dim1 > maxMatrixDim || dim2 < 0 || dim2 > maxMatrixDim {
		return Matrix{}, fmt.Errorf("implausible matrix dimensions %dx%d", dim1, dim2)
	}
	n := int(dim1) * int(dim2)
	if r.Remaining() < 8*(1+n) {
		return Matrix{}, fmt.Errorf("matrix %dx%d needs %d bytes, %d remain", dim1, dim2, 8*(1+n), r.Remaining())
	}
	if _, err := r.Float64(); err != nil { // empty_
		return Matrix{}, fmt.Errorf("reading empty_: %w", err)
	}
	m := Matrix{Rows: int(dim1), Cols: int(dim2), Values: make([]float64, n)}
	for i := range m.Values {
		if m.Values[i], err = r.Float64(); err != nil {
			return Matrix{}, fmt.Errorf("reading element %d of %dx%d: %w", i, dim1, dim2, err)
		}
	}
	return m, nil
}
```

**(e)** In `parseLayer`, reject training dumps right after reading the header, and
drop the `isTraining` variable and the `training bool` parameter from
`parseFullyConnected` / `parseLSTM`:

```go
	// TS_ENABLED means the file is a training dump, whose weight matrices carry
	// an extra updates_ array (and dw_sq_sum_ when kAdamFlag is set) that this
	// parser does not read. LSTMTrainer::SaveRecognitionDump flips the state to
	// TS_TEMP_DISABLE before writing, so no released model has TS_ENABLED;
	// every layer of tessdata_best eng has state 2.
	if training == tsEnabled {
		return nil, fmt.Errorf("tessdata: %v layer %q has training state TS_ENABLED: training dumps are not supported", typ, name)
	}
```

**(f)** Replace the three `skipInt32s` call sites with real field reads:

```go
	case LayerConvolve:
		// half_x_, half_y_ (src/lstm/convolve.cpp:45). Convolve holds no
		// weights: it is a pure im2col gather and recomputes
		// no_ = ni_*(2*half_x+1)*(2*half_y+1).
		l.HalfX, l.HalfY, err = parseInt32Pair(r)
	case LayerMaxpool, LayerReconfig:
		// x_scale_, y_scale_ (src/lstm/reconfig.cpp:60). Maxpool serializes
		// identical bytes and then overrides no_ = ni_ (src/lstm/maxpool.cpp:29).
		l.XScale, l.YScale, err = parseInt32Pair(r)
	case LayerInput:
		l.Shape, err = parseInputShape(r)
```

with

```go
func parseInt32Pair(r *Reader) (int, int, error) {
	a, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading first field: %w", err)
	}
	b, err := r.Int32()
	if err != nil {
		return 0, 0, fmt.Errorf("reading second field: %w", err)
	}
	return int(a), int(b), nil
}

// parseInputShape reads StaticShape's batch_, height_, width_, depth_ and
// loss_type_, in that order (src/lstm/static_shape.h:83).
func parseInputShape(r *Reader) (*InputShape, error) {
	var s InputShape
	for i, dst := range []*int{&s.Batch, &s.Height, &s.Width, &s.Depth, &s.LossType} {
		v, err := r.Int32()
		if err != nil {
			return nil, fmt.Errorf("reading input shape field %d: %w", i, err)
		}
		*dst = int(v)
	}
	return &s, nil
}
```

**(g)** In `parseLSTM`, record `na_` and use `parseMatrix`:

```go
	l.NA = int(na)
```

**(h)** At the end of `parseLayer`, before `return l, nil`, add the free
structural check:

```go
	// A layer's num_weights is the total element count of its own matrices
	// (bias columns included) plus its children's num_weights. Verified exactly
	// on every layer of tessdata_best eng: ConvNL 160 == 16*10,
	// Lfx96 61824 == 4*96*161, Output 56943 == 111*513, root Series 1461007 ==
	// the sum of its children. A mismatch means the parse desynchronised.
	//
	// The identity holds by construction for every layer type, including
	// NT_LSTM_SOFTMAX / NT_LSTM_SOFTMAX_ENCODED, whose nested softmax lives in
	// Children and which eng does not use:
	//
	//	LSTM::InitWeights           src/lstm/lstm.cpp:175   sums the gate
	//	                            matrices, then adds softmax_->InitWeights()
	//	                            when softmax_ != nullptr.
	//	Plumbing::InitWeights       src/lstm/plumbing.cpp:50 sums its stack.
	//	FullyConnected::InitWeights src/lstm/fullyconnected.cpp:86 is
	//	                            no_ * (ni_ + 1), i.e. rows*cols.
	total := 0
	for i := range l.Matrices {
		total += l.Matrices[i].Rows * l.Matrices[i].Cols
	}
	for _, c := range l.Children {
		total += c.NumWeights
	}
	if total != l.NumWeights {
		return nil, fmt.Errorf("tessdata: %v layer %q declares num_weights=%d but its matrices and children total %d", typ, name, l.NumWeights, total)
	}
```

**(j)** Update the file's own header comment block (`network.go:38-47`), which
is not otherwise touched by this task and would be left pointing at functions
that no longer exist. Replace the `WeightMatrix::DeSerialize` stanza's last two
branch lines so they name the new functions and say that the two unsupported
branches are now errors:

```go
// WeightMatrix::DeSerialize (src/lstm/weightmatrix.cpp):
//
//	uint8   mode               ; bit0 int8 weights, bit2 Adam, bit7 new format
//	if !(mode & 128)           → DeSerializeOld (float32). REJECTED by parseMatrix.
//	if mode & 1                → GENERIC_2D_ARRAY<int8> + scales. REJECTED by parseMatrix.
//	else                       → GENERIC_2D_ARRAY<float64>, see readFloat64Matrix
//	                             (training dumps, which add updates_/dw_sq_sum_,
//	                             are rejected in parseLayer on the TS_ENABLED
//	                             training-state byte)
```

Leave the `LSTMRecognizer::DeSerialize` line alone here — Task 4 Step 3 updates
it when it deletes `parseRecognizerTrailer`.

**(i)** Update `Tree` so the matrix rendering drops the removed `Int8` field and
prints one indented line per matrix with its statistics:

```go
// gateNames labels an LSTM layer's matrices in serialization order
// (LSTM::WeightType, src/lstm/lstm.h:32). Non-LSTM layers get a bare index.
var gateNames = [...]string{"CI", "GI", "GF1", "GO", "GFS"}

func (l *Layer) tree(w io.Writer, depth int) {
	pad := strings.Repeat("  ", depth)
	fmt.Fprintf(w, "%s%v %q ni=%d no=%d", pad, l.Type, l.Name, l.NumInputs, l.NumOutputs)
	if l.NA != 0 {
		fmt.Fprintf(w, " na=%d", l.NA)
	}
	fmt.Fprintf(w, " weights=%d\n", l.NumWeights)

	switch l.Type {
	case LayerInput:
		if s := l.Shape; s != nil {
			fmt.Fprintf(w, "%s    shape batch=%d height=%d width=%d depth=%d loss=%d\n",
				pad, s.Batch, s.Height, s.Width, s.Depth, s.LossType)
		}
	case LayerConvolve:
		fmt.Fprintf(w, "%s    half_x=%d half_y=%d\n", pad, l.HalfX, l.HalfY)
	case LayerMaxpool, LayerReconfig:
		fmt.Fprintf(w, "%s    x_scale=%d y_scale=%d\n", pad, l.XScale, l.YScale)
	}

	isLSTM := l.Type == LayerLSTM || l.Type == LayerLSTMSummary ||
		l.Type == LayerLSTMSoftmax || l.Type == LayerLSTMSoftmaxEncoded
	for i := range l.Matrices {
		m := &l.Matrices[i]
		label := ""
		if isLSTM && i < len(gateNames) {
			label = gateNames[i]
		}
		min, max, mean := m.Stats()
		fmt.Fprintf(w, "%s    [%d] %-3s %dx%d min=%f max=%f mean=%f\n",
			pad, i, label, m.Rows, m.Cols, min, max, mean)
	}
	for _, c := range l.Children {
		c.tree(w, depth+1)
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -v`
Expected: PASS, including `TestParseNetworkRealModelWeights` with its exact
`Output` statistics.

- [ ] **Step 5: Eyeball the weights**

```bash
go run ./cmd/cadmusdump testdata/eng.traineddata
```

Compare against the tree in the "Facts established by research" section above.
Every matrix must show a mean near zero and a range that straddles it, with
`Output` the wide outlier at `[-29.279424, 35.155323]`. A matrix of all zeroes, a
mean far from zero, or a range in the thousands means the payload offset is wrong
— stop and find it, do not proceed.

- [ ] **Step 6: Commit**

```bash
CGO_ENABLED=0 go vet ./...
git add internal/tessdata/network.go internal/tessdata/network_test.go
git commit -m "feat(tessdata): load float weight values, not just matrix shapes

Reads GENERIC_2D_ARRAY<double> payloads into Matrix.Values, captures the
type-specific layer fields (Input shape, Convolve window, Maxpool scale,
LSTM na_), and cross-checks each layer's num_weights against its matrices
and children. int8 matrices, the pre-kDoubleFlag format and training dumps
are now hard errors."
```

---

## Task 4: Keep the recognizer trailer, and assert float mode

`parseRecognizerTrailer` currently reads the eight LSTMRecognizer fields and
discards them. Three of them are load-bearing: `null_char_` is the CTC blank,
`sample_iteration_` seeds the randomizer L1b needs for Convolve's edge padding,
and `training_flags_` is the cheapest float-mode guard there is.

Trailer layout, from `LSTMRecognizer::DeSerialize` (`src/lstm/lstmrecognizer.cpp:133`)
with `include_charsets == false` — the case for every model that ships
`lstm-unicharset` and `lstm-recoder` as separate components:

```
string  network_str_
int32   training_flags_        ; TF_INT_MODE=1, TF_COMPRESS_UNICHARSET=64
int32   training_iteration_
int32   sample_iteration_
int32   null_char_
f32     adam_beta_             ; 4 bytes, not 8
f32     learning_rate_
f32     momentum_
```

**Files:**
- Modify: `internal/tessdata/network.go`
- Modify: `internal/tessdata/network_test.go`
- Modify: `cmd/cadmusdump/main.go`

**Interfaces:**
- Consumes: `Reader.Float32` (Task 2), `Layer` (Task 3).
- Produces:

```go
type Recognizer struct {
	Network           *Layer
	NetworkStr        string
	TrainingFlags     int32
	TrainingIteration int32
	SampleIteration   int32
	NullChar          int32
	AdamBeta          float32
	LearningRate      float32
	Momentum          float32
}

func ParseRecognizer(data []byte, swap bool) (*Recognizer, error)
```

`ParseNetwork` is **replaced** by `ParseRecognizer`, not kept alongside it. There
are three call sites (`network_test.go` ×2, `cmd/cadmusdump/main.go` ×1).

- [ ] **Step 1: Write the failing tests**

Replace `loadRealNetwork` in `network_test.go` with:

```go
// loadRealRecognizer parses the LSTM component of the fetched fixture,
// skipping the test when the fixture has not been fetched.
func loadRealRecognizer(t *testing.T) *Recognizer {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	lstm, ok := c.Entry(TypeLSTM)
	if !ok {
		t.Fatal("eng.traineddata has no lstm component")
	}
	rec, err := ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	return rec
}

func loadRealNetwork(t *testing.T) *Layer { return loadRealRecognizer(t).Network }
```

and add:

```go
// Exact trailer values, read out of tessdata_best 4.1.0 eng.traineddata during
// planning and independently confirmed by `combine_tessdata -l`.
func TestParseRecognizerRealModelTrailer(t *testing.T) {
	rec := loadRealRecognizer(t)

	const wantSpec = "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"
	if rec.NetworkStr != wantSpec {
		t.Errorf("NetworkStr = %q; want %q", rec.NetworkStr, wantSpec)
	}
	if rec.TrainingFlags != 64 {
		t.Errorf("TrainingFlags = %d; want 64 (TF_COMPRESS_UNICHARSET, TF_INT_MODE clear)", rec.TrainingFlags)
	}
	if rec.TrainingIteration != 814100 {
		t.Errorf("TrainingIteration = %d; want 814100", rec.TrainingIteration)
	}
	if rec.SampleIteration != 814136 {
		t.Errorf("SampleIteration = %d; want 814136", rec.SampleIteration)
	}
	if rec.NullChar != 110 {
		t.Errorf("NullChar = %d; want 110", rec.NullChar)
	}
	if rec.AdamBeta != 0.999 {
		t.Errorf("AdamBeta = %v; want 0.999", rec.AdamBeta)
	}
	if rec.LearningRate != 0.001 {
		t.Errorf("LearningRate = %v; want 0.001", rec.LearningRate)
	}
	if rec.Momentum != 0.5 {
		t.Errorf("Momentum = %v; want 0.5", rec.Momentum)
	}
	// The spec string is a placeholder for the output count: ParseOutput in
	// src/training/common/networkbuilder.cpp overrides whatever number appears
	// after O1c with the recoder's code range, which is why the persisted
	// string reads O1c1. Never derive the output count from it.
	if rec.Network.NumOutputs != 111 {
		t.Errorf("Network.NumOutputs = %d; want 111", rec.Network.NumOutputs)
	}
}

// buildTrailer emits the eight LSTMRecognizer fields that follow the root layer.
func buildTrailer(spec string, flags, trainIter, sampleIter, nullChar int32) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(len(spec)))
	b = append(b, spec...)
	for _, v := range []int32{flags, trainIter, sampleIter, nullChar} {
		b = binary.LittleEndian.AppendUint32(b, uint32(v))
	}
	for _, v := range []float32{0.999, 0.001, 0.5} {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	return b
}

// A minimal single-layer "network" plus a trailer, for the flag assertions.
//
// null_char MUST be < the layer's no (2 here), or ParseRecognizer's range check
// rejects the fixture and TestParseRecognizerAcceptsTiny can never pass. The
// real model's 110 belongs to a 111-output network, not to this one.
func buildTinyRecognizer(flags int32) []byte {
	data := buildLayerHeader(LayerTanh, 2, 0, 2, 2, 6, "T")
	data = append(data, buildFloatMatrix(132, 2, 3, []float64{1, 2, 3, 4, 5, 6})...)
	return append(data, buildTrailer("[tiny]", flags, 1, 2, 1)...)
}

func TestParseRecognizerRejectsIntMode(t *testing.T) {
	// training_flags 65 == TF_COMPRESS_UNICHARSET|TF_INT_MODE: Homebrew's
	// tessdata build of eng.
	_, err := ParseRecognizer(buildTinyRecognizer(65), false)
	if err == nil {
		t.Fatal("ParseRecognizer() with TF_INT_MODE: want error, got nil")
	}
	// Match the flag NAME. "int" alone is satisfied by "point", "printing" or
	// any int-typed field name, so it cannot distinguish this cause from an
	// unrelated failure.
	if !strings.Contains(err.Error(), "TF_INT_MODE") {
		t.Errorf("error %q does not name int mode as the cause", err)
	}
}

func TestParseRecognizerRejectsNonRecodingModel(t *testing.T) {
	// TF_COMPRESS_UNICHARSET clear means LoadRecoder builds a pass-through
	// recoder from the unicharset instead of reading the lstm-recoder
	// component. No stock model does this and cadmus does not implement it.
	if _, err := ParseRecognizer(buildTinyRecognizer(0), false); err == nil {
		t.Fatal("ParseRecognizer() with TF_COMPRESS_UNICHARSET clear: want error, got nil")
	}
}

func TestParseRecognizerRejectsTrailingBytes(t *testing.T) {
	data := append(buildTinyRecognizer(64), 0xde, 0xad)
	if _, err := ParseRecognizer(data, false); err == nil {
		t.Fatal("ParseRecognizer() with 2 trailing bytes: want error, got nil")
	}
}

func TestParseRecognizerAcceptsTiny(t *testing.T) {
	rec, err := ParseRecognizer(buildTinyRecognizer(64), false)
	if err != nil {
		t.Fatalf("ParseRecognizer() error = %v", err)
	}
	if rec.NetworkStr != "[tiny]" || rec.NullChar != 1 {
		t.Fatalf("trailer = %q, null %d; want \"[tiny]\", 1", rec.NetworkStr, rec.NullChar)
	}
}
```

Update `TestParseNetworkRejectsAbsurdStackSize` to call `ParseRecognizer`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run TestParseRecognizer -v`
Expected: FAIL — `undefined: ParseRecognizer`.

- [ ] **Step 3: Implement `ParseRecognizer`**

In `internal/tessdata/network.go`, first update the file's header comment block:
its last stanza currently reads "see `parseRecognizerTrailer` for the fields that
follow it", and that function is deleted here. Replace it with a reference to
`ParseRecognizer` and the field list below.

Then replace `ParseNetwork` and `parseRecognizerTrailer` with:

```go
// TrainingFlags bits, from src/lstm/lstmrecognizer.h:44.
const (
	tfIntMode            int32 = 1
	tfCompressUnicharset int32 = 64
)

// Recognizer is a parsed lstm component: the network graph plus the
// LSTMRecognizer fields serialized after it.
type Recognizer struct {
	Network *Layer

	// NetworkStr is the spec string the model was built from, e.g.
	// "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]". Its output count is
	// a PLACEHOLDER: NetworkBuilder::ParseOutput overrides the number after
	// O1c with the recoder's code range, which is why released models persist
	// "O1c1". Never derive the output count from this string.
	NetworkStr        string
	TrainingFlags     int32
	TrainingIteration int32

	// SampleIteration seeds LSTMRecognizer::SetRandomSeed, which drives the
	// random edge padding Convolve applies at image borders. L1b needs it.
	SampleIteration int32

	// NullChar is the network output index of the CTC blank. It is the
	// authority: UnicharCompress::DefragmentCodeValues relocates the null code
	// to the top of the range, so it is code_range-1 in practice, but the
	// format does not require that and this field does.
	NullChar int32

	AdamBeta     float32
	LearningRate float32
	Momentum     float32
}

// IsIntMode reports whether the model carries int8-quantized weights.
func (rec *Recognizer) IsIntMode() bool { return rec.TrainingFlags&tfIntMode != 0 }

// ParseRecognizer deserializes a .traineddata lstm component. swap comes from
// Container.Swapped.
//
// The component holds an LSTMRecognizer, not a bare network: the root layer is
// followed by the recognizer's own fields. Requiring the buffer to end exactly
// where the format says it should is what catches a parser that desynchronised
// mid-graph and returned a plausible-looking tree.
func ParseRecognizer(data []byte, swap bool) (*Recognizer, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	root, err := parseLayer(r)
	if err != nil {
		return nil, err
	}
	rec := &Recognizer{Network: root}
	if rec.NetworkStr, err = r.String(); err != nil {
		return nil, fmt.Errorf("tessdata: reading network spec string (the graph parse may have desynchronised): %w", err)
	}
	for _, f := range []struct {
		name string
		dst  *int32
	}{
		{"training_flags", &rec.TrainingFlags},
		{"training_iteration", &rec.TrainingIteration},
		{"sample_iteration", &rec.SampleIteration},
		{"null_char", &rec.NullChar},
	} {
		if *f.dst, err = r.Int32(); err != nil {
			return nil, fmt.Errorf("tessdata: reading %s: %w", f.name, err)
		}
	}
	// adam_beta_, learning_rate_ and momentum_ are declared `float` in
	// src/lstm/lstmrecognizer.h:350 — 4 bytes each, not 8.
	for _, f := range []struct {
		name string
		dst  *float32
	}{
		{"adam_beta", &rec.AdamBeta},
		{"learning_rate", &rec.LearningRate},
		{"momentum", &rec.Momentum},
	} {
		if *f.dst, err = r.Float32(); err != nil {
			return nil, fmt.Errorf("tessdata: reading %s: %w", f.name, err)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the lstm component; the parse desynchronised", r.Remaining())
	}

	if rec.IsIntMode() {
		return nil, fmt.Errorf("tessdata: training_flags %d has TF_INT_MODE set: this is an int8-quantized model. Cadmus L1 implements the float weight path only; run ./testdata/fetch.sh to get the tessdata_best model", rec.TrainingFlags)
	}
	if rec.TrainingFlags&tfCompressUnicharset == 0 {
		return nil, fmt.Errorf("tessdata: training_flags %d has TF_COMPRESS_UNICHARSET clear: the model has no recoder and LSTMRecognizer::LoadRecoder would synthesise a pass-through one. Cadmus requires a recoding model", rec.TrainingFlags)
	}
	if rec.NullChar < 0 || int(rec.NullChar) >= root.NumOutputs {
		return nil, fmt.Errorf("tessdata: null_char %d is outside the network's %d outputs", rec.NullChar, root.NumOutputs)
	}
	return rec, nil
}
```

Note the ordering: the trailer is read *before* the flag assertions so that a
desynchronised parse reports the desync rather than a nonsense flag value.

- [ ] **Step 4: Update `cmd/cadmusdump/main.go`**

Replace the `ParseNetwork` call:

```go
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return fmt.Errorf("parsing lstm component: %w", err)
	}
	fmt.Fprintf(w, "\nrecognizer:\n  spec           %s\n  training_flags %d\n  iteration      %d\n  sample_iter    %d\n  null_char      %d\n  adam_beta      %g\n  learning_rate  %g\n  momentum       %g\n",
		rec.NetworkStr, rec.TrainingFlags, rec.TrainingIteration,
		rec.SampleIteration, rec.NullChar, rec.AdamBeta, rec.LearningRate, rec.Momentum)
	fmt.Fprintln(w, "\nnetwork:")
	rec.Network.Tree(w)
	return nil
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./... -v 2>&1 | tail -40`
Expected: PASS everywhere.

- [ ] **Step 6: Cross-check the trailer against the oracle**

```bash
go run ./cmd/cadmusdump testdata/eng.traineddata | head -30
combine_tessdata -l testdata/eng.traineddata
```

The `recognizer:` block must agree with `combine_tessdata -l` field for field:
`int_mode=0`, `recoding=1`, `iteration=814100`, `sample_iteration=814136`,
`null_char=110`, `learning_rate=0.001`, `momentum=0.5`, `adam_beta=0.999`. If
`combine_tessdata` is not installed, note that in the task report — the Go test
asserts the same values from the bytes.

- [ ] **Step 7: Commit**

```bash
git add internal/tessdata/network.go internal/tessdata/network_test.go cmd/cadmusdump/main.go
git commit -m "feat(tessdata): retain the LSTMRecognizer trailer and assert float mode

ParseNetwork becomes ParseRecognizer, returning null_char (the CTC blank),
sample_iteration (the Convolve randomizer seed), the spec string and the
training hyperparameters. TF_INT_MODE and a missing TF_COMPRESS_UNICHARSET
are now hard errors."
```

- [ ] **Step 8: Close `cad-jgq`**

This task is exactly the bead `cad-jgq`, "Record the LSTMRecognizer trailer
format" — the eight-field layout is now transcribed in `ParseRecognizer` and
asserted by `TestParseRecognizerRealModelTrailer`.

```bash
bd update cad-jgq --append-notes "Trailer format recorded and asserted in internal/tessdata/network.go (ParseRecognizer): network_str_, training_flags_, training_iteration_, sample_iteration_, null_char_ (int32), then adam_beta_, learning_rate_, momentum_ (float32, 4 bytes each). Values for tessdata_best 4.1.0 eng: 64, 814100, 814136, 110, 0.999, 0.001, 0.5."
bd close cad-jgq
```

---

## Task 5: `lstm-unicharset` text parser

Component 21 is a plain text file: a decimal count line, then that many entry
lines. The LSTM path never touches `TESSDATA_UNICHARSET` (component 1) — it is
the legacy Tesseract-3 classifier's set and is absent from both eng models.

The C++ loader (`UNICHARSET::load_via_fgets`, `src/ccutil/unicharset.cpp:784`)
tries six progressively shorter `istream` extractions. The shapes those six accept
collapse to: a mandatory unichar and **hexadecimal** properties, an optional
comma-separated metrics tuple, a script name, and up to four further fields
(`other_case`, `direction`, `mirror`, `normed`). Verified against the real file:
eng uses exactly two shapes — 4 whitespace fields (id 0 only) and 8 (the other 111).

Traps:

- **Properties is hex.** `stream >> std::hex >> properties`. `|Broken|0|1` has
  properties `f`. The C++ *writer* emits decimal — that is a round-trip bug in
  Tesseract, and it makes the writer a bad reference. The loader is the authority.
- **`NULL` in column 1 is the literal spelling of the space character.**
- **id 0's stored `normed` is the string `"NULL"`** in C++, and
  `get_normed_unichar` dodges it with a `UNICHAR_SPACE` special case. Rewriting
  the unichar to `" "` before defaulting `normed` to it reproduces the observable
  behaviour without the special case.
- **`normed != unichar` for 6 eng entries** — ids 55 `’`→`'`, 59 `™`→`TM`,
  60 `“`→`"`, 70 `—`→`-`, 71 `”`→`"`, 84 `‘`→`'`. Which accessor L1d picks is
  therefore observable; that decision belongs to L1d, not here. Expose both.

Deliberate divergences from `load_via_fgets`, all of them rejections of inputs
Tesseract accepts, and all of them stated so a later reader does not treat them
as bugs:

- **A count of `0` is rejected.** `load_via_fgets` accepts it (its `for` loop
  simply does not run) and returns true. A zero-entry LSTM unicharset cannot
  produce a usable model — `LoadModel`'s `Recoder.Size() == Unicharset.Size()`
  and `Softmax.no == CodeRange` checks would fail anyway — so failing at the
  parse gives the better error message. Asserted by `TestParseUnicharsetErrors`'s
  `"zero count"` case.
- **`IsSpaceDelimited` compares script names, not resolved ids** (see the
  function's own comment).
- **Line length is not capped at 255 bytes** (see
  `TestUnicharsetLinesFitTesseractsBuffer`).

**Files:**
- Create: `internal/tessdata/unicharset.go`
- Test: `internal/tessdata/unicharset_test.go`

**Interfaces:**
- Consumes: `Container.Entry(TypeLSTMUnicharset)`.
- Produces:

```go
const (
	UnicharSpace  = 0
	UnicharJoined = 1
	UnicharBroken = 2
)

const (
	PropAlpha uint32 = 0x1
	PropLower uint32 = 0x2
	PropUpper uint32 = 0x4
	PropDigit uint32 = 0x8
	PropPunct uint32 = 0x10
)

type Unichar struct {
	Text       string
	Normed     string
	Script     string
	Properties uint32
}

// Unicharset holds the entries in file order; id is the slice index.
type Unicharset struct {
	chars []Unichar
}

func ParseUnicharset(data []byte) (*Unicharset, error)
func (u *Unicharset) Size() int
func (u *Unicharset) Char(id int) (Unichar, bool)
func (u *Unicharset) Text(id int) string
func (u *Unicharset) Normed(id int) string
func (u *Unicharset) IsSpaceDelimited(id int) bool
```

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/unicharset_test.go`:

```go
package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadRealUnicharset(t *testing.T) *Unicharset {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, ok := c.Entry(TypeLSTMUnicharset)
	if !ok {
		t.Fatal("eng.traineddata has no lstm-unicharset component")
	}
	u, err := ParseUnicharset(b)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	return u
}

func TestParseUnicharsetRealModel(t *testing.T) {
	u := loadRealUnicharset(t)

	if u.Size() != 112 {
		t.Fatalf("Size() = %d; want 112", u.Size())
	}

	// The three mandatory SpecialUnicharCodes, src/ccutil/unicharset.cpp:79.
	// Note id 0 is written as the literal token "NULL".
	for _, tc := range []struct{ id int; text, normed string }{
		{UnicharSpace, " ", " "},
		{UnicharJoined, "Joined", "Joined"},
		{UnicharBroken, "|Broken|0|1", "|Broken|0|1"},
		{3, "C", "C"},
		{55, "’", "'"},   // right single quote -> ASCII apostrophe
		{59, "™", "TM"},  // trade mark sign
		{70, "—", "-"},   // em dash
		{111, "é", "é"},
	} {
		if got := u.Text(tc.id); got != tc.text {
			t.Errorf("Text(%d) = %q; want %q", tc.id, got, tc.text)
		}
		if got := u.Normed(tc.id); got != tc.normed {
			t.Errorf("Normed(%d) = %q; want %q", tc.id, got, tc.normed)
		}
	}

	// Properties are HEXADECIMAL. |Broken|0|1 carries "f"; a decimal parse
	// would fail outright, and every value below 0xa would parse the same
	// either way, so this entry is the one that proves the base.
	if got := u.mustChar(t, UnicharBroken).Properties; got != 0xf {
		t.Errorf("Properties(|Broken|0|1) = %#x; want 0xf", got)
	}
	if got := u.mustChar(t, 3).Properties; got != PropAlpha|PropUpper {
		t.Errorf("Properties(C) = %#x; want %#x", got, PropAlpha|PropUpper)
	}

	// eng uses only Common and Latin, and every entry is space delimited.
	for id := range u.Size() {
		c, _ := u.Char(id)
		if c.Script != "Common" && c.Script != "Latin" {
			t.Errorf("Char(%d).Script = %q; want Common or Latin", id, c.Script)
		}
		if !u.IsSpaceDelimited(id) {
			t.Errorf("IsSpaceDelimited(%d) = false; want true for a Latin script", id)
		}
	}
}

// Tesseract reads unicharset lines through a char[256] buffer via TFile::FGets
// (src/ccutil/serialis.cpp:213), which copies at most 255 bytes and stops early
// at '\n'. It does NOT discard the rest of an over-long line: the remainder
// stays in the stream and is consumed as the NEXT entry line, so every
// subsequent unichar id shifts and the whole file desynchronises. Our parser
// reads whole lines and does not. eng's longest line is 82 bytes; if a model
// ever exceeds the cap the two parses disagree about every id past that point,
// and that must be visible, not silent.
func TestUnicharsetLinesFitTesseractsBuffer(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, _ := c.Entry(TypeLSTMUnicharset)
	longest := 0
	for _, line := range strings.Split(string(b), "\n") {
		if len(line) > longest {
			longest = len(line)
		}
	}
	t.Logf("longest lstm-unicharset line: %d bytes", longest)
	if longest > 255 {
		t.Errorf("longest line is %d bytes; Tesseract's FGets stops at 255 and reads the remainder as the next entry, shifting every subsequent unichar id", longest)
	}
}

func TestParseUnicharsetShapes(t *testing.T) {
	// Shape A (space only, 4 fields) and shape B (full, 8 fields).
	in := "2\n" +
		"NULL 0 Common 0\n" +
		"é 3 0,255,0,255,0,0,0,0,0,0 Latin 111 0 111 e\t# é [e9 ]a\n"
	u, err := ParseUnicharset([]byte(in))
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	if u.Size() != 2 {
		t.Fatalf("Size() = %d; want 2", u.Size())
	}
	if u.Text(0) != " " || u.Normed(0) != " " {
		t.Errorf("id 0 = %q/%q; want \" \"/\" \"", u.Text(0), u.Normed(0))
	}
	if u.Text(1) != "é" || u.Normed(1) != "e" {
		t.Errorf("id 1 = %q/%q; want \"é\"/\"e\"", u.Text(1), u.Normed(1))
	}
	if got := u.mustChar(t, 1).Properties; got != PropAlpha|PropLower {
		t.Errorf("Properties(id 1) = %#x; want %#x", got, PropAlpha|PropLower)
	}
	if u.mustChar(t, 1).Script != "Latin" {
		t.Errorf("Script(id 1) = %q; want Latin", u.mustChar(t, 1).Script)
	}
}

func TestParseUnicharsetErrors(t *testing.T) {
	for name, in := range map[string]string{
		"empty":            "",
		"bad count":        "abc\nNULL 0 Common 0\n",
		"count too high":   "3\nNULL 0 Common 0\n",
		"non-hex property": "1\nA zz Latin 0\n",
		"too few fields":   "1\nA 3\n",
		"zero count":       "0\n",
	} {
		if _, err := ParseUnicharset([]byte(in)); err == nil {
			t.Errorf("ParseUnicharset(%s): want error, got nil", name)
		}
	}
}

// Trailing bytes after the declared entry count are ignored, matching
// load_via_fgets, which reads exactly unicharset_size lines.
func TestParseUnicharsetIgnoresTrailingLines(t *testing.T) {
	u, err := ParseUnicharset([]byte("1\nNULL 0 Common 0\nJoined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n"))
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	if u.Size() != 1 {
		t.Errorf("Size() = %d; want 1", u.Size())
	}
}

func (u *Unicharset) mustChar(t *testing.T, id int) Unichar {
	t.Helper()
	c, ok := u.Char(id)
	if !ok {
		t.Fatalf("Char(%d) not present", id)
	}
	return c
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Unicharset -v`
Expected: FAIL — `undefined: ParseUnicharset`.

- [ ] **Step 3: Implement the parser**

Create `internal/tessdata/unicharset.go`:

```go
// This file is a Go translation of src/ccutil/unicharset.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Reserved unichar ids. kSpecialUnicharCodes, src/ccutil/unicharset.cpp:79,
// is {" ", "Joined", "|Broken|0|1"} and every unicharset with special codes
// starts with exactly those three.
const (
	UnicharSpace  = 0
	UnicharJoined = 1
	UnicharBroken = 2
)

// Character property bits. Stored HEXADECIMAL in the file.
// src/ccutil/unicharset.cpp:41.
const (
	PropAlpha uint32 = 0x1
	PropLower uint32 = 0x2
	PropUpper uint32 = 0x4
	PropDigit uint32 = 0x8
	PropPunct uint32 = 0x10
)

// maxUnicharsetSize is a corruption guard, not a Tesseract constant. The
// largest shipped unicharset (chi_sim) is a few thousand entries.
const maxUnicharsetSize = 1_000_000

// Unichar is one unicharset entry, reduced to the columns the LSTM
// recognition path reads.
//
// Deliberately not retained: the comma-separated metrics tuple (top/bottom,
// width, bearing and advance statistics), other_case, direction and mirror.
// Every accessor for those is reachable only from the legacy Tesseract-3
// classifier, the wordrec language model, or x-height fixing — none of which
// cadmus ports.
type Unichar struct {
	// Text is the raw file token, the UTF-8 that id_to_unichar returns and
	// that the `tesseract` binary emits.
	Text string
	// Normed is get_normed_unichar's result: LSTMRecognizer::DecodeLabel uses
	// it, so it differs from Text for 6 eng entries (curly quotes, em dash,
	// trade mark). Defaults to Text when the column is absent.
	Normed string
	// Script drives IsSpaceDelimited, i.e. word splitting.
	Script     string
	Properties uint32
}

// Unicharset is a parsed lstm-unicharset component.
type Unicharset struct {
	chars []Unichar
}

// ParseUnicharset parses the text format read by UNICHARSET::load_via_fgets.
func ParseUnicharset(data []byte) (*Unicharset, error) {
	sc := bufio.NewScanner(bytes.NewReader(data))
	if !sc.Scan() {
		return nil, errors.New("tessdata: unicharset is empty")
	}
	n, err := strconv.Atoi(strings.TrimSpace(sc.Text()))
	if err != nil {
		return nil, fmt.Errorf("tessdata: unicharset entry count %q: %w", sc.Text(), err)
	}
	// Deliberate divergence: load_via_fgets accepts a count of 0 and returns
	// true. An empty LSTM unicharset cannot yield a usable model, so reject it
	// here where the error can say why.
	if n <= 0 || n > maxUnicharsetSize {
		return nil, fmt.Errorf("tessdata: implausible unicharset size %d", n)
	}
	u := &Unicharset{chars: make([]Unichar, 0, n)}
	for id := range n {
		if !sc.Scan() {
			return nil, fmt.Errorf("tessdata: unicharset declares %d entries but ends after %d", n, id)
		}
		c, err := parseUnicharLine(sc.Text())
		if err != nil {
			return nil, fmt.Errorf("tessdata: unicharset entry %d: %w", id, err)
		}
		u.chars = append(u.chars, c)
	}
	// Anything after the declared count is ignored, matching load_via_fgets.
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("tessdata: reading unicharset: %w", err)
	}
	return u, nil
}

// parseUnicharLine parses one entry line. The C++ loader's six progressive
// extractions accept: unichar, HEX properties, an optional comma-separated
// metrics tuple, a script name, then up to four further fields
// (other_case, direction, mirror, normed).
func parseUnicharLine(line string) (Unichar, error) {
	// Everything from the first tab is a human-readable debug comment that
	// load_via_fgets never reaches, because its istream extraction stops at
	// the field count it wants.
	if i := strings.IndexByte(line, '\t'); i >= 0 {
		line = line[:i]
	}
	f := strings.Fields(line)
	if len(f) < 3 {
		return Unichar{}, fmt.Errorf("want at least 3 fields, got %d in %q", len(f), line)
	}
	c := Unichar{Text: f[0]}
	// The literal token "NULL" in column 1 spells the space character.
	if c.Text == "NULL" {
		c.Text = " "
	}
	props, err := strconv.ParseUint(f[1], 16, 32)
	if err != nil {
		return Unichar{}, fmt.Errorf("properties %q is not hexadecimal: %w", f[1], err)
	}
	c.Properties = uint32(props)

	// Field 3 is the metrics tuple when it contains commas, else the script.
	si := 2
	if strings.ContainsRune(f[2], ',') {
		si = 3
	}
	if si >= len(f) {
		return Unichar{}, fmt.Errorf("no script field in %q", line)
	}
	c.Script = f[si]

	// After the script come other_case, direction, mirror, normed, in that
	// order. normed is present only when all four are.
	if rest := f[si+1:]; len(rest) == 4 {
		c.Normed = rest[3]
	} else {
		// set_normed(id, normed[0] != '\0' ? normed : unichar). Because Text
		// has already been rewritten from "NULL" to " ", this also subsumes
		// get_normed_unichar's UNICHAR_SPACE special case.
		c.Normed = c.Text
	}
	return c, nil
}

func (u *Unicharset) Size() int { return len(u.chars) }

func (u *Unicharset) Char(id int) (Unichar, bool) {
	if id < 0 || id >= len(u.chars) {
		return Unichar{}, false
	}
	return u.chars[id], true
}

// Text is id_to_unichar: the raw file representation.
func (u *Unicharset) Text(id int) string {
	c, _ := u.Char(id)
	return c.Text
}

// Normed is get_normed_unichar.
func (u *Unicharset) Normed(id int) string {
	c, _ := u.Char(id)
	return c.Normed
}

// IsSpaceDelimited mirrors UNICHARSET::IsSpaceDelimited
// (src/ccutil/unicharset.h:668): every script except Han, Thai, Hangul,
// Hiragana and Katakana writes words with spaces between them.
//
// Deliberate divergence: Tesseract compares resolved script *ids*, and
// get_script_id_from_name returns 0 (the null script) for a name it has never
// seen — so in a unicharset with no Han entry, han_sid_ collides with
// null_sid_ and a null-script character is reported as NOT space delimited.
// Comparing names directly avoids that. eng has no null-script entry, so the
// two agree on the model at hand.
func (u *Unicharset) IsSpaceDelimited(id int) bool {
	c, ok := u.Char(id)
	if !ok {
		return true // INVALID_UNICHAR_ID
	}
	switch c.Script {
	case "Han", "Thai", "Hangul", "Hiragana", "Katakana":
		return false
	}
	return true
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Unicharset -v`
Expected: PASS. `TestUnicharsetLinesFitTesseractsBuffer` logs `82 bytes`.

- [ ] **Step 5: Verify the extracted component against Tesseract**

```bash
mkdir -p /tmp/cadmus-comp && cd /tmp/cadmus-comp
combine_tessdata -u /Users/christopherdobbyn/work/dobbo-ca/cadmus/testdata/eng.traineddata eng.
head -5 eng.lstm-unicharset
wc -l eng.lstm-unicharset
```

Expected: first line `112`, then `NULL 0 Common 0`, then the `Joined` and
`|Broken|0|1` lines; 113 lines total. If `combine_tessdata` is unavailable, skip
— the Go test asserts the same content.

- [ ] **Step 6: Commit**

```bash
git add internal/tessdata/unicharset.go internal/tessdata/unicharset_test.go
git commit -m "feat(tessdata): parse the lstm-unicharset component"
```

---

## Task 6: `lstm-recoder` binary parser

Component 22 is `UnicharCompress::Serialize`, which is one line:
`fp->Serialize(encoder_)` over a `std::vector<RecodedCharID>`. That lands in the
class-vector branch of `TFile::Serialize` (`src/ccutil/serialis.h:157`):

```
uint32  n                       ; entry count, one per unichar id
repeat n times:
  int8    self_normalized       ; NEVER byte-swapped (FReadEndian skips size 1)
  uint32  length                ; rejected above kMaxCodeLen = 9
  int32   code[length]
```

Everything else — `code_range_`, `decoder_`, `is_valid_start_`, `next_codes_`,
`final_codes_` — is recomputed at load by `ComputeCodeRange` + `SetupDecoder`.

L1a builds `code_range_`, `decoder_` and `is_valid_start_`. It deliberately does
**not** build `next_codes_`/`final_codes_`: they are consumed only by
`RecodeBeamSearch`, which is L2b.

**Files:**
- Create: `internal/tessdata/recoder.go`
- Test: `internal/tessdata/recoder_test.go`

**Interfaces:**
- Consumes: `Reader` (Task 2), `Container.Entry(TypeLSTMRecoder)`.
- Produces:

```go
// codeKey is a comparable form of a code sequence, usable as a map key.
type codeKey struct {
	n    int
	code [maxCodeLen]int32
}

type Recoder struct {
	codes      [][]int32
	codeRange  int
	maxLen     int
	decoder    map[codeKey]int
	validStart []bool
}

func ParseRecoder(data []byte, swap bool) (*Recoder, error)
func (rc *Recoder) Size() int
func (rc *Recoder) CodeRange() int
func (rc *Recoder) MaxCodeLen() int
func (rc *Recoder) Encode(unicharID int) []int32
func (rc *Recoder) DecodeUnichar(code []int32) (int, bool)
func (rc *Recoder) IsValidFirstCode(code int32) bool
```

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/recoder_test.go`:

```go
package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func loadRealRecoder(t *testing.T) *Recoder {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	b, ok := c.Entry(TypeLSTMRecoder)
	if !ok {
		t.Fatal("eng.traineddata has no lstm-recoder component")
	}
	rc, err := ParseRecoder(b, c.Swapped())
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	return rc
}

func TestParseRecoderRealModel(t *testing.T) {
	rc := loadRealRecoder(t)

	if rc.Size() != 112 {
		t.Fatalf("Size() = %d; want 112 (one entry per unichar id)", rc.Size())
	}
	if rc.CodeRange() != 111 {
		t.Fatalf("CodeRange() = %d; want 111", rc.CodeRange())
	}
	// eng's recoder is a near-permutation: every code is a single value, which
	// is why greedy CTC decoding is viable for Latin at all. A model where this
	// is false needs the multi-code lookahead in LSTMRecognizer::DecodeLabel.
	if rc.MaxCodeLen() != 1 {
		t.Errorf("MaxCodeLen() = %d; want 1 for eng", rc.MaxCodeLen())
	}

	// LSTMRecognizer::LoadRecoder's own check.
	if got := rc.Encode(UnicharSpace); len(got) != 1 || got[0] != 0 {
		t.Errorf("Encode(UnicharSpace) = %v; want [0]", got)
	}

	for _, tc := range []struct {
		code []int32
		id   int
	}{
		{[]int32{0}, UnicharSpace},
		{[]int32{1}, 3},   // 'C'
		{[]int32{2}, 4},   // 'H'
		{[]int32{109}, 111}, // 'e' with acute
		// Ids 1 (Joined) and 2 (|Broken|0|1) BOTH encode to 110, and
		// SetupDecoder iterates ids ascending and overwrites, so the higher id
		// wins. 110 is also null_char_.
		{[]int32{110}, UnicharBroken},
	} {
		got, ok := rc.DecodeUnichar(tc.code)
		if !ok || got != tc.id {
			t.Errorf("DecodeUnichar(%v) = %d, %v; want %d, true", tc.code, got, ok, tc.id)
		}
	}
	if _, ok := rc.DecodeUnichar([]int32{999}); ok {
		t.Error("DecodeUnichar([999]) reported a hit; want miss")
	}
	if _, ok := rc.DecodeUnichar(nil); ok {
		t.Error("DecodeUnichar(nil) reported a hit; want miss")
	}

	// Every code value in [0, 111) is a valid first code, because all codes
	// have length 1 and every value is used.
	for c := range int32(111) {
		if !rc.IsValidFirstCode(c) {
			t.Errorf("IsValidFirstCode(%d) = false; want true", c)
		}
	}
	if rc.IsValidFirstCode(111) {
		t.Error("IsValidFirstCode(111) = true; want false (outside the code range)")
	}

	// Both eng models ship a byte-identical recoder; the encoder must be a
	// total function over the unicharset.
	for id := range rc.Size() {
		if len(rc.Encode(id)) == 0 {
			t.Errorf("Encode(%d) is empty", id)
		}
	}
}

// buildRecoder emits the component format: uint32 count, then per entry
// int8 self_normalized, uint32 length, int32 code[length].
func buildRecoder(codes [][]int32) []byte {
	b := binary.LittleEndian.AppendUint32(nil, uint32(len(codes)))
	for _, code := range codes {
		b = append(b, 1) // self_normalized
		b = binary.LittleEndian.AppendUint32(b, uint32(len(code)))
		for _, v := range code {
			b = binary.LittleEndian.AppendUint32(b, uint32(v))
		}
	}
	return b
}

func TestParseRecoderHigherIDWinsOnCollision(t *testing.T) {
	// ids 0,1,2 -> codes 0,5,5. SetupDecoder overwrites in ascending id order.
	rc, err := ParseRecoder(buildRecoder([][]int32{{0}, {5}, {5}}), false)
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	got, ok := rc.DecodeUnichar([]int32{5})
	if !ok || got != 2 {
		t.Errorf("DecodeUnichar([5]) = %d, %v; want 2, true", got, ok)
	}
	if rc.CodeRange() != 6 {
		t.Errorf("CodeRange() = %d; want 6", rc.CodeRange())
	}
}

func TestParseRecoderMultiCode(t *testing.T) {
	// Untested against a real CJK model — this exercises the multi-code layout
	// the format allows but eng never uses.
	rc, err := ParseRecoder(buildRecoder([][]int32{{0}, {1, 2, 3}}), false)
	if err != nil {
		t.Fatalf("ParseRecoder() error = %v", err)
	}
	if rc.MaxCodeLen() != 3 {
		t.Errorf("MaxCodeLen() = %d; want 3", rc.MaxCodeLen())
	}
	if got, ok := rc.DecodeUnichar([]int32{1, 2, 3}); !ok || got != 1 {
		t.Errorf("DecodeUnichar([1 2 3]) = %d, %v; want 1, true", got, ok)
	}
	if _, ok := rc.DecodeUnichar([]int32{1, 2}); ok {
		t.Error("DecodeUnichar([1 2]) reported a hit on a proper prefix; want miss")
	}
	if !rc.IsValidFirstCode(1) || rc.IsValidFirstCode(2) {
		t.Error("IsValidFirstCode must be true only for code[0] values")
	}
}

func TestParseRecoderErrors(t *testing.T) {
	tooLong := binary.LittleEndian.AppendUint32(nil, 1)
	tooLong = append(tooLong, 1)
	tooLong = binary.LittleEndian.AppendUint32(tooLong, 10) // > kMaxCodeLen 9
	if _, err := ParseRecoder(tooLong, false); err == nil {
		t.Error("ParseRecoder() with length 10: want error, got nil")
	}

	trailing := append(buildRecoder([][]int32{{0}}), 0xde, 0xad)
	if _, err := ParseRecoder(trailing, false); err == nil {
		t.Error("ParseRecoder() with trailing bytes: want error, got nil")
	}

	// Space must encode onto itself, or the whole decode is misaligned.
	if _, err := ParseRecoder(buildRecoder([][]int32{{7}, {0}}), false); err == nil {
		t.Error("ParseRecoder() with a garbled space: want error, got nil")
	}

	if _, err := ParseRecoder(buildRecoder([][]int32{{0}, {-1}}), false); err == nil {
		t.Error("ParseRecoder() with a negative code: want error, got nil")
	}

	if _, err := ParseRecoder([]byte{0x01}, false); err == nil {
		t.Error("ParseRecoder() on a truncated header: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Recoder -v`
Expected: FAIL — `undefined: ParseRecoder`.

- [ ] **Step 3: Implement the parser**

Create `internal/tessdata/recoder.go`:

```go
// This file is a Go translation of src/ccutil/unicharcompress.cpp and
// src/ccutil/unicharcompress.h from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import "fmt"

// maxCodeLen is RecodedCharID::kMaxCodeLen, src/ccutil/unicharcompress.h:37.
const maxCodeLen = 9

// codeKey is a comparable form of a code sequence, usable as a map key.
type codeKey struct {
	n    int
	code [maxCodeLen]int32
}

func makeCodeKey(code []int32) (codeKey, bool) {
	if len(code) == 0 || len(code) > maxCodeLen {
		return codeKey{}, false
	}
	k := codeKey{n: len(code)}
	copy(k.code[:], code)
	return k, true
}

// Recoder maps unichar ids to the network's output code sequences and back.
//
// For eng every code has length 1 and the mapping is a near-permutation: the
// only collapse is unichar ids 1 ("Joined") and 2 ("|Broken|0|1") both encoding
// to 110, which is also the CTC blank. CJK and Indic models use genuinely
// multi-code sequences.
type Recoder struct {
	codes      [][]int32
	codeRange  int
	maxLen     int
	decoder    map[codeKey]int
	validStart []bool
}

// ParseRecoder deserializes a .traineddata lstm-recoder component. swap comes
// from Container.Swapped.
func ParseRecoder(data []byte, swap bool) (*Recoder, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	n, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading recoder entry count: %w", err)
	}
	// TFile::DeSerialize(std::vector<T>&) rejects counts above 50,000,000.
	if n > maxStringLen {
		return nil, fmt.Errorf("tessdata: implausible recoder entry count %d", n)
	}
	// That bound alone still lets a 4-byte header ask for ~1.2 GB before a
	// single entry is read. The smallest possible entry is 5 bytes (int8
	// self_normalized + uint32 length, zero codes), so the buffer itself is
	// the tighter bound. ParseDawg has the same guard.
	if int(n) > r.Remaining()/5 {
		return nil, fmt.Errorf("tessdata: recoder declares %d entries but only %d bytes remain", n, r.Remaining())
	}
	rc := &Recoder{codes: make([][]int32, n), decoder: make(map[codeKey]int, n)}
	for i := range rc.codes {
		// self_normalized_ is int8 and is therefore never byte-swapped
		// (FReadEndian only reverses items wider than one byte). It is set to
		// 1 by RecodedCharID's constructor and assigned nowhere else in
		// Tesseract, so it carries no information. Read and discard.
		if _, err := r.Int8(); err != nil {
			return nil, fmt.Errorf("tessdata: recoder entry %d self_normalized: %w", i, err)
		}
		length, err := r.Uint32()
		if err != nil {
			return nil, fmt.Errorf("tessdata: recoder entry %d length: %w", i, err)
		}
		if length > maxCodeLen {
			return nil, fmt.Errorf("tessdata: recoder entry %d length %d exceeds kMaxCodeLen %d", i, length, maxCodeLen)
		}
		code := make([]int32, length)
		for j := range code {
			if code[j], err = r.Int32(); err != nil {
				return nil, fmt.Errorf("tessdata: recoder entry %d code %d: %w", i, j, err)
			}
			if code[j] < 0 {
				return nil, fmt.Errorf("tessdata: recoder entry %d code %d is negative (%d)", i, j, code[j])
			}
		}
		rc.codes[i] = code
		if int(length) > rc.maxLen {
			rc.maxLen = int(length)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the recoder", r.Remaining())
	}

	rc.computeCodeRange()
	if err := rc.setupDecoder(); err != nil {
		return nil, err
	}
	return rc, nil
}

// computeCodeRange mirrors UnicharCompress::ComputeCodeRange: max code + 1.
func (rc *Recoder) computeCodeRange() {
	max := -1
	for _, code := range rc.codes {
		for _, v := range code {
			if int(v) > max {
				max = int(v)
			}
		}
	}
	rc.codeRange = max + 1
}

// setupDecoder mirrors UnicharCompress::SetupDecoder. It iterates unichar ids
// in ascending order and overwrites, so when two ids share a code the HIGHER id
// wins: in eng, ids 1 and 2 both encode to [110], and [110] decodes to 2.
//
// next_codes_ and final_codes_ are deliberately not built. They are consumed
// only by RecodeBeamSearch, which is L2b.
//
// Deliberate divergence for a zero-length entry: Tesseract registers it in
// decoder_ and sets is_valid_start_[code(0)] BEFORE its `if (len == 0) continue`
// (src/ccutil/unicharcompress.cpp:400-408), where code(0) of an empty
// RecodedCharID is 0. We skip it entirely — an empty code sequence cannot be
// decoded, and makeCodeKey refuses it. No shipped model has one.
//
// No range check on code[0] is needed: computeCodeRange has just set codeRange
// to max(all codes)+1 over this same array, so every code is inside [0, range)
// by construction, and negatives were rejected at read time.
func (rc *Recoder) setupDecoder() error {
	rc.validStart = make([]bool, rc.codeRange)
	for id, code := range rc.codes {
		if len(code) == 0 {
			// FReadEndian(..., 0) succeeds, so a zero-length entry is legal.
			continue
		}
		k, ok := makeCodeKey(code)
		if !ok {
			return fmt.Errorf("tessdata: recoder entry %d has an unusable code length %d", id, len(code))
		}
		rc.decoder[k] = id
		rc.validStart[code[0]] = true
	}
	// LSTMRecognizer::LoadRecoder's own check (src/lstm/lstmrecognizer.cpp:198):
	// "Space was garbled in recoding!!"
	if len(rc.codes) <= UnicharSpace || len(rc.codes[UnicharSpace]) == 0 ||
		rc.codes[UnicharSpace][0] != UnicharSpace {
		return fmt.Errorf("tessdata: space was garbled in recoding: unichar %d encodes to %v", UnicharSpace, rc.Encode(UnicharSpace))
	}
	return nil
}

// Size is the number of entries, which is one per unichar id.
func (rc *Recoder) Size() int { return len(rc.codes) }

// CodeRange is the number of distinct code values, and therefore the network's
// output count. It is the authority: NetworkBuilder::ParseOutput overrides the
// spec string's output count with it.
func (rc *Recoder) CodeRange() int { return rc.codeRange }

// MaxCodeLen is the longest code sequence in the recoder. 1 means the mapping
// is flat and a decoder can skip the multi-code lookahead entirely.
func (rc *Recoder) MaxCodeLen() int { return rc.maxLen }

// Encode returns the code sequence for a unichar id, or nil if out of range.
// The slice aliases the Recoder; do not modify it.
func (rc *Recoder) Encode(unicharID int) []int32 {
	if unicharID < 0 || unicharID >= len(rc.codes) {
		return nil
	}
	return rc.codes[unicharID]
}

// DecodeUnichar returns the unichar id for a complete code sequence.
func (rc *Recoder) DecodeUnichar(code []int32) (int, bool) {
	k, ok := makeCodeKey(code)
	if !ok {
		return 0, false
	}
	id, ok := rc.decoder[k]
	return id, ok
}

// IsValidFirstCode reports whether a code value can begin a sequence.
func (rc *Recoder) IsValidFirstCode(code int32) bool {
	if code < 0 || int(code) >= len(rc.validStart) {
		return false
	}
	return rc.validStart[code]
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Recoder -v`
Expected: PASS.

- [ ] **Step 5: Record the untested path**

`TestParseRecoderMultiCode` exercises the multi-code layout with a synthetic
fixture only. The layout was inferred from `RecodedCharID::Serialize` and
confirmed by consuming eng's 1012-byte component to exactly 0 bytes, but no
model with `length > 1` was available during planning. Note this in the task
report; it is the input to L1d's decision about whether to ship the length-1
fast path with a `MaxCodeLen() > 1` guard.

- [ ] **Step 6: Commit**

```bash
git add internal/tessdata/recoder.go internal/tessdata/recoder_test.go
git commit -m "feat(tessdata): parse the lstm-recoder component"
```

---

## Task 7: `SquishedDawg` reader

Three DAWG lexicons ship with eng: `lstm-punc-dawg` (18), `lstm-word-dawg` (19),
`lstm-number-dawg` (20). `SquishedDawg::read_squished_dawg`
(`src/dict/dawg.cpp:340`):

```
int16   magic                   ; kDawgMagicNumber == 42
uint32  unicharset_size
uint32  num_edges
uint64  edges[num_edges]        ; EDGE_RECORD
```

Total = `10 + 8*num_edges`. Confirmed on all three eng components:
`10+8*539 == 4322`, `10+8*461848 == 3694794`, `10+8*591 == 4738`.

Bit packing is **derived from `unicharset_size`**, not fixed (`Dawg::init`,
`src/dict/dawg.cpp:188`). For eng (`unicharset_size = 112`):

```
bits [0,7)    unichar_id            letter_mask_ = 0x7f
bit  7        MARKER    = last edge of this node's run
bit  8        DIRECTION = 1 for a backward edge
bit  9        WERD_END  = this edge terminates a word
bits [10,64)  next_node = the edge index of the successor node
```

**The trap:** `flag_start_bit_ = CeilLog2(unicharset_size)`, and Tesseract's
`CeilLog2` is misnamed — it is a **bit length**, not `ceil(log2)`. They agree at
112 (both 7) but differ at exact powers of two: `CeilLog2(64) == 7`, while
`ceil(log2(64)) == 6`. `network.go` already has a `ceilLog2`, which is
`src/lstm/lstm.cpp`'s genuinely-ceil-log2 function. **These are different
functions and must not be shared.**

Structural facts, verified on all three eng components: **every edge is forward**
(`write_squished_dawg` emits forward edges only and rewrites `next_node` through
`build_node_map`), no `next_node` is out of range, no `unichar_id` reaches
`unicharset_size`. A node is identified by the index of its first edge; its edges
are the consecutive run ending at the edge with MARKER set. Node 0 is the root,
so `next_node == 0` means "no successor".

**Files:**
- Create: `internal/tessdata/dawg.go`
- Test: `internal/tessdata/dawg_test.go`

**Interfaces:**
- Consumes: `Reader.Int16`, `Reader.Uint64` (Task 2).
- Produces:

```go
type Dawg struct {
	UnicharsetSize int

	edges            []uint64
	letterMask       uint64
	flagStartBit     uint
	nextNodeStartBit uint
}

func ParseDawg(data []byte, swap bool) (*Dawg, error)
func (d *Dawg) NumEdges() int

// Word-dawg semantics ONLY — see the doc comments in Step 3.
func (d *Dawg) Contains(ids []int) bool
func (d *Dawg) HasPrefix(ids []int) bool
```

**`Contains`/`HasPrefix` are word-dawg-only, and that is a real constraint.**
`Dawg::kPatternUnicharID == 0` (`src/dict/dawg.h:117`): in a
`DAWG_TYPE_PUNCTUATION` or `DAWG_TYPE_NUMBER` dawg, unichar id 0 on an edge is a
**wildcard** standing for "any dictionary word" / "any digit run", not the space
character. These two methods match id 0 literally, so they are meaningful only
on `lstm-word-dawg`. L1a's `Dawg` carries no `DawgType` because nothing in L1a
needs one — `cadmusdump -dawgs` prints only `NumEdges` and `UnicharsetSize`, and
`LoadModel` validates only `UnicharsetSize`. The pattern-dawg machinery
(`DawgPosition`, `Dict::LetterIsOkay`) is L2b's `internal/dict`, and that is
where the type tag belongs. Until then the doc comments must say so loudly.

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/dawg_test.go`:

```go
package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func loadRealDawgs(t *testing.T) (punc, word, number *Dawg, u *Unicharset) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	get := func(typ Type) *Dawg {
		b, ok := c.Entry(typ)
		if !ok {
			t.Fatalf("eng.traineddata has no %v component", typ)
		}
		d, err := ParseDawg(b, c.Swapped())
		if err != nil {
			t.Fatalf("ParseDawg(%v) error = %v", typ, err)
		}
		return d
	}
	ub, _ := c.Entry(TypeLSTMUnicharset)
	u, err = ParseUnicharset(ub)
	if err != nil {
		t.Fatalf("ParseUnicharset() error = %v", err)
	}
	return get(TypeLSTMPuncDawg), get(TypeLSTMSystemDawg), get(TypeLSTMNumberDawg), u
}

func TestParseDawgRealModel(t *testing.T) {
	punc, word, number, _ := loadRealDawgs(t)
	for _, tc := range []struct {
		name  string
		d     *Dawg
		edges int
	}{
		{"punc", punc, 539},
		{"word", word, 461848},
		{"number", number, 591},
	} {
		if tc.d.NumEdges() != tc.edges {
			t.Errorf("%s dawg NumEdges() = %d; want %d", tc.name, tc.d.NumEdges(), tc.edges)
		}
		if tc.d.UnicharsetSize != 112 {
			t.Errorf("%s dawg UnicharsetSize = %d; want 112", tc.name, tc.d.UnicharsetSize)
		}
	}
}

// Words verified against the real lstm-word-dawg during planning by walking the
// edge array directly. If one of these fails, suspect the unichar-id mapping in
// the test helper before suspecting the walk.
func TestDawgContainsRealWords(t *testing.T) {
	_, word, _, u := loadRealDawgs(t)

	byText := map[string]int{}
	for id := range u.Size() {
		byText[u.Text(id)] = id
	}
	ids := func(s string) []int {
		out := make([]int, 0, len(s))
		for _, r := range s {
			id, ok := byText[string(r)]
			if !ok {
				t.Fatalf("%q: no unichar id for %q", s, string(r))
			}
			out = append(out, id)
		}
		return out
	}

	for _, w := range []string{"the", "and", "zebra", "hello", "Cadmus", "a", "I", "of"} {
		if !word.Contains(ids(w)) {
			t.Errorf("Contains(%q) = false; want true", w)
		}
	}
	for _, w := range []string{"qqqqq", "tesseract"} {
		if word.Contains(ids(w)) {
			t.Errorf("Contains(%q) = true; want false", w)
		}
	}
	if word.Contains(nil) {
		t.Error("Contains(nil) = true; want false")
	}
	// "th" is a prefix of "the" but is not itself a word.
	if !word.HasPrefix(ids("th")) {
		t.Error("HasPrefix(\"th\") = false; want true")
	}
	if word.Contains(ids("th")) {
		t.Error("Contains(\"th\") = true; want false")
	}
}

// Dawg::init uses CeilLog2, a BIT LENGTH. network.go's ceilLog2 is
// src/lstm/lstm.cpp's ceil_log2, a genuine ceil(log2). They differ at exact
// powers of two and confusing them shifts every mask by one bit.
func TestDawgFlagStartBitIsBitLength(t *testing.T) {
	for _, tc := range []struct{ size, want int }{
		{1, 1}, {63, 6}, {64, 7}, {112, 7}, {128, 8},
	} {
		d := &Dawg{UnicharsetSize: tc.size}
		d.initMasks()
		if int(d.flagStartBit) != tc.want {
			t.Errorf("flagStartBit for unicharset_size %d = %d; want %d", tc.size, d.flagStartBit, tc.want)
		}
	}
	if got := ceilLog2(64); got != 6 {
		t.Errorf("ceilLog2(64) = %d; want 6 — the LSTM one must stay ceil(log2)", got)
	}
}

// buildDawg emits the component header plus raw edge records.
func buildDawg(unicharsetSize int, edges []uint64) []byte {
	b := binary.LittleEndian.AppendUint16(nil, 42)
	b = binary.LittleEndian.AppendUint32(b, uint32(unicharsetSize))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(edges)))
	for _, e := range edges {
		b = binary.LittleEndian.AppendUint64(b, e)
	}
	return b
}

// makeEdge packs one record for a 112-entry unicharset (flag_start_bit 7).
func makeEdge(id int, last, eow bool, next int) uint64 {
	e := uint64(id)
	if last {
		e |= 1 << 7
	}
	if eow {
		e |= 4 << 7
	}
	return e | uint64(next)<<10
}

func TestParseDawgTinyGraph(t *testing.T) {
	// Node 0: 'a'(id 1) -> node 2, last. Node 1 unused filler.
	// Node 2: 'b'(id 2), eow, last.
	edges := []uint64{
		makeEdge(1, true, false, 2),
		makeEdge(9, true, false, 0),
		makeEdge(2, true, true, 0),
	}
	d, err := ParseDawg(buildDawg(112, edges), false)
	if err != nil {
		t.Fatalf("ParseDawg() error = %v", err)
	}
	if !d.Contains([]int{1, 2}) {
		t.Error("Contains([1 2]) = false; want true")
	}
	if d.Contains([]int{1}) {
		t.Error("Contains([1]) = true; want false — 'a' alone is not word-end")
	}
	if !d.HasPrefix([]int{1}) {
		t.Error("HasPrefix([1]) = false; want true")
	}
	if d.Contains([]int{1, 3}) {
		t.Error("Contains([1 3]) = true; want false")
	}
}

func TestParseDawgErrors(t *testing.T) {
	good := []uint64{makeEdge(1, true, true, 0)}

	bad := buildDawg(112, good)
	bad[0] = 43 // wrong magic
	if _, err := ParseDawg(bad, false); err == nil {
		t.Error("ParseDawg() with magic 43: want error, got nil")
	}

	if _, err := ParseDawg(buildDawg(112, nil), false); err == nil {
		t.Error("ParseDawg() with 0 edges: want error, got nil")
	}

	if _, err := ParseDawg(buildDawg(0, good), false); err == nil {
		t.Error("ParseDawg() with unicharset_size 0: want error, got nil")
	}

	if _, err := ParseDawg(append(buildDawg(112, good), 0xde), false); err == nil {
		t.Error("ParseDawg() with a trailing byte: want error, got nil")
	}

	truncated := buildDawg(112, good)
	if _, err := ParseDawg(truncated[:len(truncated)-2], false); err == nil {
		t.Error("ParseDawg() with a truncated edge: want error, got nil")
	}

	// A backward edge cannot appear in a written dawg.
	backward := []uint64{makeEdge(1, true, true, 0) | 2<<7}
	if _, err := ParseDawg(buildDawg(112, backward), false); err == nil {
		t.Error("ParseDawg() with a backward edge: want error, got nil")
	}

	// next_node beyond the edge array.
	if _, err := ParseDawg(buildDawg(112, []uint64{makeEdge(1, true, true, 99)}), false); err == nil {
		t.Error("ParseDawg() with an out-of-range next_node: want error, got nil")
	}

	// unichar_id at or beyond unicharset_size. Size 1 (not 4) is deliberate:
	// makeEdge hardcodes flag_start_bit = 7, so with size 4 (bit length 3,
	// letter mask 0x7) the id reads back as 1 < 4 and it is the NEXT_NODE check
	// that fires, not the id check. With size 1 the mask is 0x1, the id reads
	// back as 1 >= 1, and the assertion tests what it claims to test.
	if _, err := ParseDawg(buildDawg(1, []uint64{makeEdge(1, true, true, 0)}), false); err == nil {
		t.Error("ParseDawg() with unichar_id >= unicharset_size: want error, got nil")
	}
}

// Node 0 is the ONLY node Tesseract binary-searches (SquishedDawg::edge_char_of,
// src/dict/dawg.cpp:207); every other node it linear-scans. edgeOf linear-scans
// everywhere, which is equivalent only while node 0's edges carry no repeated
// unichar id — the binary search's comparator orders on
// (unichar_id, next_node, word_end), so the format permits two node-0 edges with
// the same id (one with WERD_END clear, one set), and on such a node the two
// searches can return different edges and therefore different next_nodes.
//
// Only the word dawg's node 0 was measured during planning (67 edges, sorted,
// no duplicates). Punctuation dawgs are exactly where duplicate ids are most
// likely, so this test measures all three. If it ever fails, do NOT relax it:
// implement the binary search for node 0 in edgeOf and fix its comment.
//
// node0UnicharIDs lives here rather than in dawg.go because nothing outside
// this test uses it; the test file is in package tessdata and so reaches the
// unexported fields directly.
func (d *Dawg) node0UnicharIDs() []int {
	var ids []int
	for e := 0; e < len(d.edges); e++ {
		ids = append(ids, int(d.edges[e]&d.letterMask))
		if d.edges[e]&(dawgMarkerFlag<<d.flagStartBit) != 0 {
			break
		}
	}
	return ids
}

func TestDawgNode0IsSortedAndDuplicateFree(t *testing.T) {
	punc, word, number, _ := loadRealDawgs(t)
	for _, tc := range []struct {
		name string
		d    *Dawg
	}{
		{"punc", punc},
		{"word", word},
		{"number", number},
	} {
		ids := tc.d.node0UnicharIDs()
		t.Logf("%s dawg node 0: %d edges", tc.name, len(ids))
		if len(ids) == 0 {
			t.Errorf("%s dawg node 0 has no edges", tc.name)
			continue
		}
		for i := 1; i < len(ids); i++ {
			if ids[i] < ids[i-1] {
				t.Errorf("%s dawg node 0 edge %d id %d < edge %d id %d; node 0 must be sorted ascending for the binary search edgeOf replaces",
					tc.name, i, ids[i], i-1, ids[i-1])
			}
			if ids[i] == ids[i-1] {
				t.Errorf("%s dawg node 0 has duplicate unichar id %d at edges %d and %d; a linear scan is NOT equivalent to Tesseract's binary search here",
					tc.name, ids[i], i-1, i)
			}
		}
	}
	if got := len(word.node0UnicharIDs()); got != 67 {
		t.Errorf("word dawg node 0 has %d edges; want 67 (measured during planning)", got)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Dawg -v`
Expected: FAIL — `undefined: ParseDawg`.

- [ ] **Step 3: Implement the reader**

Create `internal/tessdata/dawg.go`:

```go
// This file is a Go translation of src/dict/dawg.cpp and src/dict/dawg.h from
// Tesseract OCR (https://github.com/tesseract-ocr/tesseract), licensed under
// the Apache License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"fmt"
	"math"
	"math/bits"
)

// dawgMagic is kDawgMagicNumber, src/dict/dawg.h:113.
const dawgMagic int16 = 42

// Edge-record flags, all shifted left by flagStartBit. src/dict/dawg.h:77.
const (
	dawgMarkerFlag    = 1 // last edge of this node's run
	dawgDirectionFlag = 2 // set for a backward edge
	dawgWordEndFlag   = 4
	dawgNumFlagBits   = 3
)

// maxDawgEdges is read_squished_dawg's resource-exhaustion guard.
const maxDawgEdges = 50_000_000

// Dawg is a SquishedDawg: a directed acyclic word graph, as stored in the
// lstm-punc-dawg, lstm-word-dawg and lstm-number-dawg components.
//
// A node is identified by the index of its first edge; its edges are the
// consecutive run ending at the edge whose MARKER flag is set. Node 0 is the
// root, so a next_node of 0 means "no successor" rather than "back to the
// root" — see Trie::add_word_to_dawg.
type Dawg struct {
	UnicharsetSize int

	edges            []uint64
	letterMask       uint64
	flagStartBit     uint
	nextNodeStartBit uint
}

// ParseDawg deserializes one lstm-*-dawg component. swap comes from
// Container.Swapped.
func ParseDawg(data []byte, swap bool) (*Dawg, error) {
	r := NewReader(data)
	r.SetSwap(swap)

	magic, err := r.Int16()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg magic: %w", err)
	}
	if magic != dawgMagic {
		return nil, fmt.Errorf("tessdata: bad dawg magic %d, want %d", magic, dawgMagic)
	}
	size, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg unicharset size: %w", err)
	}
	if size == 0 || size > math.MaxInt32 {
		return nil, fmt.Errorf("tessdata: bad dawg unicharset size %d", size)
	}
	numEdges, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading dawg edge count: %w", err)
	}
	if numEdges == 0 {
		return nil, fmt.Errorf("tessdata: dawg has 0 edges")
	}
	if numEdges > maxDawgEdges {
		return nil, fmt.Errorf("tessdata: dawg edge count %d exceeds the %d hard limit", numEdges, maxDawgEdges)
	}
	if int(numEdges) > r.Remaining()/8 {
		return nil, fmt.Errorf("tessdata: dawg declares %d edges but only %d bytes remain", numEdges, r.Remaining())
	}

	d := &Dawg{UnicharsetSize: int(size), edges: make([]uint64, numEdges)}
	for i := range d.edges {
		if d.edges[i], err = r.Uint64(); err != nil {
			return nil, fmt.Errorf("tessdata: reading dawg edge %d: %w", i, err)
		}
	}
	if r.Remaining() != 0 {
		return nil, fmt.Errorf("tessdata: %d bytes left unconsumed after the dawg", r.Remaining())
	}
	d.initMasks()
	if err := d.validate(); err != nil {
		return nil, err
	}
	return d, nil
}

// initMasks mirrors Dawg::init (src/dict/dawg.cpp:188).
//
// flagStartBit is the BIT LENGTH of unicharset_size. Tesseract's CeilLog2 is
// misnamed: it counts bits, so CeilLog2(64) == 7 while ceil(log2(64)) == 6.
// This is NOT the same function as ceilLog2 in network.go, which is
// src/lstm/lstm.cpp's ceil_log2. Confusing them shifts every mask by one bit.
func (d *Dawg) initMasks() {
	d.flagStartBit = uint(bits.Len(uint(d.UnicharsetSize)))
	d.nextNodeStartBit = d.flagStartBit + dawgNumFlagBits
	d.letterMask = 1<<d.flagStartBit - 1
}

// validate enforces the invariants SquishedDawg::write_squished_dawg
// guarantees: it emits forward edges only, and rewrites every next_node through
// build_node_map so it indexes the written array. Verified on all three
// lstm-*-dawg components of tessdata_best eng.
func (d *Dawg) validate() error {
	for i, e := range d.edges {
		if e&(dawgDirectionFlag<<d.flagStartBit) != 0 {
			return fmt.Errorf("tessdata: dawg edge %d is a backward edge; only written (forward-only) dawgs are supported", i)
		}
		if id := int(e & d.letterMask); id >= d.UnicharsetSize {
			return fmt.Errorf("tessdata: dawg edge %d has unichar id %d outside the %d-entry unicharset", i, id, d.UnicharsetSize)
		}
		if next := int(e >> d.nextNodeStartBit); next >= len(d.edges) {
			return fmt.Errorf("tessdata: dawg edge %d points at node %d beyond the %d edges in the file", i, next, len(d.edges))
		}
	}
	return nil
}

func (d *Dawg) NumEdges() int { return len(d.edges) }

func (d *Dawg) nextNode(edge int) int { return int(d.edges[edge] >> d.nextNodeStartBit) }

// edgeOf mirrors SquishedDawg::edge_char_of (src/dict/dawg.cpp:207).
//
// Tesseract binary-searches node 0 and linear-scans every other node. edgeOf
// linear-scans everywhere, which is a simplification, not an identity: the
// binary search's comparator (given_greater_than_edge_rec, src/dict/dawg.h:242)
// orders on (unichar_id, next_node, word_end), so the format permits node 0 to
// hold two edges with the same unichar id, and on such a node the two searches
// can pick different edges.
//
// What was measured: tessdata_best eng's lstm-word-dawg node 0 has 67 edges,
// sorted ascending, with no repeated unichar id. What enforces it for all three
// lexicons at test time: TestDawgNode0IsSortedAndDuplicateFree. If that test
// ever fails on a model, this function must grow the binary search for node 0
// rather than the test being relaxed.
func (d *Dawg) edgeOf(node, id int, wordEnd bool) int {
	for e := node; e < len(d.edges); e++ {
		rec := d.edges[e]
		if int(rec&d.letterMask) == id &&
			(!wordEnd || rec&(dawgWordEndFlag<<d.flagStartBit) != 0) {
			return e
		}
		if rec&(dawgMarkerFlag<<d.flagStartBit) != 0 {
			break
		}
	}
	return -1
}

// Contains reports whether the dawg holds the complete word spelled by ids.
// It is Dawg::word_in_dawg, i.e. prefixInDawg with requiresComplete = true.
//
// WORD DAWGS ONLY. Every unichar id on an edge is matched literally, which is
// correct for lstm-word-dawg (DAWG_TYPE_WORD) and WRONG for lstm-punc-dawg and
// lstm-number-dawg: in those, Dawg::kPatternUnicharID (== 0, src/dict/dawg.h:117)
// on an edge is a WILDCARD meaning "any dictionary word" / "any digit run", not
// the space character. Matching it literally silently mis-answers both of those
// lexicons. L1a has no consumer that needs pattern semantics — cadmusdump prints
// only edge counts — so the wildcard machinery (DawgPosition,
// Dict::LetterIsOkay) is deferred to L2b's internal/dict along with the
// DawgType tag that would make these methods safe to call on all three.
func (d *Dawg) Contains(ids []int) bool { return d.prefixInDawg(ids, true) }

// HasPrefix reports whether ids is a prefix of at least one word in the dawg.
// It is Dawg::prefix_in_dawg with requires_complete = false.
//
// WORD DAWGS ONLY, for the same reason as Contains.
func (d *Dawg) HasPrefix(ids []int) bool { return d.prefixInDawg(ids, false) }

// prefixInDawg mirrors Dawg::prefix_in_dawg (src/dict/dawg.cpp:42).
func (d *Dawg) prefixInDawg(ids []int, requiresComplete bool) bool {
	if len(ids) == 0 {
		return !requiresComplete
	}
	node := 0
	for _, id := range ids[:len(ids)-1] {
		e := d.edgeOf(node, id, false)
		if e < 0 {
			return false
		}
		// next_node == 0 means every word through this edge terminates here;
		// there are no longer words. See Trie::add_word_to_dawg.
		if node = d.nextNode(e); node == 0 {
			return false
		}
	}
	return d.edgeOf(node, ids[len(ids)-1], requiresComplete) >= 0
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run Dawg -v`
Expected: PASS, including `TestDawgContainsRealWords` and
`TestDawgNode0IsSortedAndDuplicateFree` (whose `t.Logf` lines record node 0's
edge count for all three lexicons — copy them into the task report; only the
word dawg's 67 was known before this task).

**If `TestDawgNode0IsSortedAndDuplicateFree` fails** on punc or number: that is
the one measurement this plan could not make in advance. Do not relax the test.
Implement `SquishedDawg::edge_char_of`'s node-0 binary search in `edgeOf`
(`src/dict/dawg.cpp:207-226`, comparator at `src/dict/dawg.h:242`) and keep the
test as a record of why.

**If a word assertion fails:** dump the walk before touching the assertion. The
in/out sets were verified during planning by walking the real edge array; the
likely cause is the test's unichar-id mapping (for example a multi-rune entry
colliding in `byText`), not the DAWG walk.

- [ ] **Step 5: Cross-check against the Tesseract oracle**

```bash
cd /tmp/cadmus-comp
ls -l eng.lstm-punc-dawg eng.lstm-word-dawg eng.lstm-number-dawg
```

Expected sizes 4322, 3694794, 4738 — i.e. `10 + 8*num_edges` for 539, 461848 and
591 edges, matching the Go test. If `combine_tessdata` is unavailable, the Go
test already asserts the edge counts.

- [ ] **Step 6: Commit**

```bash
git add internal/tessdata/dawg.go internal/tessdata/dawg_test.go
git commit -m "feat(tessdata): read the SquishedDawg lexicon components"
```

---

## Task 8: `LoadModel` and the cross-component invariants

Individually-correct components can still be mutually inconsistent. This task is
where the four numbers that bind them get asserted in one place:

```
unicharset size ........ 112   ids 0..111
recoder entries ........ 112   one per unichar id
recoder code_range ..... 111   codes 0..110, contiguous
Softmax no ............. 111   ==  code_range
null_char_ ............. 110   ==  code_range - 1  ==  the LAST output index
```

112 collapses to 111 because unichar ids 1 (`Joined`) and 2 (`|Broken|0|1`) both
encode to code 110, and `DefragmentCodeValues` deliberately relocates the null
code to the top of the range.

**Files:**
- Create: `internal/tessdata/model.go`
- Test: `internal/tessdata/model_test.go`

**Interfaces:**
- Consumes: everything from Tasks 3–7.
- Produces:

```go
type Model struct {
	Version    string
	Swapped    bool
	Recognizer *Recognizer
	Unicharset *Unicharset
	Recoder    *Recoder
	PuncDawg   *Dawg // nil when the component is absent
	WordDawg   *Dawg // nil when the component is absent
	NumberDawg *Dawg // nil when the component is absent
}

func LoadModel(data []byte) (*Model, error)
func (m *Model) Softmax() *Layer
func (m *Model) NumOutputs() int
func (m *Model) NullChar() int
```

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/model_test.go`:

```go
package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadRealModel(t *testing.T) *Model {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	m, err := LoadModel(raw)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	return m
}

// The L1a deliverable, asserted in one place.
func TestLoadModelRealModel(t *testing.T) {
	m := loadRealModel(t)

	if !strings.HasPrefix(m.Version, "4.00.00alpha:eng:") {
		t.Errorf("Version = %q; want a 4.00.00alpha:eng: prefix", m.Version)
	}
	if m.Swapped {
		t.Error("Swapped = true; the fixture is little-endian")
	}

	// The chain that binds the components together.
	if got := m.Unicharset.Size(); got != 112 {
		t.Errorf("Unicharset.Size() = %d; want 112", got)
	}
	if got := m.Recoder.Size(); got != m.Unicharset.Size() {
		t.Errorf("Recoder.Size() = %d; want %d (one entry per unichar id)", got, m.Unicharset.Size())
	}
	if got := m.Recoder.CodeRange(); got != 111 {
		t.Errorf("Recoder.CodeRange() = %d; want 111", got)
	}
	if got := m.NumOutputs(); got != m.Recoder.CodeRange() {
		t.Errorf("NumOutputs() = %d; want %d", got, m.Recoder.CodeRange())
	}
	if got := m.NullChar(); got != 110 {
		t.Errorf("NullChar() = %d; want 110", got)
	}

	sm := m.Softmax()
	if sm == nil {
		t.Fatal("Softmax() = nil")
	}
	if sm.Name != "Output" || sm.NumOutputs != 111 || sm.NumInputs != 512 {
		t.Errorf("Softmax = %q ni=%d no=%d; want \"Output\" ni=512 no=111", sm.Name, sm.NumInputs, sm.NumOutputs)
	}
	// Print SHAPES, never the matrices themselves: %v on sm.Matrices would dump
	// 56,943 float64s into the test log.
	if len(sm.Matrices) != 1 {
		t.Fatalf("Softmax has %d matrices; want one 111x513", len(sm.Matrices))
	}
	if sm.Matrices[0].Rows != 111 || sm.Matrices[0].Cols != 513 {
		t.Fatalf("Softmax matrix = %dx%d; want 111x513", sm.Matrices[0].Rows, sm.Matrices[0].Cols)
	}

	for _, tc := range []struct {
		name  string
		d     *Dawg
		edges int
	}{
		{"punc", m.PuncDawg, 539},
		{"word", m.WordDawg, 461848},
		{"number", m.NumberDawg, 591},
	} {
		if tc.d == nil {
			t.Errorf("%s dawg is nil; want loaded", tc.name)
			continue
		}
		if tc.d.NumEdges() != tc.edges {
			t.Errorf("%s dawg NumEdges() = %d; want %d", tc.name, tc.d.NumEdges(), tc.edges)
		}
	}

	// Every weight in the model is loaded. 1461007 is the root layer's own
	// num_weights, cross-checked recursively during parsing.
	total := 0
	var walk func(*Layer)
	walk = func(l *Layer) {
		for i := range l.Matrices {
			total += len(l.Matrices[i].Values)
		}
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(m.Recognizer.Network)
	if total != 1461007 {
		t.Errorf("loaded %d weight values; want 1461007", total)
	}
}

func TestLoadModelRejectsMissingComponents(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "eng.traineddata"))
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	// A container holding only the version string: every required component
	// is absent, so LoadModel must refuse rather than return a half-model.
	ver, _ := c.Entry(TypeVersion)
	only := buildContainer(t, map[Type][]byte{TypeVersion: ver})
	if _, err := LoadModel(only); err == nil {
		t.Fatal("LoadModel() without an lstm component: want error, got nil")
	}
}

func TestLoadModelRejectsGarbage(t *testing.T) {
	if _, err := LoadModel([]byte{0xde, 0xad, 0xbe, 0xef}); err == nil {
		t.Fatal("LoadModel() on garbage: want error, got nil")
	}
}
```

(`buildContainer` already exists in `container_test.go`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -run LoadModel -v`
Expected: FAIL — `undefined: LoadModel`.

- [ ] **Step 3: Implement `LoadModel`**

Create `internal/tessdata/model.go`:

```go
// This file is a Go translation of src/lstm/lstmrecognizer.cpp and
// src/dict/dict.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import "fmt"

// Model is a fully loaded .traineddata: everything the LSTM recognition path
// reads, parsed into memory with every weight value present.
type Model struct {
	Version    string
	Swapped    bool
	Recognizer *Recognizer
	Unicharset *Unicharset
	Recoder    *Recoder

	// The three lexicons Dict::LoadLSTM reads (src/dict/dict.cpp:292). nil when
	// the component is absent; a lexicon-free model still recognizes text, it
	// just loses the dictionary-weighted beam.
	PuncDawg   *Dawg
	WordDawg   *Dawg
	NumberDawg *Dawg
}

// LoadModel parses a .traineddata file in Tesseract's native container layout
// and validates the invariants that bind its components together.
//
// The LSTM path requires TESSDATA_LSTM (17), TESSDATA_LSTM_UNICHARSET (21) and
// TESSDATA_LSTM_RECODER (22). When 21 and 22 are both present,
// LSTMRecognizer::DeSerialize reads the charsets from the components rather
// than from inside the LSTM blob; the embedded-charset layout is a pre-4.0
// arrangement no stock model uses, and cadmus does not implement it — a model
// missing either component is rejected rather than silently mis-parsed.
func LoadModel(data []byte) (*Model, error) {
	c, err := ParseContainer(data)
	if err != nil {
		return nil, fmt.Errorf("tessdata: parsing container: %w", err)
	}
	m := &Model{Version: c.Version(), Swapped: c.Swapped()}

	required := func(t Type) ([]byte, error) {
		b, ok := c.Entry(t)
		if !ok {
			return nil, fmt.Errorf("tessdata: model has no %v component", t)
		}
		return b, nil
	}

	lstm, err := required(TypeLSTM)
	if err != nil {
		return nil, err
	}
	if m.Recognizer, err = ParseRecognizer(lstm, c.Swapped()); err != nil {
		return nil, fmt.Errorf("tessdata: lstm component: %w", err)
	}

	ucs, err := required(TypeLSTMUnicharset)
	if err != nil {
		return nil, err
	}
	if m.Unicharset, err = ParseUnicharset(ucs); err != nil {
		return nil, fmt.Errorf("tessdata: lstm-unicharset component: %w", err)
	}

	rec, err := required(TypeLSTMRecoder)
	if err != nil {
		return nil, err
	}
	if m.Recoder, err = ParseRecoder(rec, c.Swapped()); err != nil {
		return nil, fmt.Errorf("tessdata: lstm-recoder component: %w", err)
	}

	for _, l := range []struct {
		typ Type
		dst **Dawg
	}{
		{TypeLSTMPuncDawg, &m.PuncDawg},
		{TypeLSTMSystemDawg, &m.WordDawg},
		{TypeLSTMNumberDawg, &m.NumberDawg},
	} {
		b, ok := c.Entry(l.typ)
		if !ok {
			continue
		}
		d, err := ParseDawg(b, c.Swapped())
		if err != nil {
			return nil, fmt.Errorf("tessdata: %v component: %w", l.typ, err)
		}
		*l.dst = d
	}

	if err := m.validate(); err != nil {
		return nil, err
	}
	return m, nil
}

// validate enforces the cross-component invariants. Each one is free to check
// and each one catches a whole class of mis-parse.
func (m *Model) validate() error {
	if m.Recoder.Size() != m.Unicharset.Size() {
		return fmt.Errorf("tessdata: recoder has %d entries but the unicharset has %d; they are 1:1", m.Recoder.Size(), m.Unicharset.Size())
	}

	sm := m.Softmax()
	if sm == nil {
		return fmt.Errorf("tessdata: network has no softmax output layer")
	}
	// LSTMTrainer::InitNetwork builds the network with recoder_.code_range() as
	// its output count, so these must agree exactly.
	if sm.NumOutputs != m.Recoder.CodeRange() {
		return fmt.Errorf("tessdata: softmax layer has %d outputs but the recoder's code range is %d", sm.NumOutputs, m.Recoder.CodeRange())
	}
	if m.Recognizer.Network.NumOutputs != sm.NumOutputs {
		return fmt.Errorf("tessdata: root layer declares %d outputs but the softmax layer has %d", m.Recognizer.Network.NumOutputs, sm.NumOutputs)
	}

	// UnicharCompress::DefragmentCodeValues deliberately relocates the null
	// code to the top of the range, so null_char_ is the LAST output index.
	// null_char_ is the authority; if this ever fires on a real model, prefer
	// the field and relax the check rather than deriving the blank from the
	// output count.
	if int(m.Recognizer.NullChar) != m.Recoder.CodeRange()-1 {
		return fmt.Errorf("tessdata: null_char is %d but the code range is %d; expected the blank at the last output index", m.Recognizer.NullChar, m.Recoder.CodeRange())
	}

	for _, d := range []*Dawg{m.PuncDawg, m.WordDawg, m.NumberDawg} {
		if d != nil && d.UnicharsetSize != m.Unicharset.Size() {
			return fmt.Errorf("tessdata: a dawg was built for a %d-entry unicharset but the model's has %d", d.UnicharsetSize, m.Unicharset.Size())
		}
	}
	return nil
}

// Softmax returns the network's unique softmax output layer, or nil if the
// graph does not have exactly one. Its NumOutputs is the authoritative output
// count — never the number in the spec string, which ParseOutput overrides.
func (m *Model) Softmax() *Layer {
	var found *Layer
	n := 0
	var walk func(*Layer)
	walk = func(l *Layer) {
		switch l.Type {
		case LayerSoftmax, LayerSoftmaxNoCTC:
			n++
			found = l
		}
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(m.Recognizer.Network)
	if n != 1 {
		return nil
	}
	return found
}

// NumOutputs is the network's output count, and therefore the width of one
// timestep of the network's output.
func (m *Model) NumOutputs() int { return m.Recognizer.Network.NumOutputs }

// NullChar is the network output index of the CTC blank.
func (m *Model) NullChar() int { return int(m.Recognizer.NullChar) }
```

**Note the `Softmax()` multiple-match handling.** It counts, deliberately. An
earlier draft used a `found = nil` sentinel with an early `return` on the second
hit; that is wrong twice over — a third softmax finds `found == nil` again and
resurrects it, and the early `return` skips the duplicate's children. The count
has neither failure mode: `n != 1` is exactly the "graph does not have exactly
one softmax" condition the doc comment promises.

`n == 2` is reachable on a real (non-eng) model: `NT_LSTM_SOFTMAX` /
`NT_LSTM_SOFTMAX_ENCODED` nest a `FullyConnected` softmax inside the LSTM's
`Children`, so such a model has the nested one plus the network's own output
layer. `Softmax()` returning nil there — and `validate()` therefore rejecting the
model — is the correct L1a behaviour: cadmus does not implement that topology,
and failing loudly is the plan's stated rule. Add a comment saying so, so a later
reader does not "fix" it into returning the outermost one.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./internal/tessdata/ -v`
Expected: PASS, all suites.

- [ ] **Step 5: Commit**

```bash
CGO_ENABLED=0 go vet ./...
git add internal/tessdata/model.go internal/tessdata/model_test.go
git commit -m "feat(tessdata): add LoadModel and the cross-component invariants

Binds unicharset size, recoder entry count, recoder code range, softmax
output count and null_char into one validated Model."
```

---

## Task 9: `cadmusdump` — make the weights visible to a human

Automated invariants catch structure. They do not catch "the numbers are
plausible-looking garbage". A human reading per-matrix statistics, the unicharset
table and the recoder map is the single most valuable check available at this
stage, and it costs one afternoon of CLI work.

**Files:**
- Modify: `cmd/cadmusdump/main.go`
- Modify: `cmd/cadmusdump/main_test.go`

**Interfaces:**
- Consumes: `tessdata.LoadModel`, `Layer.Tree`, `Matrix.Stats`.
- Produces: `cadmusdump [-unicharset] [-recoder] [-dawgs] <model.traineddata>`.

- [ ] **Step 1: Write the failing test**

Replace `cmd/cadmusdump/main_test.go`:

```go
package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	return path
}

func TestDumpDefaultSections(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"lstm-unicharset",
		"lstm-recoder",
		"null_char      110",
		"sample_iter    814136",
		"[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]",
		`Series "Series" ni=36 no=111 weights=1461007`,
		`Softmax "Output" ni=512 no=111 weights=56943`,
		"111x513 min=-29.279424 max=35.155323",
		`LSTM "Lfx512" ni=96 no=512 na=608`,
		"[2] GF1 512x609",
		"half_x=1 half_y=1",
		"x_scale=3 y_scale=3",
		"shape batch=1 height=36 width=0 depth=1 loss=0",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("dump output missing %q\n---\n%s", want, out)
		}
	}
	// Opt-in sections must stay off by default.
	for _, notWant := range []string{"unicharset:", "recoder:", "dawgs:"} {
		if strings.Contains(out, notWant) {
			t.Errorf("dump output contains %q without the flag", notWant)
		}
	}
}

func TestDumpUnicharsetSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{unicharset: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"unicharset: 112 entries",
		`    0 " "`,
		`    2 "|Broken|0|1"`,
		"props=f",
		"normed=\"'\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("unicharset section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpRecoderSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{recoder: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{
		"recoder: 112 entries, code range 111, max code length 1",
		// dumpRecoder prints the code SLICE with %v, so the rendering is
		// "-> code [110]", not "-> code 110".
		"-> code [110]",
		"BLANK",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("recoder section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpDawgSection(t *testing.T) {
	var b strings.Builder
	if err := dump(fixture(t), options{dawgs: true}, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{"dawgs:", "word", "461848 edges"} {
		if !strings.Contains(out, want) {
			t.Errorf("dawg section missing %q\n---\n%s", want, out)
		}
	}
}

func TestDumpMissingFileErrors(t *testing.T) {
	var b strings.Builder
	if err := dump("does-not-exist.traineddata", options{}, &b); err == nil {
		t.Fatal("dump() on missing file: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CGO_ENABLED=0 go test ./cmd/cadmusdump/ -v`
Expected: FAIL — `undefined: options`, `too many arguments in call to dump`.

- [ ] **Step 3: Implement the CLI**

Rewrite `cmd/cadmusdump/main.go`:

```go
// Command cadmusdump prints the contents of a Tesseract .traineddata file: its
// component inventory, the recognizer's header fields, the layer tree with
// per-matrix weight statistics, and optionally the unicharset, the recoder
// mapping and the DAWG lexicons.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

type options struct {
	unicharset bool
	recoder    bool
	dawgs      bool
}

func dump(path string, opt options, w io.Writer) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	c, err := tessdata.ParseContainer(raw)
	if err != nil {
		return fmt.Errorf("parsing container: %w", err)
	}
	fmt.Fprintf(w, "%s\nversion: %s\nbyte-swapped: %v\n\ncomponents:\n", path, c.Version(), c.Swapped())
	for _, t := range c.Present() {
		b, _ := c.Entry(t)
		fmt.Fprintf(w, "  %-20s %9d bytes\n", t, len(b))
	}

	// Preserved from the pre-L1a cadmusdump: a legacy (Tesseract-3, no LSTM)
	// .traineddata still has a printable component inventory, and dumping it is
	// useful. LoadModel requires the LSTM component, so bail before calling it
	// rather than turning `cadmusdump some-legacy.traineddata` into exit 1.
	if _, ok := c.Entry(tessdata.TypeLSTM); !ok {
		fmt.Fprintln(w, "\nno lstm component (legacy-only model)")
		return nil
	}

	// LoadModel re-walks the container. That is deliberate: ParseContainer only
	// reads the offset table and re-slices the caller's buffer, so the second
	// walk costs nothing and copies nothing, and keeping LoadModel's signature
	// as ([]byte) -> (*Model, error) keeps the library free of a CLI-shaped
	// "also give me the container" return value.
	m, err := tessdata.LoadModel(raw)
	if err != nil {
		return fmt.Errorf("loading model: %w", err)
	}
	rec := m.Recognizer
	fmt.Fprintf(w, `
recognizer:
  spec           %s
  training_flags %d
  iteration      %d
  sample_iter    %d
  null_char      %d
  adam_beta      %g
  learning_rate  %g
  momentum       %g

network:
`, rec.NetworkStr, rec.TrainingFlags, rec.TrainingIteration,
		rec.SampleIteration, rec.NullChar, rec.AdamBeta, rec.LearningRate, rec.Momentum)
	rec.Network.Tree(w)

	if opt.unicharset {
		dumpUnicharset(w, m)
	}
	if opt.recoder {
		dumpRecoder(w, m)
	}
	if opt.dawgs {
		dumpDawgs(w, m)
	}
	return nil
}

func dumpUnicharset(w io.Writer, m *tessdata.Model) {
	u := m.Unicharset
	fmt.Fprintf(w, "\nunicharset: %d entries\n", u.Size())
	for id := range u.Size() {
		c, _ := u.Char(id)
		fmt.Fprintf(w, "  %3d %-14q props=%x script=%-8s", id, c.Text, c.Properties, c.Script)
		if c.Normed != c.Text {
			fmt.Fprintf(w, " normed=%q", c.Normed)
		}
		fmt.Fprintln(w)
	}
}

func dumpRecoder(w io.Writer, m *tessdata.Model) {
	rc := m.Recoder
	fmt.Fprintf(w, "\nrecoder: %d entries, code range %d, max code length %d\n",
		rc.Size(), rc.CodeRange(), rc.MaxCodeLen())
	for id := range rc.Size() {
		code := rc.Encode(id)
		fmt.Fprintf(w, "  unichar %3d %-14q -> code %v", id, m.Unicharset.Text(id), code)
		if len(code) == 1 && int(code[0]) == m.NullChar() {
			fmt.Fprint(w, "  BLANK")
		}
		fmt.Fprintln(w)
	}
}

func dumpDawgs(w io.Writer, m *tessdata.Model) {
	fmt.Fprintln(w, "\ndawgs:")
	for _, d := range []struct {
		name string
		d    *tessdata.Dawg
	}{
		{"punc", m.PuncDawg},
		{"word", m.WordDawg},
		{"number", m.NumberDawg},
	} {
		if d.d == nil {
			fmt.Fprintf(w, "  %-8s absent\n", d.name)
			continue
		}
		fmt.Fprintf(w, "  %-8s %9d edges, unicharset size %d\n", d.name, d.d.NumEdges(), d.d.UnicharsetSize)
	}
}

func main() {
	var opt options
	flag.BoolVar(&opt.unicharset, "unicharset", false, "print the unicharset table")
	flag.BoolVar(&opt.recoder, "recoder", false, "print the unichar -> output-code mapping")
	flag.BoolVar(&opt.dawgs, "dawgs", false, "print the DAWG lexicon summaries")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: cadmusdump [-unicharset] [-recoder] [-dawgs] <model.traineddata>")
		os.Exit(2)
	}
	if err := dump(flag.Arg(0), opt, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cadmusdump:", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CGO_ENABLED=0 go test ./cmd/cadmusdump/ -v`
Expected: PASS.

- [ ] **Step 5: Read the output as a human — this is the point of the task**

```bash
go run ./cmd/cadmusdump testdata/eng.traineddata
go run ./cmd/cadmusdump -unicharset testdata/eng.traineddata | sed -n '/unicharset:/,+15p'
go run ./cmd/cadmusdump -recoder testdata/eng.traineddata | sed -n '/recoder:/,+8p'
go run ./cmd/cadmusdump -dawgs testdata/eng.traineddata | sed -n '/dawgs:/,+4p'
```

Check, by eye, all four of:

1. **Weights.** Every matrix's mean is within ~0.05 of zero and its range
   straddles zero. `Output` is the wide outlier at `[-29.28, 35.16]`. The full
   expected tree is in the "Facts established by research" section — compare
   against it.
2. **Unicharset.** Entry 0 is `" "`, entries 1 and 2 are `Joined` and
   `|Broken|0|1`, `props=f` appears exactly once, and the six `normed=` entries
   are the curly quotes, em dash and trade-mark sign.
3. **Recoder.** 112 entries, code range 111, max code length 1, and `BLANK`
   appears on exactly two lines — unichar 1 and unichar 2, both encoding to 110.
4. **Dawgs.** 539 / 461848 / 591 edges, all with unicharset size 112.

If any of these looks wrong, the fault is in the loader, not the printer.

- [ ] **Step 6: Commit**

```bash
CGO_ENABLED=0 go vet ./... && CGO_ENABLED=0 go test ./...
git add cmd/cadmusdump
git commit -m "feat(cmd): dump weight statistics, the unicharset, the recoder and the dawgs

cadmusdump now loads the whole model and prints per-matrix min/max/mean,
which is how a human confirms the weights are real numbers rather than a
misaligned payload."
```

---

## Final step: verify and close

- [ ] **Run the full gate**

```bash
cd /Users/christopherdobbyn/work/dobbo-ca/cadmus
make build && make test && make lint
```

- [ ] **Prove the suite passes without the fixture**

The fixture is 15 MB and gitignored; CI never fetches it. Every fixture-dependent
test must skip, not fail.

```bash
mv testdata/eng.traineddata /tmp/eng.traineddata.bak
CGO_ENABLED=0 go test ./... 2>&1 | tail -20
mv /tmp/eng.traineddata.bak testdata/eng.traineddata
```

Expected: `ok` for every package, with skip messages naming `./testdata/fetch.sh`.
A FAIL here means a test uses `t.Fatal` where it should use `t.Skipf`.

- [ ] **Prove no dependency crept in**

```bash
test ! -s go.sum && echo "go.sum empty: ok"
CGO_ENABLED=0 go build ./... && echo "builds without cgo: ok"
```

- [ ] **Record the outcome**

`cad-l1a` is the bead Task 1 Step 0 created (substitute the id it handed back if
`--id` was refused). `cad-jgq` was closed in Task 4 Step 8; confirm with
`bd show cad-jgq` before closing this one.

```bash
bd update cad-l1a --append-notes "$(go run ./cmd/cadmusdump testdata/eng.traineddata)"
bd update cad-l1a --append-notes "L1a complete: tessdata_best 4.1.0 eng.traineddata fully loaded. 1461007 float weight values, 112-entry unicharset, 112-entry recoder (code range 111, all length 1), three dawgs (539/461848/591 edges). Cross-component invariants asserted: softmax no == recoder code_range == null_char+1 == 111, recoder size == unicharset size == 112. int8 models and training dumps are hard errors."
bd close cad-l1a
```

---

## Self-Review

**Scope coverage.** The task brief asks for: `fetch.sh` switched to
`tessdata_best` with a float assertion (Task 1 + the assertions in Tasks 3 and 4),
`GENERIC_2D_ARRAY` float deserialization (Task 3), `ParseNetwork` extended to load
values (Task 3), unicharset parsing (Task 5), the recoder (Task 6), the DAWG
lexicons (Task 7), and `cadmusdump` printing the unicharset, the recoder mapping
and weight statistics (Task 9). Tasks 2 (reader primitives) and 8 (`LoadModel`)
are the connective tissue: Task 2 exists because three parsers need primitives the
spike did not, and Task 8 exists because the cross-component invariants are the
only place the four components can be checked against each other.

**Out of scope, and kept out.** No forward pass, no activation tables, no image
normalization, no CTC decode, no beam search, no `next_codes_`/`final_codes_`, no
int8 path, no `DeSerializeOld`. `Matrix.Stats` is the only arithmetic on weights
and exists solely so a human can see the numbers are real.

**Placeholder scan.** Every *implementation* code block (each task's Step 3) is
complete and compilable as written. Every test block contains real assertions
against values measured from the actual model during planning. There is no "add
error handling", no "similar to Task N", no "TBD". The Interfaces blocks are
summaries and give the full struct definitions for `Unicharset`, `Recoder` and
`Dawg` alongside their method sets; where a block shows unexported fields, Step 3
is still the authority on the surrounding doc comments. Values discovered at
execution time rather than stated here: `combine_tessdata`'s availability (every
step that uses it says what to do when it is missing), and node 0's edge count
for the punc and number dawgs (Task 7's
`TestDawgNode0IsSortedAndDuplicateFree` measures and asserts it — see the
uncertainty list below).

**Where research was uncertain, and what the plan does instead of guessing.**

1. **`empty_` is 0.0 in every eng matrix, but nothing in the format requires it.**
   Task 3 reads and discards it and explicitly does not assert its value. The
   code carries a comment saying so, to stop a later reader adding the assertion.
2. **`num_weights` recursion is verified from source for every layer type,
   including the ones eng does not use.** `LSTM::InitWeights`
   (`src/lstm/lstm.cpp:175`) sums the gate matrices and then adds
   `softmax_->InitWeights()` when a nested softmax is present;
   `Plumbing::InitWeights` (`src/lstm/plumbing.cpp:50`) sums its stack;
   `FullyConnected::InitWeights` (`src/lstm/fullyconnected.cpp:86`) is
   `no_*(ni_+1)`, i.e. rows×cols. So the recursive identity Task 3(h) asserts
   holds by construction, not just empirically on eng. The citation is in the
   code comment.
3. **The 2-D LSTM `GFS` skip rule is read from source, never exercised** — eng has
   no 2-D layer. The existing, already-shipped `parseLSTM` logic is left
   unchanged; Task 3 does not touch it.
4. **The nested-softmax LSTM topology (`NT_LSTM_SOFTMAX`,
   `NT_LSTM_SOFTMAX_ENCODED`) is unexercised and is REJECTED, not supported.**
   Such a model has two softmax layers, so `Model.Softmax()` counts 2, returns
   nil, and `validate()` fails. That is the plan's "fail loudly on what we do not
   implement" rule applied consistently; Task 8 Step 3 says so explicitly so it
   is not later "fixed" into silently picking one.
5. **Byte-swapped (foreign-endian) models are unexercised.** Every new parser
   threads `Container.Swapped()` through and Task 2 tests each new primitive in
   both byte orders, but no real big-endian model was available. Stated, not
   claimed as verified.
6. **The recoder's multi-code layout was inferred from `RecodedCharID::Serialize`
   and confirmed only by byte-exact consumption of eng's all-length-1 component.**
   Task 6 Step 5 is an explicit instruction to record this as untested in the task
   report, and `TestParseRecoderMultiCode` exercises it with a synthetic fixture so
   the code path at least runs.
7. **The unicharset's 255-byte line cap.** `TFile::FGets`
   (`src/ccutil/serialis.cpp:213`) copies at most 255 bytes and does **not**
   discard the rest of an over-long line — the remainder is returned as the next
   entry line, so every subsequent unichar id shifts and the file
   desynchronises. The Go parser reads whole lines. Task 5 makes this a *failing
   test* rather than a silent divergence:
   `TestUnicharsetLinesFitTesseractsBuffer` logs the longest line (82 bytes for
   eng) and fails above 255.
8. **`IsSpaceDelimited` compares script *names*, Tesseract compares resolved
   script *ids*** — and Tesseract's `get_script_id_from_name` collapses unknown
   names onto the null script, so a unicharset without a Han entry reports
   null-script characters as not space delimited. Task 5 documents this as a
   deliberate divergence and Task 5's real-model test asserts eng has no
   null-script entry, so the two agree on the model at hand.
9. **Node 0's edge shape is measured for the word dawg only.** `edgeOf`
   linear-scans every node, including node 0, which Tesseract binary-searches.
   The two agree only while node 0 carries no duplicate unichar id, and the
   binary search's comparator orders on `(unichar_id, next_node, word_end)`, so
   duplicates are a shape the format permits — most plausibly in a punctuation
   dawg. Only `lstm-word-dawg`'s node 0 (67 edges, sorted, duplicate-free) was
   walked during planning. Task 7's
   `TestDawgNode0IsSortedAndDuplicateFree` turns this into an executable check
   over all three lexicons, with an explicit instruction to implement the node-0
   binary search rather than relax the test if it fires.
10. **`Contains`/`HasPrefix` are word-dawg-only.** `Dawg::kPatternUnicharID == 0`
    is a wildcard in punctuation and number dawgs, and these methods match id 0
    literally. Nothing in L1a calls them on those two lexicons (`cadmusdump
    -dawgs` prints only counts, `LoadModel` validates only `UnicharsetSize`), so
    L1a carries no `DawgType` field; Task 7 documents the constraint on both
    methods and names L2b's `internal/dict` as where the pattern machinery and
    the type tag belong.
11. **`makeEdge` in the DAWG test hardcodes `flag_start_bit = 7`.** The
    unichar-id-out-of-range case therefore uses `unicharset_size = 1`, not 4: at
    size 4 the letter mask is `0x7`, the id reads back as `1 < 4`, and it is the
    *next_node* check that fires. Task 7 Step 1 states this in the test comment.

**Uncertainties the research raised that this plan *resolved* rather than
deferring** — each was checked against the real model during planning and is now
stated as fact, with the check that would falsify it:

- The tessdata_best download is pinnable: tag `4.1.0` yields the byte-identical
  file `main` does, so `fetch.sh` pins by tag *and* sha256 (Task 1).
- The 15 MB fixture is already covered by `.gitignore`'s `*.traineddata`; Task 1
  Step 4 verifies rather than assumes.
- There are **no committed layer-tree goldens to regenerate** — both existing test
  files assert structural properties only. Task 1 Step 5 states the expected diff.
- `XScaleFactor == 3` is now read out of the serialized `Maxpool` layer
  (`x_scale=3 y_scale=3`) rather than derived from the spec string (Task 3).
- Node 0 of the real word-dawg has 67 edges, sorted, with no duplicate unichar
  ids, so linear scan is equivalent to Tesseract's binary search **on that
  lexicon**. The punc and number dawgs were not walked; Task 7's node-0 test
  closes that gap at execution time rather than the plan asserting it.
- The unicharset accepts a count of 0 in Tesseract and is rejected here. Task 5
  lists it with the other deliberate divergences instead of leaving it as an
  unexplained stricter check.

**Type consistency.** `Matrix` and `InputShape` are produced in Task 3 and
consumed by `Layer.Tree` (Task 3) and `cadmusdump` (Task 9). `Recognizer` is
produced in Task 4 and consumed by `Model` (Task 8) and `cadmusdump`.
`Unicharset`, `Recoder` and `Dawg` are produced in Tasks 5–7 and consumed by
`Model`. `Reader.Int16`/`Uint64` (Task 2) are used only by `ParseDawg`;
`Reader.Float32` only by `ParseRecognizer`. `ParseNetwork` is removed in Task 4
and its three call sites updated in the same commit. `UnicharSpace` is defined in
Task 5 and used by Task 6's `setupDecoder`; `ParseUnicharset` is defined in Task
5 and used by Task 7's `loadRealDawgs` test helper. **Both make Task 5 a
compile-time prerequisite of Tasks 6 and 7** — the "Task dependencies" line at
the top of this plan states it, and two agents working 5/6/7 "concurrently" would
otherwise produce non-compiling branches.

**Cross-plan consistency with L1b.** L1b Task 1
(`docs/superpowers/plans/2026-07-27-cadmus-l1b-forward-decode.md`) adds the same
geometry fields to the same `Layer` struct. Task 3 of this plan adopts L1b's
spelling verbatim — `type InputShape struct{Batch, Height, Width, Depth,
LossType int}`, `Shape *InputShape`, `NA int`, plus `HalfX/HalfY/XScale/YScale` —
so the two plans cannot collide on a field-name-versus-type-name clash. Once L1a
lands, L1b Task 1 is already done: reduce it to running
`TestParseNetworkRetainsLayerGeometry` (a subset of
`TestParseNetworkRealModelWeights`) as a verification step, and make it reuse
this plan's `loadRealNetwork`/`loadRealRecognizer` helper rather than declaring
its own `parseRealModel` in the same package. Task 3 repeats this instruction in
place so it is seen by whoever executes either plan.

**Pre-existing code this plan deliberately does not touch.**
`Reader.Float64Slice` in `tfile.go` has no callers and is not used by L1a. It is
pre-existing, not an orphan created here, so it stays. Mentioned so a later
cleanup pass has the context.
