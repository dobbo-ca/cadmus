# Cadmus L0 + Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Cadmus image core (`cad-l0`) and prove `.traineddata` deserialization is tractable (`cad-spike`).

**Architecture:** Two independent tracks over shared scaffolding. The spike track parses Tesseract's `.traineddata` container and LSTM network graph, producing a `cadmusdump` CLI that prints the layer tree — retiring the project's largest unknown before any of L1 is committed to. The L0 track builds the pure-Go image primitives (binarization, morphology, connected components, deskew, spatial index) that every detector depends on, verified against committed golden files generated from Leptonica.

**Tech Stack:** Go 1.26, standard library only. Test-only external oracles: `tesseract` and `leptonica` (via Homebrew), used to generate committed goldens — never linked, never shipped, never required in CI.

## Global Constraints

- **Go 1.26.** `go.mod` declares `go 1.26`.
- **Module path:** `github.com/dobbo-ca/cadmus`
- **Zero third-party dependencies in non-test code.** `go.mod` must have an empty `require` block for the life of this plan. If a task seems to need a dependency, stop and raise it.
- **No cgo.** All packages must build with `CGO_ENABLED=0`. CI enforces this.
- **Apache-2.0.** Every file that is a translation of Tesseract source carries a header naming the Tesseract source file it derives from (see Task 1 for the exact template). Files that are original work carry no such header.
- **Oracles are test-only.** `tesseract` and Leptonica may be invoked by golden-*generation* tooling, never by tests themselves. `go test ./...` must pass on a machine with neither installed.
- **Package layout:** implementation lives under `internal/`. Nothing is exported from the module root until L1.
- All bitmap coordinates are (x, y) with origin top-left, matching Leptonica and `image.Rectangle`.

---

## File Structure

| File | Responsibility |
|---|---|
| `go.mod`, `Makefile`, `.golangci.yml`, `.github/workflows/ci.yml` | Scaffolding (Task 1) |
| `internal/tessdata/tfile.go` | Endian-aware primitive reader mirroring Tesseract's `TFile` |
| `internal/tessdata/container.go` | `.traineddata` container: entry table + slicing |
| `internal/tessdata/network.go` | LSTM network graph deserialization (structure + shapes) |
| `cmd/cadmusdump/main.go` | Spike deliverable: print container inventory + layer tree |
| `internal/imaging/bitmap.go` | `Bitmap` type (1bpp and 8bpp), the shared substrate for L0 |
| `internal/imaging/binarize.go` | Otsu and Sauvola thresholding |
| `internal/imaging/rasterop.go` | Bitwise raster operations between bitmap rects |
| `internal/imaging/morph.go` | Brick dilate/erode/open/close |
| `internal/imaging/conncomp.go` | Connected-component labelling |
| `internal/imaging/seedfill.go` | Binary seedfill and distance transform |
| `internal/imaging/deskew.go` | Skew angle estimation and rotation |
| `internal/imaging/grid.go` | `bbgrid` spatial index over bounding boxes |
| `testdata/golden/gen/` | C harness that regenerates goldens from Leptonica (manual, not CI) |

**Track independence:** Tasks 2-5 (spike) and Tasks 6-12 (L0) share only Task 1. They can be executed concurrently by separate workers.

---

## Task 1: Repository scaffolding

**Files:**
- Create: `go.mod`, `Makefile`, `.golangci.yml`, `.github/workflows/ci.yml`, `internal/doc.go`, `testdata/fetch.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: a module named `github.com/dobbo-ca/cadmus` that builds and tests clean; `make test`, `make lint`, `make goldens`; `testdata/eng.traineddata` on disk after `./testdata/fetch.sh`.

- [ ] **Step 1: Install the test-only oracles**

Neither is present on this machine. Both are needed to *generate* fixtures and goldens; neither is needed to run tests afterwards.

```bash
brew install tesseract leptonica
tesseract --version          # expect 5.x
ls "$(brew --prefix)/share/tessdata/eng.traineddata"
```

- [ ] **Step 2: Create the module**

```bash
cd /Users/christopherdobbyn/work/dobbo-ca/cadmus
go mod init github.com/dobbo-ca/cadmus
```

Then edit `go.mod` so it reads exactly:

```
module github.com/dobbo-ca/cadmus

go 1.26
```

- [ ] **Step 3: Add the fixture fetch script**

Create `testdata/fetch.sh`:

```bash
#!/usr/bin/env bash
# Fetches the model fixtures used by tests. Not run in CI; committed goldens
# cover CI. Run this once locally before working on internal/tessdata.
set -euo pipefail
cd "$(dirname "$0")"

TESSDATA_SRC="$(brew --prefix 2>/dev/null)/share/tessdata/eng.traineddata"
if [ -f "$TESSDATA_SRC" ]; then
  cp "$TESSDATA_SRC" eng.traineddata
  echo "copied eng.traineddata from $TESSDATA_SRC"
else
  curl -fsSL -o eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
  echo "downloaded eng.traineddata from tessdata_best"
fi
ls -l eng.traineddata
```

```bash
chmod +x testdata/fetch.sh && ./testdata/fetch.sh
```

`*.traineddata` is already in `.gitignore` — the fixture is fetched, never committed.

- [ ] **Step 4: Add the Makefile**

Create `Makefile`:

```make
.PHONY: test lint build goldens

test:
	CGO_ENABLED=0 go test ./...

lint:
	CGO_ENABLED=0 go vet ./...
	golangci-lint run

build:
	CGO_ENABLED=0 go build ./...

# Regenerates Leptonica goldens. Requires leptonica + a C compiler.
# Manual step — never run in CI. Commit the results.
goldens:
	$(MAKE) -C testdata/golden/gen
```

- [ ] **Step 5: Add the linter config**

Create `.golangci.yml`, matching the convention used in `kleio`:

```yaml
version: "2"
linters:
  enable:
    - errcheck
    - govet
    - ineffassign
    - staticcheck
    - unused
```

- [ ] **Step 6: Add the license header template**

Create `internal/doc.go`:

```go
// Package internal holds Cadmus implementation packages.
//
// Files in this tree that are Go translations of Tesseract source carry a
// header of the form:
//
//	// This file is a Go translation of <path> from Tesseract OCR
//	// (https://github.com/tesseract-ocr/tesseract), licensed under the
//	// Apache License, Version 2.0. The translation is not verbatim.
//
// Files that are original work carry no such header. See NOTICE.
package internal
```

- [ ] **Step 7: Add CI**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      CGO_ENABLED: 0
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.26'
      - run: go build ./...
      - run: go vet ./...
      - run: go test ./...
      - name: assert zero dependencies
        run: |
          if [ -s go.sum ]; then
            echo "go.sum is non-empty; a dependency was added"
            exit 1
          fi
```

- [ ] **Step 8: Verify the scaffolding**

```bash
make build && make test
```

Expected: both succeed. `go test ./...` reports `no test files` for now — that is a pass.

- [ ] **Step 9: Commit**

```bash
git add go.mod Makefile .golangci.yml .github internal/doc.go testdata/fetch.sh
git commit -m "chore: scaffold the Go module, CI, and fixture fetching"
```

---

## Task 2: TFile primitive reader

Tesseract serializes with a small set of fixed-width primitives through its `TFile`
class, with a whole-file byte-swap flag decided by the container header. Every
later parsing task sits on this.

**Files:**
- Create: `internal/tessdata/tfile.go`
- Test: `internal/tessdata/tfile_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces:

```go
type Reader struct { /* ... */ }
func NewReader(data []byte) *Reader
func (r *Reader) SetSwap(swap bool)
func (r *Reader) Int8() (int8, error)
func (r *Reader) Uint8() (uint8, error)
func (r *Reader) Int32() (int32, error)
func (r *Reader) Uint32() (uint32, error)
func (r *Reader) Int64() (int64, error)
func (r *Reader) Float64() (float64, error)
func (r *Reader) String() (string, error)
func (r *Reader) Bytes(n int) ([]byte, error)
func (r *Reader) Float64Slice() ([]float64, error)
func (r *Reader) Remaining() int
```

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/tfile_test.go`:

```go
package tessdata

import "testing"

func TestReaderPrimitivesLittleEndian(t *testing.T) {
	// uint32(1), int32(-2), int8(7)
	data := []byte{
		0x01, 0x00, 0x00, 0x00,
		0xfe, 0xff, 0xff, 0xff,
		0x07,
	}
	r := NewReader(data)

	if got, err := r.Uint32(); err != nil || got != 1 {
		t.Fatalf("Uint32() = %d, %v; want 1, nil", got, err)
	}
	if got, err := r.Int32(); err != nil || got != -2 {
		t.Fatalf("Int32() = %d, %v; want -2, nil", got, err)
	}
	if got, err := r.Int8(); err != nil || got != 7 {
		t.Fatalf("Int8() = %d, %v; want 7, nil", got, err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
}

func TestReaderSwapReversesMultiByteReads(t *testing.T) {
	r := NewReader([]byte{0x00, 0x00, 0x00, 0x01})
	r.SetSwap(true)
	got, err := r.Uint32()
	if err != nil || got != 1 {
		t.Fatalf("Uint32() with swap = %d, %v; want 1, nil", got, err)
	}
}

// String is a uint32 length followed by that many raw bytes. Tesseract rejects
// lengths above 50,000,000 as corruption; so do we.
func TestReaderString(t *testing.T) {
	data := []byte{0x03, 0x00, 0x00, 0x00, 'a', 'b', 'c'}
	r := NewReader(data)
	got, err := r.String()
	if err != nil || got != "abc" {
		t.Fatalf("String() = %q, %v; want \"abc\", nil", got, err)
	}
}

func TestReaderStringRejectsAbsurdLength(t *testing.T) {
	data := []byte{0xff, 0xff, 0xff, 0xff}
	r := NewReader(data)
	if _, err := r.String(); err == nil {
		t.Fatal("String() with 4294967295-byte length: want error, got nil")
	}
}

func TestReaderTruncatedInputErrors(t *testing.T) {
	r := NewReader([]byte{0x01, 0x02})
	if _, err := r.Uint32(); err == nil {
		t.Fatal("Uint32() on 2-byte input: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/tessdata/ -run TestReader -v`
Expected: FAIL — `undefined: NewReader`.

- [ ] **Step 3: Implement the reader**

Create `internal/tessdata/tfile.go`:

```go
// This file is a Go translation of src/ccutil/serialis.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"encoding/binary"
	"fmt"
	"math"
)

// maxStringLen mirrors Tesseract's arbitrary corruption guard in
// TFile::DeSerialize(std::string&).
const maxStringLen = 50_000_000

// Reader reads Tesseract's serialization primitives from an in-memory buffer.
// Tesseract writes in host byte order and detects a foreign-endian file at the
// container header, then byte-swaps every subsequent multi-byte read; SetSwap
// reproduces that.
type Reader struct {
	data []byte
	pos  int
	swap bool
}

func NewReader(data []byte) *Reader { return &Reader{data: data} }

func (r *Reader) SetSwap(swap bool) { r.swap = swap }

func (r *Reader) Remaining() int { return len(r.data) - r.pos }

func (r *Reader) Bytes(n int) ([]byte, error) {
	if n < 0 {
		return nil, fmt.Errorf("tessdata: negative read length %d", n)
	}
	if r.Remaining() < n {
		return nil, fmt.Errorf("tessdata: short read at offset %d: want %d bytes, have %d", r.pos, n, r.Remaining())
	}
	b := r.data[r.pos : r.pos+n]
	r.pos += n
	return b, nil
}

func (r *Reader) Uint8() (uint8, error) {
	b, err := r.Bytes(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *Reader) Int8() (int8, error) {
	v, err := r.Uint8()
	return int8(v), err
}

func (r *Reader) Uint32() (uint32, error) {
	b, err := r.Bytes(4)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(b)
	if r.swap {
		v = bits32Reverse(v)
	}
	return v, nil
}

func (r *Reader) Int32() (int32, error) {
	v, err := r.Uint32()
	return int32(v), err
}

func (r *Reader) Int64() (int64, error) {
	b, err := r.Bytes(8)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(b)
	if r.swap {
		v = bits64Reverse(v)
	}
	return int64(v), nil
}

func (r *Reader) Float64() (float64, error) {
	b, err := r.Bytes(8)
	if err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(b)
	if r.swap {
		v = bits64Reverse(v)
	}
	return math.Float64frombits(v), nil
}

func (r *Reader) String() (string, error) {
	n, err := r.Uint32()
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("tessdata: implausible string length %d at offset %d", n, r.pos)
	}
	if n == 0 {
		return "", nil
	}
	b, err := r.Bytes(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Float64Slice reads a uint32 count followed by that many float64 values.
func (r *Reader) Float64Slice() ([]float64, error) {
	n, err := r.Uint32()
	if err != nil {
		return nil, err
	}
	if int(n) > r.Remaining()/8 {
		return nil, fmt.Errorf("tessdata: float64 count %d exceeds %d remaining bytes", n, r.Remaining())
	}
	out := make([]float64, n)
	for i := range out {
		if out[i], err = r.Float64(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func bits32Reverse(v uint32) uint32 {
	return v>>24 | (v>>8)&0x0000ff00 | (v<<8)&0x00ff0000 | v<<24
}

func bits64Reverse(v uint64) uint64 {
	return v>>56 | (v>>40)&0x000000000000ff00 | (v>>24)&0x0000000000ff0000 |
		(v>>8)&0x00000000ff000000 | (v<<8)&0x000000ff00000000 |
		(v<<24)&0x0000ff0000000000 | (v<<40)&0x00ff000000000000 | v<<56
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./internal/tessdata/ -run TestReader -v`
Expected: PASS, all five tests.

- [ ] **Step 5: Commit**

```bash
git add internal/tessdata/tfile.go internal/tessdata/tfile_test.go
git commit -m "feat(tessdata): add the TFile primitive reader"
```

---

## Task 3: `.traineddata` container parser

The container is a flat entry table. From `TessdataManager::LoadMemBuffer`:

```
uint32              num_entries      ; if > 1000 the file is foreign-endian → swap everything
int64[num_entries]  offset_table     ; -1 means the entry is absent
<entry bytes, concatenated>
```

Entry *i* spans from `offset_table[i]` to the next non-negative offset, or to
end-of-file if there is none. `kMaxNumTessdataEntries = 1000`.

**Files:**
- Create: `internal/tessdata/container.go`
- Test: `internal/tessdata/container_test.go`

**Interfaces:**
- Consumes: `Reader` from Task 2.
- Produces:

```go
type Type int

const (
	TypeLangConfig       Type = 0
	TypeUnicharset       Type = 1
	TypeAmbigs           Type = 2
	TypePuncDawg         Type = 6
	TypeSystemDawg       Type = 7
	TypeNumberDawg       Type = 8
	TypeFreqDawg         Type = 9
	TypeBigramDawg       Type = 14
	TypeUnambigDawg      Type = 15
	TypeLSTM             Type = 17
	TypeLSTMPuncDawg     Type = 18
	TypeLSTMSystemDawg   Type = 19
	TypeLSTMNumberDawg   Type = 20
	TypeLSTMUnicharset   Type = 21
	TypeLSTMRecoder      Type = 22
	TypeVersion          Type = 23
	numTypes             Type = 24
)

func (t Type) String() string

type Container struct{ /* ... */ }
func ParseContainer(data []byte) (*Container, error)
func (c *Container) Entry(t Type) ([]byte, bool)
func (c *Container) Present() []Type
func (c *Container) Version() string
```

- [ ] **Step 1: Write the failing test**

Create `internal/tessdata/container_test.go`:

```go
package tessdata

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// buildContainer produces a synthetic .traineddata in the native format.
// offsets of -1 mark absent entries.
func buildContainer(t *testing.T, entries map[Type][]byte) []byte {
	t.Helper()
	const n = int(numTypes)
	header := 4 + 8*n
	offsets := make([]int64, n)
	var body []byte
	for i := range offsets {
		payload, ok := entries[Type(i)]
		if !ok {
			offsets[i] = -1
			continue
		}
		offsets[i] = int64(header + len(body))
		body = append(body, payload...)
	}
	out := make([]byte, 0, header+len(body))
	out = binary.LittleEndian.AppendUint32(out, uint32(n))
	for _, off := range offsets {
		out = binary.LittleEndian.AppendUint64(out, uint64(off))
	}
	return append(out, body...)
}

func TestParseContainerRoundTrip(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{
		TypeLSTM:           []byte("network-bytes"),
		TypeLSTMUnicharset: []byte("charset"),
		TypeVersion:        []byte("5.0.0-test"),
	})

	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}

	got, ok := c.Entry(TypeLSTM)
	if !ok || string(got) != "network-bytes" {
		t.Fatalf("Entry(TypeLSTM) = %q, %v; want \"network-bytes\", true", got, ok)
	}
	if got, ok := c.Entry(TypeLSTMUnicharset); !ok || string(got) != "charset" {
		t.Fatalf("Entry(TypeLSTMUnicharset) = %q, %v; want \"charset\", true", got, ok)
	}
	if c.Version() != "5.0.0-test" {
		t.Fatalf("Version() = %q; want \"5.0.0-test\"", c.Version())
	}
}

func TestParseContainerAbsentEntry(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{TypeLSTM: []byte("x")})
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	if _, ok := c.Entry(TypeAmbigs); ok {
		t.Fatal("Entry(TypeAmbigs) reported present; want absent")
	}
}

// The last present entry runs to end-of-file rather than to a following offset.
func TestParseContainerLastEntryRunsToEOF(t *testing.T) {
	raw := buildContainer(t, map[Type][]byte{TypeVersion: []byte("tail")})
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer() error = %v", err)
	}
	if got, _ := c.Entry(TypeVersion); string(got) != "tail" {
		t.Fatalf("Entry(TypeVersion) = %q; want \"tail\"", got)
	}
}

func TestParseContainerRejectsTooManyEntries(t *testing.T) {
	// 2000 exceeds kMaxNumTessdataEntries in both byte orders, so it is
	// corruption rather than an endianness signal.
	raw := binary.LittleEndian.AppendUint32(nil, 2000)
	if _, err := ParseContainer(raw); err == nil {
		t.Fatal("ParseContainer() with 2000 entries: want error, got nil")
	}
}

// Real-model check. Skipped when the fixture has not been fetched, so the
// suite still passes on a machine without ./testdata/fetch.sh having been run.
func TestParseContainerRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	c, err := ParseContainer(raw)
	if err != nil {
		t.Fatalf("ParseContainer(eng.traineddata) error = %v", err)
	}
	for _, want := range []Type{TypeLSTM, TypeLSTMUnicharset, TypeLSTMRecoder} {
		if _, ok := c.Entry(want); !ok {
			t.Errorf("Entry(%v) absent from eng.traineddata; want present", want)
		}
	}
	if c.Version() == "" {
		t.Error("Version() is empty; want a version string")
	}
	t.Logf("eng.traineddata components: %v, version %q", c.Present(), c.Version())
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./internal/tessdata/ -run TestParseContainer -v`
Expected: FAIL — `undefined: ParseContainer`.

- [ ] **Step 3: Implement the container parser**

Create `internal/tessdata/container.go`:

```go
// This file is a Go translation of src/ccutil/tessdatamanager.cpp from
// Tesseract OCR (https://github.com/tesseract-ocr/tesseract), licensed under
// the Apache License, Version 2.0. The translation is not verbatim.

package tessdata

import (
	"fmt"
	"strings"
)

// maxNumEntries mirrors kMaxNumTessdataEntries. A count above it in the
// file's own byte order means the file was written on the other endianness.
const maxNumEntries = 1000

type Type int

const (
	TypeLangConfig     Type = 0
	TypeUnicharset     Type = 1
	TypeAmbigs         Type = 2
	TypeIntTemp        Type = 3  // legacy engine
	TypePFFMTable      Type = 4  // legacy engine
	TypeNormProto      Type = 5  // legacy engine
	TypePuncDawg       Type = 6
	TypeSystemDawg     Type = 7
	TypeNumberDawg     Type = 8
	TypeFreqDawg       Type = 9
	TypeShapeTable     Type = 13 // legacy engine
	TypeBigramDawg     Type = 14
	TypeUnambigDawg    Type = 15
	TypeParamsModel    Type = 16
	TypeLSTM           Type = 17
	TypeLSTMPuncDawg   Type = 18
	TypeLSTMSystemDawg Type = 19
	TypeLSTMNumberDawg Type = 20
	TypeLSTMUnicharset Type = 21
	TypeLSTMRecoder    Type = 22
	TypeVersion        Type = 23

	numTypes Type = 24
)

var typeNames = map[Type]string{
	TypeLangConfig: "lang_config", TypeUnicharset: "unicharset", TypeAmbigs: "ambigs",
	TypeIntTemp: "inttemp", TypePFFMTable: "pffmtable", TypeNormProto: "normproto",
	TypePuncDawg: "punc_dawg", TypeSystemDawg: "system_dawg", TypeNumberDawg: "number_dawg",
	TypeFreqDawg: "freq_dawg", TypeShapeTable: "shapetable", TypeBigramDawg: "bigram_dawg",
	TypeUnambigDawg: "unambig_dawg", TypeParamsModel: "params_model", TypeLSTM: "lstm",
	TypeLSTMPuncDawg: "lstm-punc-dawg", TypeLSTMSystemDawg: "lstm-word-dawg",
	TypeLSTMNumberDawg: "lstm-number-dawg", TypeLSTMUnicharset: "lstm-unicharset",
	TypeLSTMRecoder: "lstm-recoder", TypeVersion: "version",
}

func (t Type) String() string {
	if n, ok := typeNames[t]; ok {
		return n
	}
	return fmt.Sprintf("type(%d)", int(t))
}

// Container is a parsed .traineddata file. Entry slices alias the input buffer.
type Container struct {
	entries map[Type][]byte
	swap    bool
}

// ParseContainer parses Tesseract's native .traineddata layout.
//
// Note: Tesseract built with libarchive also accepts zip/tar archives of
// component files. The models published in tessdata, tessdata_best, and
// tessdata_fast all use the native layout parsed here.
func ParseContainer(data []byte) (*Container, error) {
	r := NewReader(data)
	n, err := r.Uint32()
	if err != nil {
		return nil, fmt.Errorf("tessdata: reading entry count: %w", err)
	}
	swap := n > maxNumEntries
	if swap {
		n = bits32Reverse(n)
		r.SetSwap(true)
	}
	if n > maxNumEntries {
		return nil, fmt.Errorf("tessdata: entry count %d exceeds maximum %d in both byte orders", n, maxNumEntries)
	}

	offsets := make([]int64, n)
	for i := range offsets {
		if offsets[i], err = r.Int64(); err != nil {
			return nil, fmt.Errorf("tessdata: reading offset %d: %w", i, err)
		}
	}

	size := int64(len(data))
	c := &Container{entries: make(map[Type][]byte), swap: swap}
	for i := range offsets {
		if offsets[i] < 0 {
			continue
		}
		if offsets[i] > size {
			return nil, fmt.Errorf("tessdata: entry %d offset %d past end of %d-byte file", i, offsets[i], size)
		}
		end := size
		for j := i + 1; j < len(offsets); j++ {
			if offsets[j] >= 0 {
				if offsets[j] > size {
					return nil, fmt.Errorf("tessdata: entry %d offset %d past end of %d-byte file", j, offsets[j], size)
				}
				end = offsets[j]
				break
			}
		}
		if end < offsets[i] {
			return nil, fmt.Errorf("tessdata: entry %d has negative length", i)
		}
		if Type(i) < numTypes {
			c.entries[Type(i)] = data[offsets[i]:end]
		}
	}
	return c, nil
}

// Swapped reports whether the source file was foreign-endian. Readers created
// for its entries must be told.
func (c *Container) Swapped() bool { return c.swap }

func (c *Container) Entry(t Type) ([]byte, bool) {
	b, ok := c.entries[t]
	return b, ok
}

// Present lists the component types in the file, in type order.
func (c *Container) Present() []Type {
	var out []Type
	for t := Type(0); t < numTypes; t++ {
		if _, ok := c.entries[t]; ok {
			out = append(out, t)
		}
	}
	return out
}

// Version returns the model's version string, or "" if the file predates it.
func (c *Container) Version() string {
	b, ok := c.entries[TypeVersion]
	if !ok {
		return ""
	}
	return strings.TrimRight(string(b), "\x00\n")
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./internal/tessdata/ -run TestParseContainer -v`
Expected: PASS. `TestParseContainerRealModel` logs the component list; confirm it includes `lstm`, `lstm-unicharset`, and `lstm-recoder`.

- [ ] **Step 5: Verify against the Tesseract oracle**

`combine_tessdata -d` prints the offset table Tesseract itself reads.

```bash
combine_tessdata -d testdata/eng.traineddata
```

Compare its component list against the `t.Logf` output from Step 4. Every type
Tesseract lists must be present in `Present()`, with no extras. If they differ,
the parser is wrong — fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add internal/tessdata/container.go internal/tessdata/container_test.go
git commit -m "feat(tessdata): parse the .traineddata container entry table"
```

---

## Task 4: LSTM network graph deserialization

This is the task the spike exists to de-risk. From `Network::CreateFromFile`,
every layer serializes a common header, then type-specific data:

```
int8    type_tag           ; 0 (NT_NONE) means a type *name* string follows instead
int8    training_state
int8    needs_to_backprop
int32   network_flags
int32   ni                 ; input count
int32   no                 ; output count
int32   num_weights
string  name
<type-specific>
```

Type-specific bodies:

- **Plumbing** (`NT_SERIES`, `NT_PARALLEL`, `NT_REPLICATED`, `NT_PAR_RL_LSTM`,
  `NT_PAR_UD_LSTM`, `NT_PAR_2D_LSTM`, `NT_XREVERSED`, `NT_YREVERSED`,
  `NT_XYTRANSPOSE`): `uint32 stack_size` (reject > 10000), then that many
  recursive layers; then, if `network_flags & NF_LAYER_SPECIFIC_LR`, a
  learning-rate vector.
- **FullyConnected** (`NT_SOFTMAX`, `NT_SOFTMAX_NO_CTC`, `NT_RELU`, `NT_TANH`,
  `NT_LINEAR`, `NT_LOGISTIC`, `NT_POSCLIP`, `NT_SYMCLIP`): one weight matrix.
- **LSTM** (`NT_LSTM`, `NT_LSTM_SUMMARY`, `NT_LSTM_SOFTMAX`,
  `NT_LSTM_SOFTMAX_ENCODED`): `int32 na_`, then 5 gate weight matrices in order
  `CI, GI, GF1, GO, GFS` — **`GFS` is skipped unless the layer is 2-D** — then,
  for the softmax variants only, a nested layer.
- **Convolve / Maxpool / Reconfig / Input**: small fixed fields; read
  `src/lstm/convolve.cpp`, `maxpool.cpp`, `reconfig.cpp`, `input.cpp`.

Weight matrices (`WeightMatrix::DeSerialize`):

```
uint8 mode      ; bit 0 (1) = int8 weights, bit 2 (4) = Adam, bit 7 (128) = new format
if !(mode & 128)  → legacy layout, see WeightMatrix::DeSerializeOld
if mode & 1       → int8 2-D array, then uint32 count + that many float64 scales
else              → float64 2-D array
                  ; if training: an updates array, and if Adam, a dw_sq_sum array
```

The 2-D array layout is `GENERIC_2D_ARRAY<T>::DeSerialize` in
`src/ccstruct/matrix.h` — **read it before implementing**; it is the one
structure this plan does not specify, and guessing it will produce a parser that
silently desynchronises.

**Scope: structure and shapes only.** Record each matrix's dimensions and skip
its payload. No arithmetic. Weight *values* arrive in L1.

**Files:**
- Create: `internal/tessdata/network.go`
- Test: `internal/tessdata/network_test.go`

**Interfaces:**
- Consumes: `Reader` (Task 2), `Container` (Task 3).
- Produces:

```go
type LayerType int8

const (
	LayerNone       LayerType = 0
	LayerInput      LayerType = 1
	LayerConvolve   LayerType = 2
	LayerMaxpool    LayerType = 3
	LayerParallel   LayerType = 4
	LayerReplicated LayerType = 5
	LayerParRLLSTM  LayerType = 6
	LayerParUDLSTM  LayerType = 7
	LayerPar2DLSTM  LayerType = 8
	LayerSeries     LayerType = 9
	LayerReconfig   LayerType = 10
	LayerXReversed  LayerType = 11
	LayerYReversed  LayerType = 12
	LayerXYTranspose LayerType = 13
	LayerLSTM       LayerType = 14
	LayerLSTMSummary LayerType = 15
	LayerLogistic   LayerType = 16
	LayerPosClip    LayerType = 17
	LayerSymClip    LayerType = 18
	LayerTanh       LayerType = 19
	LayerRelu       LayerType = 20
	LayerLinear     LayerType = 21
	LayerSoftmax    LayerType = 22
	LayerSoftmaxNoCTC LayerType = 23
	LayerLSTMSoftmax  LayerType = 24
	LayerLSTMSoftmaxEncoded LayerType = 25
	LayerTensorFlow LayerType = 26
)

func (t LayerType) String() string

type MatrixShape struct {
	Rows, Cols int
	Int8       bool
}

type Layer struct {
	Type       LayerType
	Name       string
	NumInputs  int
	NumOutputs int
	NumWeights int
	Flags      int32
	Matrices   []MatrixShape
	Children   []*Layer
}

func ParseNetwork(data []byte, swap bool) (*Layer, error)
func (l *Layer) Tree(w io.Writer)
```

- [ ] **Step 1: Read the two undocumented pieces**

Before writing code, read and take notes on:

```bash
sed -n '/DeSerialize/,/^  }/p' \
  /private/tmp/claude-501/-Users-christopherdobbyn-work-dobbo-ca/56a38a28-5026-4f14-bc12-d25e504d7a30/scratchpad/tess/src/ccstruct/matrix.h
```

and `src/lstm/convolve.cpp`, `maxpool.cpp`, `reconfig.cpp`, `input.cpp` in the
same tree. If that clone is gone: `git clone --depth 1 https://github.com/tesseract-ocr/tesseract`.

Record the exact `GENERIC_2D_ARRAY` field order in a comment at the top of
`network.go`. Everything downstream depends on it.

- [ ] **Step 2: Write the failing test**

Create `internal/tessdata/network_test.go`:

```go
package tessdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The whole point of the spike: parse the real model's graph.
func TestParseNetworkRealModel(t *testing.T) {
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

	if root.Type != LayerSeries {
		t.Errorf("root.Type = %v; want %v", root.Type, LayerSeries)
	}
	if len(root.Children) == 0 {
		t.Fatal("root has no children; the graph did not deserialize")
	}

	var b strings.Builder
	root.Tree(&b)
	t.Logf("network graph:\n%s", b.String())

	// The tessdata_best English model is a CRNN: it must contain at least one
	// convolution and at least one LSTM somewhere in the tree.
	var conv, lstmCount int
	var walk func(*Layer)
	walk = func(l *Layer) {
		switch l.Type {
		case LayerConvolve:
			conv++
		case LayerLSTM, LayerLSTMSummary, LayerLSTMSoftmax, LayerLSTMSoftmaxEncoded:
			lstmCount++
		}
		for _, ch := range l.Children {
			walk(ch)
		}
	}
	walk(root)
	if conv == 0 {
		t.Error("no convolution layers found; expected a CRNN")
	}
	if lstmCount == 0 {
		t.Error("no LSTM layers found; expected a CRNN")
	}
}

func TestParseNetworkRejectsAbsurdStackSize(t *testing.T) {
	// A Series header claiming 99999 children must be rejected, matching
	// Plumbing::DeSerialize's 10000 guard.
	var b []byte
	b = append(b, byte(LayerSeries), 0, 0)                  // type, training, backprop
	b = append(b, 0, 0, 0, 0)                               // flags
	b = append(b, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)       // ni, no, num_weights
	b = append(b, 0, 0, 0, 0)                               // empty name
	b = append(b, 0x9f, 0x86, 0x01, 0x00)                   // stack size 99999
	if _, err := ParseNetwork(b, false); err == nil {
		t.Fatal("ParseNetwork() with 99999-child stack: want error, got nil")
	}
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `go test ./internal/tessdata/ -run TestParseNetwork -v`
Expected: FAIL — `undefined: ParseNetwork`.

- [ ] **Step 4: Implement the graph parser**

Create `internal/tessdata/network.go`. Structure it as: a `LayerType.String()`
table matching `kTypeNames` in `src/lstm/network.cpp`; a `parseLayer(r *Reader)`
reading the common header; a switch dispatching to `parsePlumbing`,
`parseFullyConnected`, `parseLSTM`, and the small fixed-field types; and
`parseMatrixShape(r *Reader)` implementing `WeightMatrix::DeSerialize` for shape
only.

Required behaviours:

- A `type_tag` of 0 means a type-name string follows; resolve it against the
  name table rather than treating 0 as `LayerNone`.
- Reject a Plumbing stack size above 10000.
- For LSTM layers, read `int32 na_` first, then iterate gates `CI, GI, GF1, GO,
  GFS`, skipping `GFS` unless the layer is 2-D. Determine 2-D exactly as
  Tesseract does: after reading `CI`, `ns_ = CI.NumOutputs()` and
  `is_2d = (na - nf) == ni + 2*ns`, where `nf` is `no` for `NT_LSTM_SOFTMAX`,
  `ceil(log2(no))` for `NT_LSTM_SOFTMAX_ENCODED`, and 0 otherwise.
- After parsing the root layer, `r.Remaining()` must be 0. If it is not, the
  parse desynchronised — return an error saying so, with the byte count. **This
  check is the single most valuable line in the file**; a silently-misaligned
  parser that returns a plausible tree is the failure mode to guard against.

`Tree` writes an indented layer tree, one line per layer:
`<indent><type> "<name>" ni=<n> no=<n> weights=<n> [matrices: RxC, RxC]`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `go test ./internal/tessdata/ -run TestParseNetwork -v`
Expected: PASS. Read the logged tree: it should show a `Series` root containing a
convolution, a maxpool, several LSTM layers, and a softmax output.

- [ ] **Step 6: Verify the tree against the model's own spec string**

Tesseract names the root layer with the network spec string used to build it
(e.g. `[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`). The parsed tree must
be consistent with that string: one `Ct` convolution, one `Mp` maxpool, the LSTM
layers in the stated order and sizes, and a final softmax whose output count
equals the unicharset-derived class count.

Check `root.Name` from the Step 5 log. If it carries a spec string, walk it
against the tree by hand and confirm every element matches. **If the root name is
empty or does not resemble a spec string, say so in the task report and verify
instead by confirming `r.Remaining() == 0` plus the layer counts** — a
fully-consumed buffer with a structurally sensible tree is strong evidence, and
the definitive check arrives in L1 when the forward pass output is compared to
`tesseract --psm 7`.

- [ ] **Step 7: Commit**

```bash
git add internal/tessdata/network.go internal/tessdata/network_test.go
git commit -m "feat(tessdata): deserialize the LSTM network graph structure"
```

---

## Task 5: `cadmusdump` CLI — the spike deliverable

**Files:**
- Create: `cmd/cadmusdump/main.go`
- Test: `cmd/cadmusdump/main_test.go`

**Interfaces:**
- Consumes: `ParseContainer`, `ParseNetwork`, `Layer.Tree`.
- Produces: a binary printing a model's component inventory and layer tree.

- [ ] **Step 1: Write the failing test**

Create `cmd/cadmusdump/main_test.go`:

```go
package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDumpRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "eng.traineddata")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture not present (run ./testdata/fetch.sh): %v", err)
	}
	var b strings.Builder
	if err := dump(path, &b); err != nil {
		t.Fatalf("dump() error = %v", err)
	}
	out := b.String()
	for _, want := range []string{"lstm", "lstm-unicharset", "Series"} {
		if !strings.Contains(out, want) {
			t.Errorf("dump output missing %q\n%s", want, out)
		}
	}
}

func TestDumpMissingFileErrors(t *testing.T) {
	var b strings.Builder
	if err := dump("does-not-exist.traineddata", &b); err == nil {
		t.Fatal("dump() on missing file: want error, got nil")
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./cmd/cadmusdump/ -v`
Expected: FAIL — `undefined: dump`.

- [ ] **Step 3: Implement the CLI**

Create `cmd/cadmusdump/main.go`:

```go
// Command cadmusdump prints the contents of a Tesseract .traineddata file:
// its component inventory and, for the LSTM component, its layer tree.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

func dump(path string, w io.Writer) error {
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

	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		fmt.Fprintln(w, "\nno lstm component (legacy-only model)")
		return nil
	}
	root, err := tessdata.ParseNetwork(lstm, c.Swapped())
	if err != nil {
		return fmt.Errorf("parsing network graph: %w", err)
	}
	fmt.Fprintln(w, "\nnetwork:")
	root.Tree(w)
	return nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: cadmusdump <model.traineddata>")
		os.Exit(2)
	}
	if err := dump(os.Args[1], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cadmusdump:", err)
		os.Exit(1)
	}
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./cmd/cadmusdump/ -v`
Expected: PASS.

- [ ] **Step 5: Run it for real and record the outcome**

```bash
go run ./cmd/cadmusdump testdata/eng.traineddata
```

Paste the full output into the `cad-spike` beads issue:

```bash
bd update cad-spike --append-notes "$(go run ./cmd/cadmusdump testdata/eng.traineddata)"
```

Then record the verdict the spike exists to produce:

```bash
# if the graph parsed cleanly with the buffer fully consumed:
bd update cad-spike --append-notes "VERDICT: .traineddata deserialization is tractable. L1 proceeds as specced."
bd close cad-spike
# if it did not:
bd update cad-spike --append-notes "VERDICT: blocked — <exact failure>. Per the spec's risk table, L5's ONNX loader is promoted ahead of L2 and cad-l1 is rescoped to the tensor runtime only."
```

- [ ] **Step 6: Commit**

```bash
git add cmd/cadmusdump
git commit -m "feat(cmd): add cadmusdump for inspecting .traineddata models"
```

---

## Task 6: Bitmap type, the golden harness, and Otsu binarization

Establishes the pattern every later L0 task follows: goldens generated once from
Leptonica by a C program, committed, and compared in pure-Go tests that need no
Leptonica at run time.

**Files:**
- Create: `internal/imaging/bitmap.go`, `internal/imaging/binarize.go`
- Create: `testdata/golden/gen/Makefile`, `testdata/golden/gen/gen.c`
- Test: `internal/imaging/binarize_test.go`

**Interfaces:**
- Consumes: nothing.
- Produces:

```go
// Bitmap is an 8-bit grayscale or 1-bit bilevel image. Depth is 1 or 8.
type Bitmap struct {
	Width, Height int
	Depth         int
	Stride        int    // bytes per row
	Pix           []byte
}

func NewBitmap(w, h, depth int) *Bitmap
func FromImage(img image.Image) *Bitmap      // any image.Image → 8bpp gray
func (b *Bitmap) At(x, y int) uint8          // 0/1 for depth 1, 0-255 for depth 8
func (b *Bitmap) Set(x, y int, v uint8)
func (b *Bitmap) Bounds() image.Rectangle
func (b *Bitmap) Clone() *Bitmap

func OtsuThreshold(gray *Bitmap) uint8
func Otsu(gray *Bitmap) *Bitmap              // → depth-1 bitmap, 1 = foreground (ink)
```

**Convention, fixed here and relied on by every later task:** in a depth-1
bitmap, **1 means foreground/ink**. This matches Leptonica's convention for
binary images and inverts the usual "black is 0" grayscale intuition. Getting
this backwards makes every morphology result silently wrong.

- [ ] **Step 1: Write the golden generator**

Create `testdata/golden/gen/gen.c`:

```c
// Regenerates Leptonica goldens. Manual step; see ../../../Makefile `goldens`.
// Each golden is a raw dump: width, height, depth as int32 LE, then packed rows.
#include <stdio.h>
#include <stdlib.h>
#include <leptonica/allheaders.h>

static void dump(const char *path, PIX *p) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int32_t hdr[3] = { pixGetWidth(p), pixGetHeight(p), pixGetDepth(p) };
    fwrite(hdr, sizeof(int32_t), 3, f);
    l_uint32 *data = pixGetData(p);
    int wpl = pixGetWpl(p);
    fwrite(data, sizeof(l_uint32), (size_t)wpl * pixGetHeight(p), f);
    fclose(f);
    printf("wrote %s (%dx%d d=%d)\n", path, hdr[0], hdr[1], hdr[2]);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: gen <input.png> <outdir>\n"); return 2; }
    PIX *src = pixRead(argv[1]);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    PIX *gray = pixConvertTo8(src, 0);

    char path[512];
    snprintf(path, sizeof path, "%s/gray.bin", argv[2]);
    dump(path, gray);

    // Otsu, whole-image (no tiling), no smoothing.
    PIX *otsu = NULL;
    pixOtsuAdaptiveThreshold(gray, pixGetWidth(gray), pixGetHeight(gray), 0, 0, 0.0, NULL, &otsu);
    snprintf(path, sizeof path, "%s/otsu.bin", argv[2]);
    dump(path, otsu);

    pixDestroy(&src); pixDestroy(&gray); pixDestroy(&otsu);
    return 0;
}
```

Create `testdata/golden/gen/Makefile`:

```make
LEPT_PREFIX := $(shell brew --prefix leptonica)
CFLAGS := -I$(LEPT_PREFIX)/include -O2
LDFLAGS := -L$(LEPT_PREFIX)/lib -lleptonica

all: gen
	./gen ../input/scan.png ..

gen: gen.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f gen
```

- [ ] **Step 2: Generate the input image and the goldens**

The golden input is **synthetic and generated by a committed Go program**, not a
real scan. For these tests the only thing that matters is that Go and Leptonica
agree on the *same* pixels; realism matters for L2's character error rate, not
for L0's operator equality. A generated input also means no binary asset of
uncertain provenance enters the repo, and anyone can reproduce it.

Create `testdata/golden/input/gen.go` as a `//go:build ignore` program that
writes `scan.png`: a 1000x1400 8-bit grayscale image containing

- a light background with a gentle horizontal gradient (220 → 240), so
  thresholding is not trivially uniform;
- ~40 rows of filled dark rectangles of varying width and height standing in for
  text lines, at two distinct x-offsets so there are two "columns";
- a handful of isolated 1-2px dark specks, so despeckling and connected-component
  edge cases have something to bite on;
- a 3px-wide dark vertical rule, standing in for a table border or scan artifact;
- deterministic pseudo-random noise from a **fixed seed** (`rand.New(rand.NewPCG(1, 2))`),
  so regeneration is byte-identical.

```bash
go run testdata/golden/input/gen.go
make goldens
ls -l testdata/golden/*.bin
```

Commit `testdata/golden/input/gen.go`, the generated `scan.png`, and the
`.bin` goldens. Committing the generated PNG as well as its generator is
deliberate: it keeps `make goldens` reproducible without requiring the generator
to be re-run, and makes any accidental change to the input visible in a diff.

```bash
make goldens
ls -l testdata/golden/*.bin
```

Commit `testdata/golden/input/scan.png`, `testdata/golden/gray.bin`, and
`testdata/golden/otsu.bin`.

- [ ] **Step 3: Write the failing test**

Create `internal/imaging/binarize_test.go`:

```go
package imaging

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// loadGolden reads a dump produced by testdata/golden/gen/gen.c: three int32
// header fields (width, height, depth) then packed 32-bit-word rows, exactly as
// Leptonica stores them.
func loadGolden(t *testing.T, name string) *Bitmap {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", name))
	if err != nil {
		t.Skipf("golden %s not present (run make goldens): %v", name, err)
	}
	if len(raw) < 12 {
		t.Fatalf("golden %s is truncated: %d bytes", name, len(raw))
	}
	w := int(int32(binary.LittleEndian.Uint32(raw[0:4])))
	h := int(int32(binary.LittleEndian.Uint32(raw[4:8])))
	d := int(int32(binary.LittleEndian.Uint32(raw[8:12])))

	b := NewBitmap(w, h, d)
	wpl := (w*d + 31) / 32
	body := raw[12:]
	for y := range h {
		for x := range w {
			word := binary.BigEndian.Uint32(body[(y*wpl+(x*d)/32)*4:])
			switch d {
			case 1:
				b.Set(x, y, uint8((word>>(31-uint(x*d)%32))&1))
			case 8:
				b.Set(x, y, uint8((word>>(24-uint(x*d)%32))&0xff))
			default:
				t.Fatalf("unsupported golden depth %d", d)
			}
		}
	}
	return b
}

func TestOtsuMatchesLeptonica(t *testing.T) {
	gray := loadGolden(t, "gray.bin")
	want := loadGolden(t, "otsu.bin")

	got := Otsu(gray)

	if got.Width != want.Width || got.Height != want.Height {
		t.Fatalf("Otsu() size = %dx%d; want %dx%d", got.Width, got.Height, want.Width, want.Height)
	}
	// Leptonica's binary convention is 1 = foreground; ours must match exactly.
	var diff int
	for y := range got.Height {
		for x := range got.Width {
			if got.At(x, y) != want.At(x, y) {
				diff++
			}
		}
	}
	if diff != 0 {
		total := got.Width * got.Height
		t.Errorf("Otsu() differs from Leptonica in %d of %d pixels (%.4f%%)",
			diff, total, 100*float64(diff)/float64(total))
	}
}

func TestOtsuThresholdBimodal(t *testing.T) {
	// Half the pixels at 20, half at 200: the threshold must land between.
	b := NewBitmap(10, 10, 8)
	for y := range 10 {
		for x := range 10 {
			if y < 5 {
				b.Set(x, y, 20)
			} else {
				b.Set(x, y, 200)
			}
		}
	}
	got := OtsuThreshold(b)
	if got < 20 || got > 200 {
		t.Errorf("OtsuThreshold() = %d; want a value in [20,200]", got)
	}
}

func TestOtsuForegroundIsOne(t *testing.T) {
	// A dark blob on a light field must binarize to 1s on the blob.
	b := NewBitmap(10, 10, 8)
	for y := range 10 {
		for x := range 10 {
			b.Set(x, y, 240)
		}
	}
	for y := 2; y < 5; y++ {
		for x := 2; x < 5; x++ {
			b.Set(x, y, 10)
		}
	}
	got := Otsu(b)
	if got.At(3, 3) != 1 {
		t.Errorf("Otsu() at dark pixel = %d; want 1 (foreground)", got.At(3, 3))
	}
	if got.At(8, 8) != 0 {
		t.Errorf("Otsu() at light pixel = %d; want 0 (background)", got.At(8, 8))
	}
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `go test ./internal/imaging/ -v`
Expected: FAIL — `undefined: NewBitmap`.

- [ ] **Step 5: Implement Bitmap and Otsu**

Create `internal/imaging/bitmap.go` with the `Bitmap` type per the Interfaces
block. Store depth-1 rows packed MSB-first so the golden loader's bit order and
the internal order agree. `FromImage` converts via
`color.GrayModel.Convert`.

Create `internal/imaging/binarize.go`:

```go
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
```

- [ ] **Step 6: Run the tests**

Run: `go test ./internal/imaging/ -v`
Expected: the two synthetic tests PASS.

`TestOtsuMatchesLeptonica` may show a small pixel difference. Leptonica's
`pixOtsuAdaptiveThreshold` computes the same criterion but may differ by one
level at ties. **Acceptance: differences below 0.01% of pixels are acceptable —
relax the assertion to that bound and note the measured figure in the commit
message. Above 0.01%, the implementation or the foreground convention is wrong;
fix rather than widen the bound.**

- [ ] **Step 7: Commit**

```bash
git add internal/imaging testdata/golden
git commit -m "feat(imaging): add Bitmap, Otsu binarization, and the Leptonica golden harness"
```

---

## Tasks 7-12: remaining L0 primitives

Each follows Task 6's pattern exactly: extend `testdata/golden/gen/gen.c` with
the Leptonica call, `make goldens`, commit the golden, write the failing
comparison test, implement, verify, commit.

| Task | File | Functions | Leptonica oracle |
|---|---|---|---|
| **7** | `binarize.go` | `Sauvola(gray *Bitmap, window int, k float64) *Bitmap`, via integral images for O(1) per-window mean and variance | `pixSauvolaBinarize` |
| **8** | `rasterop.go`, `morph.go` | `RasterOp(dst *Bitmap, dr image.Rectangle, op Op, src *Bitmap, sp image.Point)`; `Dilate/Erode/Open/Close(b *Bitmap, w, h int) *Bitmap` | `pixRasterop`, `pixDilateBrick`, `pixErodeBrick`, `pixOpenBrick`, `pixCloseBrick` |
| **9** | `conncomp.go` | `ConnComp(b *Bitmap, connectivity int) []Component` where `Component{Bounds image.Rectangle; Pixels int}`; connectivity 4 or 8 | `pixConnComp` |
| **10** | `seedfill.go` | `SeedFill(seed, mask *Bitmap, connectivity int) *Bitmap`; `DistanceFunction(b *Bitmap) *Bitmap` | `pixSeedfillBinary`, `pixDistanceFunction` |
| **11** | `deskew.go` | `SkewAngle(b *Bitmap) float64` (radians, by projection-profile variance over candidate angles); `Rotate(b *Bitmap, radians float64) *Bitmap` | `pixFindSkew`, `pixRotate` |
| **12** | `grid.go` | `Grid` indexing `image.Rectangle`s into cells: `NewGrid(bounds image.Rectangle, cellSize int)`, `Insert(id int, r image.Rectangle)`, `Query(r image.Rectangle) []int` | none — original work, unit-tested directly |

**Per-task requirements:**

- **Every task ends with `go test ./... ` green and a commit.**
- **Task 8's `Op`** is an enum over Leptonica's raster ops; implement at minimum
  `OpSet`, `OpClear`, `OpSrc`, `OpNotSrc`, `OpSrcOrDst`, `OpSrcAndDst`,
  `OpSrcXorDst`. Read `pixRasterop`'s semantics for partially-overlapping and
  out-of-bounds rectangles and match them; clipping behaviour is where hand-rolled
  raster ops usually diverge.
- **Task 9** must handle the pathological cases explicitly: an all-background
  bitmap returns an empty slice, a single-pixel component is still a component,
  and a component touching the image border is not clipped away. Write a test for
  each before implementing.
- **Task 11's `SkewAngle`** searches candidate angles by maximising the variance
  of the horizontal projection profile. Sweep ±5° at 0.1° steps, then refine ±0.1°
  at 0.01° steps around the best. Test with a synthetically rotated version of the
  golden input: rotate a known-straight bitmap by a known angle, recover it, and
  assert the recovered angle is within 0.05°.
- **Task 12 is original work, not a port** — no Tesseract attribution header.
  Test it directly: insert overlapping and disjoint rectangles, query, and assert
  the returned id sets. Include a query that matches nothing and a rectangle
  spanning many cells.

- [ ] **Task 7: Sauvola binarization** — implement, verify against golden, commit
- [ ] **Task 8: raster operations and brick morphology** — implement, verify against goldens, commit
- [ ] **Task 9: connected components** — implement, verify against golden, commit
- [ ] **Task 10: seedfill and distance transform** — implement, verify against goldens, commit
- [ ] **Task 11: deskew** — implement, verify against golden, commit
- [ ] **Task 12: bbgrid spatial index** — implement, unit test, commit

- [ ] **Final step: close the L0 issue**

```bash
make test && make lint
bd update cad-l0 --append-notes "L0 complete: Bitmap, Otsu, Sauvola, rasterop, brick morphology, connected components, seedfill, distance transform, deskew, bbgrid. All verified against committed Leptonica goldens."
bd close cad-l0
```

---

## Self-Review

**Spec coverage.** L0's spec scope is "Otsu and Sauvola binarization, deskew, the
33-function morphology kernel, connected components, bbgrid spatial index." Tasks
6-12 cover binarization (6, 7), morphology and rasterop (8), connected components
(9), seedfill and distance (10), deskew (11), and the grid (12). The spec lists 33
Leptonica functions; Tasks 8-11 implement the ~12 that carry real algorithmic
content, while the remainder (`pixCreate`, `pixDestroy`, `pixGetWidth`,
`pixGetData`, `pixSetAll`, `pixClipRectangle`, and similar accessors) are
subsumed by `Bitmap` in Task 6. **The following are deliberately deferred to L4,
where the textord code that calls them lands:** `pixGenerateHalftoneMask`,
`pixReduceRankBinaryCascade`, `pixBlockconv`, `pixNearlyRectangular`,
`pixExpandReplicate`, `pixRenderBoxArb`. Add them to `cad-l4` when starting it.

Spike scope — container parse, unicharset/recoder/dawg entry access, network
graph deserialization — is covered by Tasks 3-5. Entry *contents* beyond the
network graph (unicharset parsing, dawg decoding) are L1 work, not spike work;
the spike only proves the entries are reachable and the graph is parseable.

**Placeholder scan.** One item is a deliberate pointer rather than a
specification: `GENERIC_2D_ARRAY<T>::DeSerialize` in Task 4 Step 1. That is the
one structure this plan does not spell out, it is flagged as such with the exact
file to read, and discovering it is precisely what the spike is for. Task 6 Step 6
carries a numeric acceptance bound (0.01%) rather than "close enough". Task 4
Step 6 gives an explicit fallback if the spec-string oracle turns out not to
exist.

**Type consistency.** `Bitmap.At`/`Set` are used identically in the golden loader
and the implementations. `Layer.Children` is used by both `Tree` and the test
walker. `Container.Swapped()` is produced in Task 3 and consumed in Tasks 4 and 5.
`Reader.SetSwap` is produced in Task 2 and consumed in Task 3. `MatrixShape` is
defined in Task 4 and used only there. The depth-1 foreground convention (1 = ink)
is stated in Task 6 and relied on by Tasks 8-11.
