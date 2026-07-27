# Cadmus — a pure-Go OCR engine

**Status:** approved design
**Date:** 2026-07-27
**Repo:** `dobbo-ca/cadmus`
**Primary consumer:** `dobbo-ca/kleio`

Cadmus brought the alphabet to Greece. The library turns pixels back into letters.

---

## 1. Problem

Kleio's OCR stage shells out to a `kleio-toolchain` container image bundling
`ocrmypdf`, `tesseract`, `ghostscript`, `jbig2enc`, `pngquant`, and `poppler`.
That creates three problems:

1. **Binary dependencies.** Kleio cannot ship as a self-contained Go binary. The
   toolchain image is a build, supply-chain, and cross-compilation burden.
2. **No accuracy control.** When OCR is wrong, there is no lever. Tesseract is an
   opaque subprocess; tuning stops at flag-twiddling.
3. **No feedback loop.** Kleio's UI should let a user report an OCR error, and that
   correction should measurably improve the model. A subprocess cannot consume
   corrections.

### Goals

| # | Goal | Implication |
|---|------|-------------|
| G1 | Remove OCR binary dependencies from Kleio | Pure Go: **no cgo, and no `.so`**. Rules out `purego`-loaded `libonnxruntime`, which still ships a shared library. |
| G2 | Ability to improve accuracy directly | Own the pipeline end to end. Every heuristic and weight is ours to tune. |
| G3 | UI-reported errors improve the model | **In-process training**, a line-image/ground-truth corpus, model versioning, and re-OCR of affected documents. |

G3 is the binding constraint. It eliminates any design that wraps Tesseract as a
black box (cgo bindings, or Tesseract-compiled-to-WASM under wazero), because a
wrapped engine can be neither tuned nor trained.

### Non-goals

- **PDF handling.** Cadmus accepts `image.Image` and returns text. Rasterization,
  compression, and text-layer stamping stay in Kleio. This is what keeps the repo
  to one concern.
- **Training from scratch.** Cadmus fine-tunes pretrained weights. Bootstrapping a
  model from random init is out of scope.
- **Tesseract's legacy (pre-4.0) engine.** LSTM path only.
- **GPU.** CPU-only. Kleio's OCR is queue-driven batch work behind KEDA; latency
  is not the constraint, and GPU means cgo.

---

## 2. Why not the alternatives

Measured against `tesseract-ocr/tesseract@main`, `src/` only, excluding tests:

| Option | Verdict |
|---|---|
| Shell out to `tesseract` (status quo) | Fails G1, G2, G3. |
| cgo bindings (`gosseract`) | Fails G1 (cgo + libtesseract + libleptonica), G2, G3. |
| Tesseract→WASM→wazero (`gogosseract`) | Satisfies G1 only. Fails G2 and G3 — a WASM blob is not tunable or trainable. Also unmaintained (broken by wazero 1.8.0) and ~6× slower than cgo. |
| ONNX via `purego` | Fails G1 — dynamically loads `libonnxruntime.so`. No cgo is not the same as no binary dependency. |
| Pure-Go ONNX (`gonnx`, `onnx-go`) | Satisfies G1. Inference only — no training path for G3. Viable as an *input* to our own runtime (see L5), not as the runtime. |
| **Port the LSTM path to Go** | Satisfies G1, G2, G3. Chosen. |

### Port surface, measured

```
src/textord/     42,775   layout analysis  — port subset (L4)
src/ccutil/      32,249   utils/params     — mostly replaced by Go stdlib
src/ccstruct/    25,153   page structs     — port subset
src/ccmain/      20,140   control flow     — port subset
src/lstm/        18,290   recognizer       — port (L1)
src/training/    18,620   trainer          — reimplement, not port (L3)
src/classify/    16,981   legacy engine    — SKIP
src/dict/         5,058   dawg lexicon     — port (L1)
src/wordrec/      6,604   legacy engine    — SKIP
src/api/          6,520   C/C++ API        — SKIP (we define our own)
src/arch/         2,252   SIMD             — SKIP (Go compiler / later)
src/viewer/       1,987   debug viewer     — SKIP
src/cutil/          460   legacy C utils   — SKIP
                -------
                198,204   total
```

Within `textord`, the LSTM path does not need the fixed-pitch and legacy-baseline
machinery: `topitch.cpp` (1,722), `pithsync.cpp` (685), `cjkpitch.cpp` (1,144),
`fpchop.cpp` (812), `oldbasel.cpp` (1,641) — 6,004 LOC dropped. Also droppable:
`drawtord.cpp` (413, debug viewer), `devanagari_processing.cpp` (493,
script-specific), and optionally `tablefind.cpp` + `tablerecog.cpp` (3,216, table
detection). **Working `textord` surface: ~24k LOC, not 43k.**

### Coupling, measured

**Leptonica.** `textord` calls **33 distinct Leptonica functions**, all standard
raster morphology:

```
pixConnComp  pixSeedfillBinary  pixDistanceFunction  pixRasterop
pixDilateBrick  pixErodeBrick  pixOpenBrick  pixCloseBrick
pixBlockconv  pixReduceRankBinaryCascade  pixExpandReplicate
pixCountPixels  pixCountPixelsByRow  pixCountPixelsInRow
pixClipRectangle  pixClipBoxToForeground  pixNearlyRectangular
pixGenerateHalftoneMask  pixGenHalftoneMask  pixSubtract
pixCreate  pixDestroy  pixConvertTo  pixSetAll  pixSetInRect  pixClearInRect
pixGetData  pixGetDepth  pixGetWidth  pixGetHeight  pixGetWpl
pixRenderBoxArb  pixWrite
```

This is a bounded morphology kernel, not a dependency on Leptonica's ~250k LOC.
Estimated ~2k LOC of Go (L0).

**ccstruct.** `textord` includes 24 `ccstruct` headers. The substantive ones are
`blobbox.h` (855, `BLOBNBOX`/`TO_ROW`/`TO_BLOCK`), `coutln.h`/`stepblob.h`/
`crakedge.h` (outline→blob construction), `ocrblock.h`/`ocrrow.h`/`werd.h`/
`pdblock.h` (output structures), and `detlinefit.h`/`linlsq.h`/`quadlsq.h`/
`quspline.h` (line and spline fitting). The remainder (`points.h`, `rect.h`,
`quadratc.h`, `statistc.h`) collapse into Go geometry types and stdlib.

**Intrusive lists.** Only 20 `ELISTIZE`/`CLISTIZE` macro instantiations across
`textord`. Go slices replace them; the macro machinery does not need porting.

**Tunable parameters.** 102 `*_VAR` declarations in `textord`. These are copied
constants, not logic.

### Tesseract's LSTM engine is a CRNN

From `src/lstm/network.h`:

```
NT_CONVOLVE  NT_MAXPOOL  NT_LSTM  NT_LSTM_SUMMARY
NT_PAR_RL_LSTM (bidirectional)  NT_PAR_UD_LSTM  NT_PAR_2D_LSTM
NT_SOFTMAX (with CTC)  NT_LINEAR  NT_TANH  NT_RELU  NT_LOGISTIC
plumbing: NT_SERIES  NT_PARALLEL  NT_XREVERSED  NT_YREVERSED
          NT_XYTRANSPOSE  NT_RECONFIG  NT_REPLICATED
```

conv → maxpool → bidirectional LSTM stack → softmax → CTC. That is a CRNN;
Tesseract simply serializes its graph inside `.traineddata` rather than ONNX.

This is the key architectural insight: **one Go tensor runtime serves both
Tesseract's weights and modern CRNN weights.** The op-set delta is small.

| op | Tesseract LSTM | docTR `crnn_vgg16_bn` | PP-OCR rec |
|---|---|---|---|
| conv2d, maxpool, linear, softmax, CTC | yes | yes | yes |
| LSTM cell, bidirectional | yes | yes | yes (v2-era) |
| batchnorm | no | yes | yes |
| depthwise-separable conv, hardswish | no | no | yes |
| 2D / summarizing LSTM | yes | no | no |
| transformer (attention, layernorm, GELU) | no | no | SVTR variants only |

Supporting modern CRNN weights later therefore costs batchnorm + depthwise conv +
an ONNX loader — **provided a BiLSTM-era checkpoint is chosen** (docTR
`crnn_vgg16_bn` or `crnn_mobilenet_v3`). SVTR-based checkpoints pull in a
transformer stack and are explicitly out of scope for L5. Architecture must be
verified per checkpoint before adoption.

---

## 3. Architecture

```
                    ┌─ loader: .traineddata  (L1)
                    │
                    ├─ loader: ONNX/docTR    (L5)
                    │                        │
                    │                        ▼
                    │              ┌──────────────────────┐
                    └─────────────►│  tensor runtime      │  conv2d, depthwise,
                                   │  forward + backward  │  maxpool, batchnorm,
                                   └──────────┬───────────┘  LSTM cell, linear,
                                              │              softmax, CTC
                        ┌─────────────────────┴────────────────┐
                        ▼                                      ▼
                CTC beam decode                         trainer / fine-tune
                (+ dawg lexicon)                    ◄── Kleio corrections (L3)


   image.Image ──► Detector ──► []LineImage ──► Recognizer ──► Page
                      │                              │
        ┌─────────────┴─────────────┐    ┌───────────┴────────────┐
        │ L2: classical CV          │    │ L1: Tesseract LSTM     │
        │ L4: textord port          │    │ L5: CRNN / ONNX        │
        └───────────────────────────┘    └────────────────────────┘
```

Two interfaces carry the entire extension story:

```go
type Detector interface {
    // Detect segments a page into text lines, in reading order.
    Detect(ctx context.Context, img image.Image) ([]LineRegion, error)
}

type Recognizer interface {
    // Recognize transcribes a single cropped, deskewed line image.
    Recognize(ctx context.Context, line image.Image) (Line, error)
}
```

L4 and L5 are implementations swapped behind these interfaces — not rewrites.
Defining both at L1/L2 is the discipline that makes the layering real.

The L2 classical detector is **not** deleted when L4 lands. It stays as the fast
path for clean single-column scans, which is the bulk of Kleio's corpus.

---

## 4. Public API

```go
package cadmus

type Engine struct{ /* ... */ }

func New(opts ...Option) (*Engine, error)

func WithModel(r io.Reader) Option          // .traineddata; caller owns the file
func WithDetector(d Detector) Option
func WithRecognizer(r Recognizer) Option
func WithLexicon(enabled bool) Option       // dawg-weighted beam search

func (e *Engine) Recognize(ctx context.Context, img image.Image) (*Page, error)

type Page struct {
    Blocks         []Block
    MeanConfidence float64   // Kleio's validation gate reads this
}

type Block struct {
    Lines  []Line
    Bounds image.Rectangle
}

type Line struct {
    Text       string
    Words      []Word
    Bounds     image.Rectangle
    Confidence float64
}

type Word struct {
    Text       string
    Bounds     image.Rectangle
    Confidence float64
}

// --- L3: the feedback loop ---

type Correction struct {
    LineImage image.Image
    Truth     string
}

type TrainOptions struct {
    Iterations   int
    LearningRate float64
    HoldoutRatio float64   // split for eval; reported as CER/WER
}

type Model struct {
    Version    string
    BaseModel  string
    TrainedAt  time.Time
    CER, WER   float64
}

func (e *Engine) FineTune(ctx context.Context, corpus []Correction, opts TrainOptions) (*Model, error)
func (m *Model) WriteTo(w io.Writer) (int64, error)
```

`Page`/`Line`/`Word` with per-word bounding boxes is exactly what Kleio needs for
both the PDF text layer and the UI correction widget. A user clicking a wrong word
in the UI yields a `(LineImage, Truth)` pair with no extra plumbing — G3 falls out
of the data model rather than being bolted onto it.

**Dependencies: zero third-party.** Tensor ops are hand-rolled rather than built on
`gorgonia` or `spago`. We need roughly ten ops, and we must reproduce Tesseract's
arithmetic closely enough to match its output; a general graph framework would
fight `.traineddata` deserialization rather than help it.

---

## 5. Layers

Each layer is independently shippable and independently useful. **Each layer gets
its own implementation plan** — the project as a whole is far too large for one.
LOC figures in this section are estimated *Go* lines; figures in §2 are measured
*C++* lines from Tesseract.

### L0 — image core (~3k LOC)

Binarization (Otsu, Sauvola), deskew (projection-profile / Hough), the 33-function
morphology kernel, connected components, and the `bbgrid` spatial index.

**Verify:** differential against Leptonica on a fixed image corpus — identical
output bitmaps for each morphology op, and connected-component sets matching
`pixConnComp` exactly.

### L1 — recognizer runtime + Tesseract loader (~8k LOC)

L1 has two separable halves: the **tensor runtime** (loader-agnostic, and the
foundation L3 and L5 both build on) and the **`.traineddata` loader**. Only the
loader is Tesseract-specific. If the loader proves intractable, the runtime is
unaffected and L5's ONNX loader substitutes for it.

`.traineddata` container parsing (`TessdataManager` format), then the entries the
LSTM path needs:

| entry | id | purpose |
|---|---|---|
| `TESSDATA_LSTM` | 17 | serialized network graph + weights |
| `TESSDATA_LSTM_UNICHARSET` | 21 | character set |
| `TESSDATA_LSTM_RECODER` | 22 | unichar → CTC output-code mapping |
| `TESSDATA_LSTM_PUNC_DAWG` | 18 | punctuation lexicon |
| `TESSDATA_LSTM_SYSTEM_DAWG` | 19 | word lexicon |
| `TESSDATA_LSTM_NUMBER_DAWG` | 20 | number patterns |
| `TESSDATA_VERSION` | 23 | provenance |

Legacy entries (`INTTEMP`, `PFFMTABLE`, `NORMPROTO`, `SHAPE_TABLE`, `CUBE_*`) are
skipped. Plus: network-graph deserialization into the tensor runtime, LSTM forward
pass, and CTC beam-search decode with dawg lexicon weighting.

Start with `tessdata_best` (float weights). `tessdata_fast`'s int8-quantized path
(`IntSimdMatrix`) is a later optimization, not part of L1.

**Verify:** for a corpus of line crops, Go output text must match
`tesseract --psm 7` output. Per-character confidence within tolerance.

### L2 — classical line detector (~2k LOC)

Binarize → deskew → connected components → RLSA smear → projection-profile line
cuts → crop and normalize to the recognizer's input height.

**Deliverable: end-to-end OCR. Kleio drops the `tesseract` binary here.**

**Verify:** line-level IoU against Tesseract's own line boxes on a scanned-document
corpus; end-to-end CER against Tesseract on the same corpus.

### L3 — training (~4k LOC)

Backward pass through the same graph, CTC loss, correction corpus storage, model
versioning, and identification of documents to re-OCR when a new model version
ships.

**Deliverable: G3, the feedback loop.**

**Verify:** inject synthetic corrections into a deliberately degraded model and
confirm CER decreases; confirm a held-out split is not trained on.

### L4 — textord port (~20-25k LOC)

Incremental, in dependency order: `tabfind` → `colpartition` → `colfind` →
`makerow` → `baselinedetect` → `tospace` → `wordseg`. Carries the `ccstruct`
subset it needs.

**Deliverable:** multi-column layouts, mixed text/image pages, hard scans.

**Verify:** differential against the real `tesseract` binary *per stage*. Tesseract
retains debug dump modes for intermediate page-segmentation state; each ported
stage is diffed against its C++ counterpart on a fixed corpus before the next
begins. This oracle is what converts a large port from risky to merely long.

### L5 — CRNN / ONNX loader (~3k LOC)

ONNX model loading into the same tensor runtime, plus the ops modern CRNNs need
that Tesseract does not (batchnorm, depthwise-separable conv). BiLSTM-era
checkpoints only.

**Deliverable:** modern weights, A/B evaluation against L1 on Kleio's real corpus,
and optionally a confidence-weighted ensemble of both recognizers.

**Verify:** CER/WER against L1 on the Kleio corpus; a regression must not ship.

### Ordering

**L0 → L1 → L2** gets Kleio off the Tesseract binary. **L3** delivers the feedback
loop. **L4** and **L5** raise the accuracy ceiling and are independent of each
other. L1–L3 is a complete, useful project even if L4 never happens.

---

## 6. Model data

`.traineddata` files load through `io.Reader`. The caller owns file provisioning;
Kleio may `go:embed` a language model or bake it into its image.

**A weights blob is data, not a binary dependency.** G1's constraint is no
executables and no shared libraries; a model file is neither.

Fine-tuned models serialize through `Model.WriteTo` as a **self-contained** file
in Cadmus's own format: the full weight set with fine-tuning applied, plus
provenance metadata (base model identity, corpus fingerprint, iteration count,
held-out CER/WER). Self-contained rather than a delta, so loading a fine-tuned
model never requires resolving a base model — Kleio stores and serves one file per
version. Provenance makes each fine-tune reproducible and attributable to the
corrections that produced it.

---

## 7. Testing strategy

The project's central testing asset is a **differential oracle**: the real
`tesseract` binary, retained as a *test-only* dependency. It never ships and Kleio
never depends on it, but during development every layer can be compared against
the implementation it replaces.

| layer | oracle |
|---|---|
| L0 | Leptonica per-op output |
| L1 | `tesseract --psm 7` on line crops |
| L2 | Tesseract line boxes (IoU) + end-to-end CER |
| L3 | CER movement on held-out corrections |
| L4 | Tesseract page-segmentation debug dumps, per stage |
| L5 | L1 CER/WER on the Kleio corpus |

Corpus: a committed set of scanned document images with ground-truth transcripts,
covering clean single-column scans (Kleio's common case), multi-column, tables,
skewed scans, and low-DPI/noisy scans. Ground truth is checked in; images are
checked in if licensing permits, otherwise fetched by a documented script.

---

## 8. Kleio integration

Kleio's `kleio-ocr` worker replaces its `tesseract` subprocess call with a Cadmus
`Engine`. Kleio continues to rasterize PDFs and stamp text layers.

Kleio's validation gate does **not** consume a document-wide mean. Kleio Plan 2
measures confidence *spatially*: per-word `conf` values with bounding boxes are
bucketed into a 3×3 grid per sampled page, cells qualify at ≥5 words, and the gate
reads the 10th percentile of qualifying cell means. A single mean would hide a page
that is crisp at the top and mud at the bottom.

Cadmus therefore must emit **geolocated per-word confidence** — `Word.Bounds` plus
`Word.Confidence` — which is exactly the information Kleio currently parses out of
`tesseract ... tsv`. Regional aggregation stays on Kleio's side; it is a policy
decision, not an OCR one. `Page.MeanConfidence` remains as a convenience for
simpler consumers and is not on Kleio's path.

New in Kleio, enabled by G3: a UI affordance to correct a recognized word, which
persists `(LineImage, Truth)` to a corrections table; a periodic fine-tune job; and
re-OCR of documents affected by a new model version.

**Kleio's remaining binary dependencies after L2:** `ghostscript`, `jbig2enc`,
`pngquant`, `poppler` — all PDF-side. Eliminating those is a separate project.
`pngquant` is a few hundred lines of Go (median-cut quantizer) and `pdfcpu` covers
much of `ghostscript`'s optimization role, but pure-Go PDF **rasterization**
(replacing `poppler`) requires a full content-stream interpreter and font
rasterizer, and is plausibly larger than Cadmus itself. Out of scope here.

---

## 9. Licensing

Cadmus is **Apache-2.0**, matching Tesseract. A Go translation of Apache-2.0 C++
source is a derivative work; Apache-2.0 §4 requires retaining the license,
attribution notices, and stating changes. A `NOTICE` file credits Tesseract's
authors and identifies which packages are ports rather than original work.

Ported files carry a header naming the Tesseract source file they derive from.
This also serves engineering purposes: it makes the differential oracle's mapping
explicit, and it makes upstream bug-fix tracking possible.

Modern CRNN checkpoints adopted at L5 must have their licenses verified
individually before adoption; L5 ships no weights by default.

---

## 10. Risks

| risk | severity | mitigation |
|---|---|---|
| `.traineddata` network deserialization is idiosyncratic and under-documented | high — blocks L1's loader half | It is mechanical, and `--psm 7` gives an exact oracle. First task is a spike: parse `eng.traineddata` and dump the network graph structure. If it fights back, only the loader is lost — the tensor runtime is unaffected, and L5's ONNX loader is promoted ahead of L2 to supply weights instead. |
| L4 is a 20-25k LOC port | high — but late | Per-stage differential testing; L4 is optional to the project's value. L2 remains the fast path regardless. |
| Pure-Go CPU inference is slow | medium | Kleio's OCR is queue-driven batch work behind KEDA scaling; throughput scales horizontally. Optimization path exists: int8 quantized weights, then `src/arch`-equivalent SIMD via Go assembly. |
| Fine-tuning on user corrections degrades the model (overfitting, bad corrections) | medium | Held-out eval split, CER gate before a model version is promoted, and model versions are immutable and revertible. |
| Accuracy never reaches Tesseract parity | medium | L1 uses Tesseract's own weights, so line-level parity is the expected baseline, not an aspiration. Divergence would indicate a port bug, detectable by the oracle. |

---

## 11. Decisions taken

- **Separate repo**, not vendored into Kleio. The interface is narrow
  (`image.Image` → text), so there is no co-evolution pressure justifying a
  monorepo. Kleio pins a tagged version.
- **Both Tesseract weights and modern CRNN weights**, phased — one runtime, two
  loaders. L1 first, L5 later, never both at once.
- **Both classical CV and textord** detection, phased — L2 first, L4 later, both
  retained.
- **Zero third-party dependencies** for the core.
- **CPU only.**
- **Apache-2.0** with `NOTICE`.
