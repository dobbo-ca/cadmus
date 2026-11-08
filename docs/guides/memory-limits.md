# Memory limits: what cadmus bounds, and what it will

This page states the resource-limit contract every dobbo-ca library meets
(section 9A.1 of the kleio in-process pipeline design) as it applies to
cadmus, then lists exactly what the code enforces today and what is planned,
with the phase that lands it. Today cadmus has no exported API and no
resource limits; the contract below is the target for phase 4.2.

Line numbers are at cadmus `a6ce1ca`. Planned values are the design's
estimates and are final only when phase 4.2 ships.

## The contract

### Memory budget

`cadmus.SetMemoryLimit(n int64) int64` sets how many bytes all cadmus
operations in flight may hold together. `n <= 0` means no limit. It returns
the previous value. The default is half of the Go memory limit (`GOMEMLIMIT`
or `debug.SetMemoryLimit`), else half of the cgroup `memory.max` on Linux,
else none.

An operation reserves its estimate before it allocates. The budget covers:

- the model load (the `traineddata` file and what is built from it);
- what cadmus derives from the page plane: the binarized copy, the layout
  structures and the LSTM activations.

It does not cover the page plane. The caller owns the plane, reserves it from
its own budget (kleio's OCR planes budget) and passes it in. cadmus makes its
raster-go calls on that plane by band, in buffers reserved from cadmus's own
budget, because raster-go has no package budget and caps each call's scratch
by its `Limits`.

### Per-call caps

A `Limits` struct, zero value safe, bounds one call. A cap is checked from
the header or the plane's dimensions before allocation. The defaults are:

| Cap | Default |
|---|---|
| `traineddata` file | 64 MiB |
| OCR plane | 1<<27 px (134 217 728; 128 MiB of grey) |
| lines per page | 20 000 |
| line width | 65 536 px |
| goroutines (`MaxGoroutines`) | `GOMAXPROCS` |

A plane over 1<<27 px is refused with its `LimitError`; the caller scales the
page before the call. The plane cap decides the OCR scale, not whether a page
is OCRed at all: kleio scales a 150 Mpx page down to fit. Caps hold with no
budget set.

Why banding matters: raster-go's default per-call cap is 1<<26 px, and the
whole-plane Sauvola in cadmus allocates a `uint32` and a `float64` per padded
pixel, about 1.5 GiB at 1<<27 px against a 300 MiB budget. Per band, a
512-row band of a 10 000 px wide plane needs about 59 MiB, and the 1-bpp
binarized copy needs 16 MiB at 1<<27 px.

### Errors

- `errors.Is(err, cadmus.ErrMemoryLimit)`: transient. Another operation holds
  the budget, and this one would fit alone. Retry, or answer HTTP 503.
- `errors.Is(err, cadmus.ErrLimit)`, with `*cadmus.LimitError{Name, Got, Max}`
  from `errors.As`: permanent. This input breaches a cap, or its own estimate
  is over the whole budget. It does not wrap `ErrMemoryLimit`. Do not retry.

### Time

`RecognizePage(ctx, plane, dpi, langs)` takes `ctx` and checks it between
lines. Its target is to return within 100 ms of cancellation on a 300 dpi
letter page. Any smaller entry point (one line, one matrix product) takes no
`ctx`, and its caller checks `ctx` between calls, per owner decision D2.

### Concurrency

Internal fan-out is bounded by `Limits.MaxGoroutines`, default `GOMAXPROCS`.
`RecognizePage` is safe to call concurrently on distinct planes; the budget
bounds their memory together.

## Today

cadmus is a set of `internal/` packages plus one dump tool
(`cmd/cadmusdump`). It has no importable package and so no entry point to
which a limit could attach.

### Not present

No `SetMemoryLimit`, `ErrMemoryLimit`, `Limits`, `LimitError` or `ErrLimit`.
No `context` import in any `.go` file. No goroutines. No cap on the plane,
on lines, on line width or on the size of the model file: `tessdata` parses a
`[]byte` the caller has already read (`cmd/cadmusdump/main.go:23` reads it
with `os.ReadFile`, with no size check).
`imaging.NewBitmap` allocates `stride*h` bytes from whatever `w` and `h` it
is given (`internal/imaging/bitmap.go:36-42`), and whole-plane
`imaging.Sauvola` allocates a `uint32` and a `float64` per padded pixel
(`internal/imaging/binarize.go:100-101`).

### Enforced

The only bounds are sanity checks inside the model parser, taken from
Tesseract's own guards. They protect against a corrupt model, not against
memory use, and none is configurable:

| Bound | Value | Where |
|---|---|---|
| `traineddata` container entries | 1000 | `internal/tessdata/container.go:14`, checked at `:84` |
| Serialized string or array length | 50 000 000 | `internal/tessdata/tfile.go:15`, checked at `:139` |
| DAWG edges | 50 000 000 | `internal/tessdata/dawg.go:25`, checked at `:70` |
| Unicharset size | 1 000 000 | `internal/tessdata/unicharset.go:37`, checked at `:78` |
| Network stack size | 10 000 | `internal/tessdata/network.go:142`, checked at `:499` |
| Matrix rows or columns | 65 535 | `internal/tessdata/network.go:145`, checked at `:637` |
| Recoder code length | 9 | `internal/tessdata/recoder.go:11` |

They bound counts, not bytes. Memory is set by the size of the model
file and the plane, and nothing caps either.

### Planned

| What | Phase |
|---|---|
| Importable `cadmus` package with `RecognizePage(ctx, plane, dpi, langs)` | 4.2 |
| `SetMemoryLimit`, `ErrMemoryLimit`, `Limits`, `LimitError`, `ErrLimit`, with the defaults in the table above | 4.2 |
| `ctx` checked between lines, 100 ms cancellation bound on a 300 dpi letter page | 4.2 |
| Banded raster-go calls on the plane (`TestSauvolaBandedEqualsWhole`), buffers reserved from cadmus's budget | 4.2 |
| Limit tests of 9A.4: `TestEstimateCoversAllocation`, `TestRefusesBeforeAllocating`, `TestBudgetHolds`, `TestDefaultsAreSafe`, cancellation, goroutines | 4.2 |
| Measured cost of the limiter, by jp2k-go's method, replacing the note below | 4.2 exit |
| CPU work after the switch-over (int8 matvec tiers, batched LSTM, beam pruning); these change time, not the memory contract | 4.4 |

## Cost of the limiter

Not measured yet, because the limiter does not exist. The 4.2 exit measures it
with the method of jp2k-go's memory-limits guide and this section then holds
the numbers.

## Handling the errors

- `ErrMemoryLimit`: retry once, then count it as a bug if the caller runs one
  page at a time under a static budget.
- `*LimitError` or `ErrLimit`: scale the page or park the document with the
  cap's name; do not retry.
- Put a deadline on the context; waiting for the budget has no timeout of its
  own. kleio gives each OCR page `JobDeadlinePerPage` and ships the page
  without text if it expires.
