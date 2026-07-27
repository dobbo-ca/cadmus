# Cadmus

A pure-Go OCR engine. No cgo, no shared libraries, no subprocesses.

Cadmus reads Tesseract's `.traineddata` models directly — Tesseract's LSTM engine
is a CRNN (conv → maxpool → bidirectional LSTM → softmax → CTC), so its weights
load into a Go tensor runtime that also supports training. That makes OCR output
correctable: a wrong transcription can be fed back as ground truth and the model
fine-tuned in-process.

> Cadmus brought the alphabet to Greece. This library turns pixels back into
> letters.

**Status:** design approved, implementation not started. See
[the design spec](docs/superpowers/specs/2026-07-27-cadmus-design.md).

## Why

Three properties, in priority order:

1. **No binary dependencies.** Pure Go — and specifically no `.so` either, which
   rules out `purego`-loaded ONNX runtimes.
2. **Tunable.** Every heuristic and weight is in-repo and changeable.
3. **Trainable.** Corrections from a consuming application improve the model.

Wrapping Tesseract — via cgo or via WASM under wazero — delivers only the first.

## License

Apache-2.0. Portions are Go ports of Tesseract OCR (also Apache-2.0); see
[`NOTICE`](NOTICE).
