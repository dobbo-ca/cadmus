#!/usr/bin/env bash
# Captures Tesseract's per-layer activation dump for one line image, using a
# locally built instrumented tesseract. Manual step; never run in CI.
#
# Building the instrumented binary (tesseract 5.5.3, commit a62bcef):
#
#   git clone --depth 1 --branch 5.5.3 https://github.com/tesseract-ocr/tesseract
#   cd tesseract
#   git apply /path/to/cadmus/testdata/tessdebug/debug-detail.patch
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DDISABLE_ARCHIVE=ON \
#     -DDISABLE_CURL=ON -DBUILD_TRAINING_TOOLS=OFF -DGRAPHICS_DISABLED=ON
#   cmake --build build -j8 --target tesseract
#   export TESS_BIN="$PWD/build/bin/tesseract"    # note bin/, not build/tesseract
#
# The patch raises DEBUG_DETAIL to 1, which also `#undef _OPENMP`s and keeps the
# interleaving deterministic, and it normalizes the dump so that every layer
# emits exactly one block under a left-anchored "Output:<name>" header:
#
#   * FullyConnected's header loses its "F " prefix.
#   * LSTM's Source:/State: blocks move to DEBUG_DETAIL > 1, so nothing sits
#     between two Output: headers. A block-keyed diff extracts the range from
#     "^Output:X$" to the next "^Output:", and anything in between is swept
#     into the preceding block.
#   * Convolve, Maxpool and Input gain the print they never had. Input is the
#     first block: it is the tensor the network is actually fed, and it is what
#     cadmusdump reports as "Output:Normalized".
#   * NetworkIO::Print stops windowing to the first and last 10 timesteps, and
#     prints %.9g rather than %g. 6 significant digits quantizes the comparison
#     at ~1e-6 relative, which is above the float32-store divergence the delta
#     profile is meant to measure; 9 digits round-trips a float exactly.
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
