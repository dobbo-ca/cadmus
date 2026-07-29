#!/usr/bin/env bash
# Prints the per-layer max-abs-delta profile between a cadmusdump -activations
# dump and a capture.sh dump of the same line image. Manual step; never run in
# CI.
#
#   go run ./cmd/cadmusdump -activations testdata/eng.traineddata line.png >/tmp/act/cadmus.txt
#   TESS_BIN=... TESSDATA=testdata ./testdata/tessdebug/capture.sh line.png /tmp/act/tess.txt
#   ./testdata/tessdebug/blockdelta.sh /tmp/act/cadmus.txt /tmp/act/tess.txt
#
# Read the PROFILE, not any single number. A healthy one grows slowly and
# monotonically; the first block whose delta jumps by more than two orders of
# magnitude above its predecessor is the broken layer, and nothing after it
# means anything.
#
# Two things this does that a plain `sed` range does not, both of which
# silently corrupt the comparison:
#
#   * It stops at the end of the FIRST block. Tesseract runs the recognizer
#     twice per image, so every header appears twice and a naive range match
#     concatenates both passes.
#   * It maps cadmus's "Normalized" onto tesseract's "Input". They are the same
#     tensor; cadmus renames it so its own Input layer, which is the identity,
#     does not emit a second block under a header that already exists.
set -euo pipefail
CADMUS="${1:?usage: blockdelta.sh <cadmus.txt> <tess.txt>}"
TESS="${2:?usage: blockdelta.sh <cadmus.txt> <tess.txt>}"

block() { # block <file> <layer name>
  awk -v n="$2" 'index($0,"Output:")==1 { if (f) exit; f = (substr($0,8)==n); next } f' "$1"
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

printf '%-12s %-12s %6s %6s  %s\n' cadmus tesseract rowsC rowsT max-abs-delta
for pair in Normalized:Input Convolve:Convolve ConvNL:ConvNL Maxpool:Maxpool \
            Lfys64:Lfys64 Lfx96:Lfx96 Lrx96:Lrx96 Lfx512:Lfx512 Output:Output; do
  c=${pair%%:*}
  t=${pair##*:}
  block "$CADMUS" "$c" >"$tmp/a"
  block "$TESS" "$t" >"$tmp/b"
  rows_a=$(wc -l <"$tmp/a")
  rows_b=$(wc -l <"$tmp/b")
  # Unequal row counts mean the block shapes differ, and the paste below would
  # compare unrelated features; say so rather than printing a number.
  if [ "$rows_a" -ne "$rows_b" ] || [ "$rows_a" -eq 0 ]; then
    printf '%-12s %-12s %6s %6s  SHAPE MISMATCH\n' "$c" "$t" "$rows_a" "$rows_b"
    continue
  fi
  d=$(paste "$tmp/a" "$tmp/b" |
    awk '{ h = NF/2; for (i = 1; i <= h; i++) { x = $i - $(i+h); if (x < 0) x = -x; if (x > m) m = x } }
         END { printf "%.4g", m + 0 }')
  printf '%-12s %-12s %6s %6s  %s\n' "$c" "$t" "$rows_a" "$rows_b" "$d"
done
