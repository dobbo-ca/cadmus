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

tmp="$(mktemp -d)"
# text2image drops a fontconfig stub next to itself; it is not part of the corpus.
trap 'rm -rf "$tmp"; rm -f fonts.conf' EXIT

i=0
while IFS= read -r text; do
  [ -z "$text" ] && continue
  i=$((i+1))
  id=$(printf '%04d' "$i")
  printf '%s\n' "$text" > "native/$id.gt.txt"
  cp "native/$id.gt.txt" "h36/$id.gt.txt"

  printf '%s' "$text" > "$tmp/line.txt"
  text2image --text="$tmp/line.txt" --outputbase="$tmp/$id" \
    --font="$FONT" --ptsize="$PTSIZE" --margin=8 --leading=0 \
    --xsize=4000 --ysize=120 --unicharset_file=/dev/null >/dev/null 2>&1

  # Trim to the ink and flatten to 8-bit grey.
  magick "$tmp/$id.tif" -colorspace Gray -trim +repage -bordercolor white \
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
      "$png" - 2>/dev/null | sed -e '${/^$/d;}' > "$base.psm13.txt"
  done
done

echo "corpus: $i lines x 2 arms"
wc -l corpus.txt
