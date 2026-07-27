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
