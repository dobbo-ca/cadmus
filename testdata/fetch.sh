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
