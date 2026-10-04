#!/usr/bin/env bash
# Downloads the two UXCLUMPY table models (~520 MB each) from Zenodo record 1169181. Resumable.
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/models" && cd "$HERE/models" || exit 1
for f in uxclumpy-cutoff.fits uxclumpy-cutoff-omni.fits; do
  curl -fL -C - -o "$f" "https://zenodo.org/records/1169181/files/$f?download=1" || { echo "!! download of $f failed"; exit 1; }
done
md5sum -c - <<SUMS
9bdffd414a74f6c327032fb5274d8406  uxclumpy-cutoff.fits
6dc751d3e4a9da2a41c5f01beca5bd91  uxclumpy-cutoff-omni.fits
SUMS
