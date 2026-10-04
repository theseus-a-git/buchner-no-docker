#!/usr/bin/env bash
# After the env is built: fetch BXA (it contains xagnfitter.py and the 179.pi example spectrum).
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "$HERE/BXA" ]; then
  git clone --depth 1 https://github.com/JohannesBuchner/BXA "$HERE/BXA" || { echo "!! could not clone BXA"; exit 1; }
fi
echo "BXA ready in $HERE/BXA"
[ -s "$HERE/models/uxclumpy-cutoff.fits" ] || echo "Next: bash real/xray-agn-ciao/fetch_models.sh   (downloads ~1.1 GB of UXCLUMPY tables)"
