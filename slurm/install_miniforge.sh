#!/usr/bin/env bash
# Install Miniforge (conda + conda-forge) under $CONDA_ROOT (default /scratch/$USER/miniforge3). Run on the LOGIN node.
# PARAM Seva's own module (conda-python/3.7) is Python 3.7 only; the inf-* environments need Python 3.9-3.11.
set -euo pipefail
. "$(dirname "$0")/env.sh"
if [ -x "$CONDA_ROOT/bin/conda" ]; then echo "conda already installed in $CONDA_ROOT"; exit 0; fi
mkdir -p "$(dirname "$CONDA_ROOT")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
echo "downloading $url"
if command -v curl >/dev/null; then curl -fL -o "$tmp/miniforge.sh" "$url"; else wget -O "$tmp/miniforge.sh" "$url"; fi \
  || { echo "!! download failed. Download Miniforge3-Linux-x86_64.sh on your own machine, scp it to the cluster and run: bash it -b -p $CONDA_ROOT"; exit 1; }
bash "$tmp/miniforge.sh" -b -p "$CONDA_ROOT"
echo "installed: $("$CONDA_ROOT/bin/conda" --version) in $CONDA_ROOT"
