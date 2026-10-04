#!/usr/bin/env bash
# Fit the Chandra CDFS spectrum 179.pi with BXA + UXCLUMPY (run inside the CIAO env by run.sh).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# Download the UXCLUMPY tables (~1.1 GB, resumable, md5-checked) on first use
if [ ! -s "$HERE/models/uxclumpy-cutoff.fits" ] || [ ! -s "$HERE/models/uxclumpy-cutoff-omni.fits" ]; then
  echo "UXCLUMPY tables missing, downloading (~1.1 GB)..."
  bash "$HERE/fetch_models.sh"
fi
[ -d "$HERE/BXA/examples/sherpa/chandra" ] || { echo "BXA not cloned, rerun: ./setup.sh real/xray-agn-ciao"; exit 1; }
# Fix 1: CIAO's FITS reader (pycrates) is broken with this numpy, so make Sherpa use astropy
if [ ! -f "$HOME/.sherpa.rc" ]; then
  cp "$(python -c 'import sherpa; print(sherpa.get_config())')" "$HOME/.sherpa.rc"
  sed -i 's/^io_pkg.*/io_pkg     : pyfits/' "$HOME/.sherpa.rc"
fi
# Fix 2: XSPEC table models need load_xstable_model in Sherpa >= 4.18
sed -i 's/load_table_model(/load_xstable_model(/' "$HERE/BXA/examples/sherpa/xagnfitter.py"
export MPLBACKEND=Agg MODELDIR="$HERE/models" WITHAPEC=0
cd "$HERE/BXA/examples/sherpa/chandra"
echo "179.pi 0.5 8" > filenames.txt
export MPLBACKEND=Agg MODELDIR="$HERE/models" WITHAPEC=0
python "$HERE/xray_multi.py"
