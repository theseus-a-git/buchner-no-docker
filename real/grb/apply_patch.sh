#!/usr/bin/env bash
# OPTIONAL. The original Docker image overwrote threeML's ultranest_sampler.py with the patched
# copy in patch/. That copy targets an old threeML, so it is NOT applied automatically.
# Try this only if grb.py fails inside threeML's ultranest wrapper. Original is backed up.
set -e
f=$(conda run -n inf-crab python -c "import threeML.bayesian.ultranest_sampler as m; print(m.__file__)" | tail -1)
cp -n "$f" "$f.orig"
cp "$(dirname "$0")/patch/ultranest_sampler.py" "$f"
echo "patched $f (backup: $f.orig); restore with: cp $f.orig $f"
