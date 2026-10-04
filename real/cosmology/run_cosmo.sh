#!/usr/bin/env bash
# CMB cosmology example with MontePython + CLASS. Buchner's MontePython fork (mode "-m UN") calls common/autosampler.py,
# so $SAMPLER (set by run.sh) picks the sampler and $LOGDIR is where autosampler writes its results.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/montepython_public/default.conf" ] || { echo "not set up, run: ./setup.sh real/cosmology"; exit 1; }
export MPLBACKEND=Agg
export LOGDIR="$HERE/systematiclogs/${PROBLEM:-cosmology}/$SAMPLER"
rm -rf "$LOGDIR"; mkdir -p "$(dirname "$LOGDIR")"
cd "$HERE/montepython_public"
python montepython/MontePython.py run -m UN -o "$LOGDIR" -p input/example_ns.param
echo "results in real/cosmology/systematiclogs/${PROBLEM:-cosmology}/$SAMPLER"
