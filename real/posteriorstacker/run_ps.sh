#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$HERE/PosteriorStacker" ] || { echo "not cloned, rerun: ./setup.sh real/posteriorstacker" >&2; exit 1; }
export MPLBACKEND=Agg
exec python -u "$HERE/posteriorstacker_runner.py"
