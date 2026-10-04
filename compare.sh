#!/usr/bin/env bash
# Compare finished runs (evidence, posterior accuracy vs a reference sampler, ncall, time).
#   ./compare.sh                                  everything
#   ./compare.sh --ref dynesty --problems synthetic/toy
# Uses the synthetic env (it has numpy, scipy and matplotlib); falls back to the python on PATH.
cd "$(dirname "$0")"
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx inf-synthetic; then
  exec conda run --no-capture-output -n inf-synthetic python compare_samplers.py "$@"
else
  exec python3 compare_samplers.py "$@"
fi
