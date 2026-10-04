#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  exo-transient)
    exec python -u transient.py
    ;;
  rvs-0|rvs-1|rvs-2|rvs-3)
    exec python -u exoplanet.py --planets="${1#rvs-}" --rvfile=data/rvs_0005.txt
    ;;
  *)
    echo "unknown exoplanet case: ${1:-<missing>}" >&2
    exit 2
    ;;
esac
