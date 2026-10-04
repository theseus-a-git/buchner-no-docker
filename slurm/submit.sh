#!/usr/bin/env bash
# Submit the benchmark to Slurm.  See slurm/README.md, or:  slurm/submit.sh --help
. "$(dirname "$0")/env.sh"
exec "$IE_PYTHON" "$INFENV_ROOT/slurm/submit.py" "$@"
