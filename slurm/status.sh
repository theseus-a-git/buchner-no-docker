#!/usr/bin/env bash
# Progress of the benchmark (and, with --sacct, what Slurm says about each submitted task).
. "$(dirname "$0")/env.sh"
exec "$IE_PYTHON" "$INFENV_ROOT/slurm/status.py" "$@"
