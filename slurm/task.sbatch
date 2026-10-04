#!/bin/bash
# One Slurm array task = one (problem dir, case, sampler) run. Submitted by slurm/submit.sh; the sbatch options
# (partition, time, memory, array range, log file) are given on the sbatch command line, not here.
#   INFENV_ROOT   the code tree            IE_TASKFILE  tab-separated task list; line $SLURM_ARRAY_TASK_ID is ours
# (no `set -u` before env.sh: conda's own shell hooks reference unset variables)
: "${INFENV_ROOT:?not set (submit with slurm/submit.sh)}"
. "$INFENV_ROOT/slurm/env.sh"
set -u
: "${IE_TASKFILE:?not set}"
IFS=$'\t' read -r dir case_name sampler max_ncalls < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IE_TASKFILE")
[ -n "${sampler:-}" ] || { echo "no task ${SLURM_ARRAY_TASK_ID} in $IE_TASKFILE"; exit 1; }

# node-local scratch for temp files / numba cache (the shared Lustre file system is slow for many tiny writes)
export TMPDIR="${TMPDIR:-/tmp}/ie-${USER}-${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"; export NUMBA_CACHE_DIR="$TMPDIR/numba"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[ie] job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID}  node $(hostname)  $(date '+%F %T')"
echo "[ie] $dir  case=$case_name  sampler=$sampler  MAX_NCALLS=$max_ncalls  mem=${SLURM_MEM_PER_NODE:-?}MB  cpus=${SLURM_CPUS_PER_TASK:-1}"

cd "$INFENV_ROOT" || exit 1
[ "${IE_FORCE:-0}" = 1 ] && export FORCE=1
ONLY="$case_name" MAX_NCALLS="$max_ncalls" ./run.sh "$dir" "$sampler"
rc=$?
echo "[ie] finished rc=$rc  $(date '+%F %T')"
exit $rc
