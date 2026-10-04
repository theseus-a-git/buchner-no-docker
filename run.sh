#!/usr/bin/env bash
# ./run.sh [--force] [--list] <path> [sampler ...]
#   ./run.sh real/ligo ultranest dynesty nestle emcee pso
#   ./run.sh synthetic/toy                  # every problem in toy/problems.list, all five backends
#   ONLY="eggbox-2d beta-2d" ./run.sh synthetic/toy dynesty
#   ./run.sh real ultranest-safe               # every real problem
# --force / FORCE=1 reruns finished jobs; --list shows the available problem directories.
ROOT="$(cd "$(dirname "$0")" && pwd)"
FORCE=${FORCE:-0}
while [[ "${1:-}" == --* ]]; do
  case $1 in --force) FORCE=1;; --list) find "$ROOT"/{real,synthetic} -name problem.conf -printf '%h\n' | sed "s#$ROOT/##" | sort; exit 0;;
    *) echo "unknown option $1"; exit 2;; esac; shift
done
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 2; }
target="${1%/}"; shift; SAMPLERS=("$@"); [ ${#SAMPLERS[@]} -gt 0 ] || SAMPLERS=(ultranest dynesty nestle emcee pso)
[ -d "$ROOT/$target" ] || { echo "no such directory: $target (try --list)"; exit 2; }

run_dir() {
  local dir="$ROOT/$1"
  (
    cd "$dir" || exit 1
    unset ENV CMD CASES FIXED_SAMPLER BROKEN MANUAL
    source ./problem.conf
    [ -n "${MANUAL:-}" ] && { echo "[$1] manual: $MANUAL"; exit 0; }
    [ -n "${BROKEN:-}" ] && [ "${FORCE_BROKEN:-0}" != 1 ] && { echo "[$1] skipped: $BROKEN"; exit 0; }
    local samplers=("${SAMPLERS[@]}")
    [ ${#samplers[@]} -gt 0 ] || { echo "[$1] needs at least one sampler name"; exit 2; }
    for s in "${samplers[@]}"; do
      case "$s" in
        ultranest|dynesty|nestle|emcee|pso) ;;   # CORE: the five-sampler benchmark
        multinest|ultranest-safe|dynesty-multiell|testsampler|ultranest-fast|ultranest-faster|ultranest-fast-fixed100|ultranest-fast-fixed4d|ultranest-HD|goodman-weare|slice|vbis|vbis-wide|vegas|lhsgrid)
          echo "[$1] note: '$s' is a legacy/optional backend, not part of the five-sampler comparison" ;;
        *) echo "[$1] unknown sampler: $s"; exit 2 ;;
      esac
    done
    local cases=${CASES:-$(basename "$1")}
    [ -n "${ONLY:-}" ] && cases=$ONLY
    rc=0
    for CASE in $cases; do for SAMPLER in "${samplers[@]}"; do
      mark="runs/$CASE/$SAMPLER"
      if [ -e "$mark/.done" ] && [ "$FORCE" != 1 ]; then echo "skip $1 $CASE $SAMPLER (done)"; continue; fi
      mkdir -p "$mark"; rm -f "$mark/.done"
      echo; echo "===== $1  CASE:$CASE  SAMPLER:$SAMPLER  ENV:$ENV ====="
      cmd=$(eval "echo \"$CMD\"")
      SAMPLER=$SAMPLER PROBLEM=$CASE MAX_NCALLS=${MAX_NCALLS:-100000} PSO_USE_MAX_NCALLS=${PSO_USE_MAX_NCALLS:-1} PYTHONPATH="$ROOT/common:${PYTHONPATH:-}" \
        conda run -n "$ENV" --no-capture-output $cmd 2>&1 | tee "$mark/log.txt"
      if [ "${PIPESTATUS[0]}" -eq 0 ]; then touch "$mark/.done"
      else echo "!! $1 $CASE $SAMPLER failed, see $dir/$mark/log.txt"; rc=1; fi
    done; done
    exit $rc
  )
}

# exit status is non-zero if any run failed (used by the Slurm array tasks in slurm/)
RC=0
if [ -f "$ROOT/$target/problem.conf" ]; then run_dir "$target" || RC=1
else while read -r d; do run_dir "$d" || RC=1; done < <(find "$ROOT/$target" -name problem.conf -printf '%h\n' | sed "s#$ROOT/##" | sort); fi
exit $RC
