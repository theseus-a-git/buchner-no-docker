#!/usr/bin/env bash
# Print one line per run:  <problem dir> TAB <case> TAB <sampler>
#   slurm/make_manifest.sh                      every real/ and synthetic/ problem x the five samplers
#   slurm/make_manifest.sh real/ligo synthetic/toy
#   IE_SAMPLERS="pso emcee" slurm/make_manifest.sh
# The case expansion is the same as in run.sh (problem.conf: CASES, default = directory name).
# Problems marked MANUAL / BROKEN in their problem.conf are reported on stderr and left out.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLERS=${IE_SAMPLERS:-"ultranest dynesty nestle emcee pso"}
TARGETS=("$@"); [ ${#TARGETS[@]} -gt 0 ] || TARGETS=(real synthetic)
for t in "${TARGETS[@]}"; do
  t="${t%/}"
  [ -d "$ROOT/$t" ] || { echo "make_manifest: no such directory: $t" >&2; exit 2; }
  find "$ROOT/$t" -name problem.conf -printf '%h\n' | sed "s#^$ROOT/##" | sort
done | sort -u | while read -r d; do
  (
    cd "$ROOT/$d" || exit 1
    unset ENV CMD CASES FIXED_SAMPLER BROKEN MANUAL
    source ./problem.conf
    [ -n "${MANUAL:-}" ] && { echo "skip $d (manual): $MANUAL" >&2; exit 0; }
    [ -n "${BROKEN:-}" ] && { echo "skip $d (broken): $BROKEN" >&2; exit 0; }
    cases=${CASES:-$(basename "$d")}
    for c in $cases; do for s in $SAMPLERS; do printf '%s\t%s\t%s\n' "$d" "$c" "$s"; done; done
  )
done
