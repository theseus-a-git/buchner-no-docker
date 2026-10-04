#!/usr/bin/env bash
# Remove outputs, caches, backups and downloaded data from this folder.
#   bash clean.sh          dry run: only lists what WOULD be deleted
#   bash clean.sh --yes    actually delete
# Also removed: slurm/logs and slurm/jobs (Slurm job logs, task lists and the ledger used by slurm/status.sh --sacct).
# Kept: scripts, environment.yml/problem.conf/smoke.py, exoplanet data/ and juliet/, mosfit LSQ12dlf.json,
#       xray-agn-ciao models/ (1.1 GB) and the BXA clone, common/, setup.sh, run.sh. Conda envs are not touched.
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DO=0; [ "${1:-}" = "--yes" ] && DO=1
LIST=$(mktemp)
{
  find real synthetic -maxdepth 2 \( -name runs -o -name systematiclogs \) -type d
  find . -name __pycache__ -type d
  find . \( -name "*:Zone.Identifier" -o -name "*.orig" -o -name "*.bak" -o -name "*~" -o -name "*_nlive50" \)
  find real/crab -maxdepth 1 \( -name Crab_data -o -name chains -o -name '__[0-9a-f]*' \)
  find real/grb -maxdepth 1 \( -name 'glg_*' -o -name '*_bkg.h5' \)
  find real/lj6 -maxdepth 1 -name 'LJ[0-9]*' -type d
  find . -maxdepth 1 -name 'sampler_comparison.*'
  find slurm -maxdepth 1 \( -name logs -o -name jobs \) -type d 2>/dev/null
  find real/posteriorstacker/PosteriorStacker -maxdepth 1 -name 'posteriorsamples.txt*' 2>/dev/null
  find real/xray-agn-ciao/BXA/examples/sherpa/chandra -maxdepth 1 -name '179.pi_out*' 2>/dev/null
} | sed 's#^\./##' | sort -u > "$LIST"
if [ ! -s "$LIST" ]; then echo "nothing to clean"; rm -f "$LIST"; exit 0; fi
while read -r p; do [ -e "$p" ] && printf '%8s  %s\n' "$(du -sh "$p" 2>/dev/null | cut -f1)" "$p"; done < "$LIST"
echo "--- $(wc -l < "$LIST") items, total $(xargs -a "$LIST" -d '\n' du -ch 2>/dev/null | tail -1 | cut -f1)"
if [ $DO = 1 ]; then
  xargs -a "$LIST" -d '\n' rm -rf; echo "deleted."
else
  echo "dry run only. Run: bash clean.sh --yes"
fi
rm -f "$LIST"
