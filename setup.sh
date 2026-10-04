#!/usr/bin/env bash
# Build (or update) the conda env for each problem directory.
#   ./setup.sh                      all environments
#   ./setup.sh real/ligo synthetic  only these (any dir containing environment.yml)
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
command -v conda >/dev/null || { echo "conda not found on PATH (install Miniforge first)"; exit 1; }
if [ $# -gt 0 ]; then DIRS=("${@%/}"); else mapfile -t DIRS < <(find real synthetic -name environment.yml -printf '%h\n' | sort); fi

OK=(); FAIL=()
for d in "${DIRS[@]}"; do
  [ -f "$d/environment.yml" ] || { echo "skip $d (no environment.yml)"; continue; }
  name=$(awk '/^name:/{print $2}' "$d/environment.yml")
  echo; echo "=== $d  ->  $name ==="
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    conda env update -n "$name" -f "$d/environment.yml" --prune
  else
    conda env create -f "$d/environment.yml"
  fi
  if [ $? -ne 0 ]; then FAIL+=("$d"); continue; fi
  [ -x "$d/post_setup.sh" ] && "$d/post_setup.sh"
  if [ -f "$d/smoke.py" ] && ! conda run -n "$name" python "$d/smoke.py" >/dev/null 2>&1; then
    echo "!! $name built, but its import test failed: conda run -n $name python $d/smoke.py"; FAIL+=("$d (imports)")
  else OK+=("$d"); fi
done
echo; echo "ok:     ${OK[*]:-none}"; echo "failed: ${FAIL[*]:-none}"
[ ${#FAIL[@]} -eq 0 ]
