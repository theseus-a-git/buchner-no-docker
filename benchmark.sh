#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GROUP="all"; FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    real|synthetic|all) GROUP="$1";;
    --force) FORCE="--force";;
    *) echo "usage: $0 [real|synthetic|all] [--force]" >&2; exit 2;;
  esac
  shift
done
case "$GROUP" in
  real) "$ROOT/run.sh" $FORCE real ultranest dynesty nestle emcee pso;;
  synthetic) "$ROOT/run.sh" $FORCE synthetic ultranest dynesty nestle emcee pso;;
  all) "$ROOT/run.sh" $FORCE real ultranest dynesty nestle emcee pso; "$ROOT/run.sh" $FORCE synthetic ultranest dynesty nestle emcee pso;;
esac
"$ROOT/compare.sh" --samplers ultranest dynesty nestle emcee pso --ref ultranest
