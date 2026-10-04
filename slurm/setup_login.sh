#!/usr/bin/env bash
# One-time setup on the LOGIN node: Miniforge (if conda is missing) + the conda environments (../setup.sh).
#   slurm/setup_login.sh                    all environments (long: run it inside screen/tmux or with nohup)
#   slurm/setup_login.sh synthetic real/ligo     only these
# Environments and package caches go to $IE_SCRATCH (default /scratch/$USER), not to the 50 GB $HOME quota.
# real/cosmology compiles CLASS: if the default gcc is old, set IE_SETUP_MODULES to a newer one (see `module avail gcc`).
. "$(dirname "$0")/env.sh"
if ! command -v conda >/dev/null 2>&1; then
  bash "$INFENV_ROOT/slurm/install_miniforge.sh" || exit 1
  . "$INFENV_ROOT/slurm/env.sh"
fi
[ -n "${IE_SETUP_MODULES:-}" ] && ie_module $IE_SETUP_MODULES
mkdir -p "$INFENV_ROOT/slurm/logs"
cd "$INFENV_ROOT" && exec ./setup.sh "$@"
