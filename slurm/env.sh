# Sourced (never executed) by every script in slurm/.  Edit the defaults here or override them from
# your shell / ~/.bashrc, e.g.   export IE_ACCOUNT=myproject
#
# PARAM Seva facts this is based on (user manual v2, Slurm 20.11): partitions standard(default)/cpu/gpu/hm, max
# walltime 4 days, default walltime 2 h, 48-core 192 GB CPU nodes, $HOME quota 50 GB, /scratch quota 200 GB
# (files not accessed for 3 months are deleted), compute nodes may be shared between users.
_ie_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INFENV_ROOT="$(cd "$_ie_here/.." && pwd)"

# ---- where things live ------------------------------------------------------------------------------------
: "${IE_SCRATCH:=/scratch/${USER}}"                 # big, fast, but purged after 3 months without access
: "${CONDA_ROOT:=$IE_SCRATCH/miniforge3}"           # conda + all inf-* environments (several GB each: keep OFF $HOME)

# ---- scheduler defaults (all overridable per call or in resources.tsv) ---------------------------------------
: "${IE_PARTITION:=cpu}"          # CPU-only nodes. 'standard' also contains the GPU nodes; see `sinfo`
: "${IE_ACCOUNT:=}"               # sbatch --account (the manual says to set it "if you face any problems")
: "${IE_QOS:=}"
: "${IE_MAIL:=}"                  # e-mail address for END/FAIL notices (one mail per array, not per task)
: "${IE_THROTTLE:=48}"            # max simultaneously running tasks of one array (one node's worth of cores)
: "${IE_MAX_TIME:=4-00:00:00}"    # cluster limit; --time-mult is clamped to this
: "${IE_SAMPLERS:=ultranest dynesty nestle emcee pso}"
: "${IE_SETUP_MODULES:=}"         # e.g. a gcc module for building CLASS on the login node (see `module avail gcc`)
export IE_SCRATCH CONDA_ROOT IE_PARTITION IE_ACCOUNT IE_QOS IE_MAIL IE_THROTTLE IE_MAX_TIME IE_SAMPLERS IE_SETUP_MODULES

# ---- conda -------------------------------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  for _c in "$CONDA_ROOT/etc/profile.d/conda.sh" "$HOME/miniforge3/etc/profile.d/conda.sh" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$_c" ] && { . "$_c"; break; }
  done
fi
# python used by the helper scripts (submit/status): the Miniforge one if present, else the system python3
if [ -x "$CONDA_ROOT/bin/python" ]; then : "${IE_PYTHON:=$CONDA_ROOT/bin/python}"; else : "${IE_PYTHON:=python3}"; fi
export IE_PYTHON

# ---- keep caches off the small $HOME quota and make them identical on login and compute nodes ----------------
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$IE_SCRATCH/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$IE_SCRATCH/.cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$IE_SCRATCH/.cache}"      # astropy's download cache lives here
export MPLCONFIGDIR="${MPLCONFIGDIR:-$IE_SCRATCH/.cache/matplotlib}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" 2>/dev/null || true

# ---- one core per task: stop numpy/BLAS/OpenMP from oversubscribing a shared node ----------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" \
       MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

# optional environment modules (non-interactive shells do not load the `module` function by themselves)
ie_module() { [ $# -gt 0 ] || return 0; { type module >/dev/null 2>&1 || . /etc/profile.d/modules.sh 2>/dev/null; } ; module load "$@"; }
