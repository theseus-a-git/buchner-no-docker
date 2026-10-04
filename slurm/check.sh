#!/usr/bin/env bash
# Read-only preflight on the login node.   slurm/check.sh          what the scheduler / storage / conda look like
#                                          slurm/check.sh --net    also run a 1-minute test job to see if COMPUTE nodes have internet
. "$(dirname "$0")/env.sh"
cd "$INFENV_ROOT" || exit 1
echo "== code tree: $INFENV_ROOT"
case "$INFENV_ROOT" in
  /scratch/*) echo "   on /scratch: good (runs/ and systematiclogs/ can get large; scratch is purged after 3 months without access)";;
  /home/*)    echo "   !! on /home (50 GB quota). Prefer /scratch/$USER for the code tree: all results are written inside it";;
esac
echo "== conda ($CONDA_ROOT)"
if command -v conda >/dev/null 2>&1; then
  echo "   $(conda --version)"
  envs=$(conda env list 2>/dev/null | awk '{print $1}')
  for y in $(find real synthetic -name environment.yml | sort); do
    n=$(awk '/^name:/{print $2}' "$y"); echo "$envs" | grep -qx "$n" && echo "   ok       $n" || echo "   MISSING  $n   ($(dirname "$y"))"
  done
else echo "   conda not found -> slurm/setup_login.sh"; fi
echo "== Slurm"
sinfo -o '   %-10P avail=%a  timelimit=%l  nodes=%D' 2>&1 | head -8
scontrol show config 2>/dev/null | grep -E "^(MaxArraySize|MaxJobCount|SelectType |SelectTypeParameters|DefMemPerCPU|MaxMemPerNode)" | sed 's/^/   /'
echo "   limits (QoS):"; sacctmgr -n -P show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU 2>&1 | head -6 | sed 's/^/     /'
echo "   your account(s): $(sacctmgr -n -P show assoc user=$USER format=Account 2>/dev/null | sort -u | tr '\n' ' ')   (set IE_ACCOUNT if needed)"
echo "== storage"
df -h "$IE_SCRATCH" "$HOME" 2>/dev/null | sed 's/^/   /'
command -v lfs >/dev/null && lfs quota -h -u "$USER" /scratch 2>/dev/null | sed 's/^/   /'
echo "== internet from the login node"
curl -sI -m 8 https://conda.anaconda.org >/dev/null 2>&1 && echo "   yes (conda.anaconda.org)" || echo "   NO: setup.sh / prefetch.sh need it. Ask the admins about a proxy, or build the envs elsewhere"
if [ "${1:-}" = --net ]; then
  echo "== internet from a compute node (srun, 2 min)"
  srun ${IE_ACCOUNT:+--account=$IE_ACCOUNT} -p "$IE_PARTITION" -N1 -n1 -t 00:02:00 --mem=500M bash -c \
    'echo "   node: $(hostname)"; for u in https://archive.stsci.edu https://heasarc.gsfc.nasa.gov https://gwosc.org https://fermi.gsfc.nasa.gov; do curl -sI -m 8 $u >/dev/null 2>&1 && echo "   reachable:   $u" || echo "   unreachable: $u"; done'
  echo "   if these are unreachable, real/crab, real/grb and real/ligo cannot run on compute nodes: use  slurm/submit.sh --no-net"
fi
