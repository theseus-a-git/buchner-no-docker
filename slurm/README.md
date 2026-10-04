# Running the whole benchmark on PARAM Seva (Slurm)

Every (problem, case, sampler) run becomes one single-core Slurm array task: **73 cases x 5 samplers
(ultranest, dynesty, nestle, emcee, pso) = 365 tasks**. Nothing in the samplers or in `common/PSO-master` was changed.
`slurm/submit.sh` groups the tasks into 13 arrays (one per `resources.tsv` row), each with its own time and memory
request, and queues a final job that builds the comparison tables.

Based on the PARAM Seva user manual: Slurm 20.11, partitions `standard` (default) / `cpu` / `gpu` / `hm`, 4-day maximum,
48-core 192 GB CPU nodes (shared between jobs), 2 h default walltime, `$HOME` 50 GB, `/scratch` 200 GB.

## Quick start (all on the login node)

```bash
# 0. copy the tree to /scratch (results can get large; scratch files unused for 3 months are deleted)
scp -P 4422 inference-envs-five-samplers-*.zip <user>@paramseva.iith.ac.in:/scratch/<user>/
ssh -p 4422 <user>@paramseva.iith.ac.in
cd /scratch/$USER && unzip inference-envs-*.zip && cd inference-envs

slurm/check.sh                     # partitions, limits, quota, conda, internet from the login node
export IE_ACCOUNT=<your-account>   # only if sbatch asks for one; put it in ~/.bashrc
slurm/setup_login.sh               # Miniforge + all 12 conda envs (hours: run inside tmux/screen or nohup)
slurm/prefetch.sh                  # downloads TESS / LIGO / UXCLUMPY data while you still have internet
slurm/check.sh --net               # 2-minute test job: do COMPUTE nodes have internet? (decides --no-net below)

# smoke test: 2 cheap runs, then look at the result
slurm/submit.sh --problems synthetic/toy --cases '^asymgauss-4d$' --no-compare
slurm/status.sh

# everything (use --no-net if the compute nodes have no internet)
slurm/submit.sh --dry-run          # shows the plan and the worst-case core-hours; nothing is submitted
slurm/submit.sh [--no-net]
slurm/status.sh                    # progress matrix from the .done markers
slurm/status.sh --sacct            # Slurm state / elapsed / peak memory; lists TIMEOUT and OOM tasks
```

Re-running `slurm/submit.sh` only submits runs that are not `.done`, so it is also the "retry" command:
`slurm/submit.sh --time-mult 2` (after timeouts) or `--mem-mult 2` (after out-of-memory). It refuses to submit while
`ie-*` jobs are still queued (use `--allow-active` to override). Other options: `--problems`, `--samplers`, `--cases REGEX`,
`--only-net`, `--max-ncalls N`, `--partition P`, `--throttle N`, `--all` (force rerun), `--no-compare`.

The comparison job writes the same tables as `./compare.sh`; finished data stays in `<problem>/runs` and
`<problem>/systematiclogs`. Copy it back with `rsync -av` (or scp -P 4422) when done.

## What is where

| file | purpose |
|---|---|
| `env.sh` | all defaults (`IE_*` variables), conda + cache setup, one thread per task. **Read this first.** |
| `resources.tsv` | per-problem time / memory / partition / serial (mutex) / needs-internet / `max_ncalls` |
| `submit.sh` (`submit.py`) | build the task lists, group them, call `sbatch`, write `slurm/jobs/<stamp>/ledger.tsv` |
| `task.sbatch` | what each array task runs: `ONLY=<case> ./run.sh <dir> <sampler>` |
| `status.sh` (`status.py`) | progress and `sacct` summary |
| `setup_login.sh`, `install_miniforge.sh`, `prefetch.sh`, `check.sh` | one-time setup and preflight |
| `make_manifest.sh` | prints the 365 `dir / case / sampler` lines (same case expansion as `run.sh`) |

Job logs: `slurm/logs/<jobname>_<jobid>_<task>.out`; per-run logs as before: `<problem>/runs/<case>/<sampler>/log.txt`.

## Design decisions you may want to change

* **One core per task, many tasks in parallel** (default `--throttle 48`). The manual says nodes are shared, so single-core
  tasks should not waste a node. If your allocation is charged per whole node, ask the admins before submitting 365 of them.
* **Serial groups.** `real/crab`, `real/grb`, `real/xray-agn-ciao`, `real/posteriorstacker` keep working files in their own
  directory (fermipy output, the BXA `chandra/` folder, `posteriorsamples.txt`, a first-run data download). Five samplers
  running at once would overwrite each other, so these are arrays with `%1` (one at a time), like `run.sh` did. Other problems
  write only to per-sampler folders and run in parallel. Two plot filenames that were shared (`bixrayspectrum-*.pdf`,
  `xrayspectrum.pdf`) now include the sampler name.
* **Partition `cpu`.** `standard` also contains the GPU nodes. The manual text lists three partitions but its `sinfo`
  screenshot shows `cpu` too; if `cpu` is rejected, `export IE_PARTITION=standard`.
* **Equal budgets.** All samplers get `MAX_NCALLS=100000` unless `resources.tsv` says otherwise (set `max_ncalls`
  there for slow likelihoods). See the main README ("Budget, timing and plots"): `ncall_counted` is the fair number.
* `run.sh` now exits non-zero when a run fails (it used to exit 0), otherwise Slurm would report failures as COMPLETED.

## Internet on compute nodes

Unknown for PARAM Seva (the manual does not say), so everything that can be fetched in advance is (`prefetch.sh`).
Three problems cannot be: `real/crab` and `real/grb` query the Fermi / HEASARC servers while running, and `real/ligo`
queries the GWOSC event catalogue (this one is my reading of the pycbc API, not tested). They are marked `net=1`. If
`slurm/check.sh --net` says the servers are unreachable from compute nodes, submit with `--no-net`: that skips these
15 of 365 runs. Running them on the login node is not allowed by the manual; ask the C-DAC/IITH support
(sevasupport@iith.ac.in) about a proxy or about running them on a node with outbound access.

## Resource numbers are guesses

Wall times in `resources.tsv` come from reading the code, not from measurements. Expect `real/cosmology` (CLASS, seconds per
call), `real/crab`, `real/grb` and probably `real/xray-agn-ciao` to be the problems where 100000 calls do not fit the
4-day limit; lower their `max_ncalls`, or accept that they time out. Start with the cheap problems, read
`slurm/status.sh --sacct` ("what finished runs actually used") and edit `resources.tsv`.

## Not verified

This was written from the manual without access to the cluster. Tested with fake `sbatch`/`conda` on a laptop: task
enumeration (365), grouping, dry run, submission arguments, ledger, skip-done, failure exit codes, `status`/`sacct` parsing,
and an end-to-end run of one synthetic task through `task.sbatch` -> `run.sh` -> `autosampler.py` with a stub sampler.
Not tested: real Slurm, the real conda environments, the real samplers, CLASS compilation, the data downloads.
First things to check on the cluster: `slurm/check.sh`, the partition name, `--account`, and that a smoke-test task
finishes (`slurm/logs/*.out`).

## Troubleshooting

* `sbatch: error: invalid account/partition` -> `IE_ACCOUNT` / `IE_PARTITION`; `sinfo`, `sacctmgr show assoc user=$USER`.
* `conda: command not found` inside a job -> `CONDA_ROOT` in `env.sh` does not point at your Miniforge.
* Task shows COMPLETED but nothing in `runs/` -> look at `slurm/logs/` first, then `runs/<case>/<sampler>/log.txt`.
* Quota: `du -sh $CONDA_ROOT runs`; the conda environments and caches are the big items; keep them on `/scratch`.
* CLASS (`real/cosmology`) does not compile on the login node -> `module avail gcc`, then
  `IE_SETUP_MODULES="<gcc module>" slurm/setup_login.sh real/cosmology`.
