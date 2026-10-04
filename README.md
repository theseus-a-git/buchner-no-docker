# inference-envs

One Conda environment per inference problem from Johannes Buchner's *space-of-inference-spaces* benchmark suite. The framework keeps each scientific likelihood in its native package while exposing a common sampler interface where possible.

Core comparison backends:

```text
UltraNest   nested sampling
Dynesty     nested sampling
Nestle      nested sampling
emcee        ensemble MCMC
PSO         particle-swarm optimization
```

The supplied PSO source under `common/PSO-master/` is vendored **unchanged**. Only an external adapter calls it.

```text
inference-envs/
├── setup.sh
├── run.sh
├── benchmark.sh
├── compare.sh
├── common/
│   ├── autosampler.py
│   ├── emcee_backend.py
│   ├── pso_backend.py
│   └── PSO-master/          # unchanged upstream PSO source
├── real/
└── synthetic/
```

## Run the benchmark

With no sampler names, `run.sh` uses the five core backends automatically:

```bash
./setup.sh
./run.sh real
./run.sh synthetic
```

Or run one problem:

```bash
./run.sh real/powerlaw-relation
./run.sh real/powerlaw-relation ultranest dynesty nestle emcee pso
```

Set a common likelihood-evaluation budget with `MAX_NCALLS` (default `100000`):

```bash
MAX_NCALLS=100000 ./run.sh real/powerlaw-relation
```

To rerun completed cases, use `--force`. To select only certain cases, use `ONLY`, for example:

```bash
ONLY="eggbox-2d beta-2d" ./run.sh synthetic/toy
```

## Compare samplers

After runs finish:

```bash
./compare.sh
```

This compares `UltraNest`, `Dynesty`, `Nestle`, `emcee` and `PSO` by default and writes `sampler_comparison.md`, `.csv` and `.png`. Use `--problems` or `--samplers` to restrict the report:

```bash
./compare.sh --problems real/ligo real/icecube
./compare.sh --samplers ultranest dynesty nestle emcee pso --ref ultranest
```

Nested samplers report evidence (`logZ`, `logZerr`) and posterior-sample agreement. emcee reports posterior samples and MCMC diagnostics but no evidence. PSO reports best-fit/maximum-likelihood information and optimization history but no posterior samples or evidence. Missing requested runs appear explicitly in the report instead of silently disappearing.

For a one-command run+compare workflow:

```bash
./benchmark.sh real
./benchmark.sh synthetic
```

`PSO_N_RUNS`, `PSO_N_PARTICLES`, `PSO_N_ITERATIONS` and the other `PSO_*` environment variables tune PSO without modifying the vendored code. By default the adapter converts `MAX_NCALLS` into a PSO iteration count unless `PSO_N_ITERATIONS` is explicitly set.

### Budget, timing and plots

- `MAX_NCALLS` is passed to every backend, but each stops by its own rule (nested samplers finish the iteration in progress; emcee uses `nwalkers x nsteps`; PSO converts it to an iteration count). To make them comparable anyway, `autosampler.py` counts every likelihood evaluation itself and writes it as `ncall_counted` (plus `max_ncalls_requested`) to `results.json`; `compare_samplers.py` reports `ncall_counted`. Check it against the budget rather than assuming it is equal. Counting is per process, so PSO with `PSO_*_PARALLEL_MODE` other than `series`, and resumed UltraNest runs, undercount.
- `walltime_s` is the sampling time only (set right after the sampler returns, before plots and post-processing) for all five backends. UltraNest's per-iteration region/sample logging is included in its time.
- Corner/diagnostic plots are skipped above `PLOT_MAX_DIM` dimensions (default 30), and a plotting error no longer fails a finished run.
- Nestle (all environments except `cosmology`, which pins NumPy 1.23.5) uses NumPy aliases removed in NumPy >= 1.24 (e.g. `np.int` in `resample_equal`). `autosampler.py` restores them in the sampler process only, so the shared NumPy stays unpinned.
- Core backends: `ultranest dynesty nestle emcee pso`. `run.sh` still accepts the other `autosampler` modes (`multinest`, `vbis`, `vegas`, `ultranest-fast`, ...) but prints that they are legacy/optional.

## Running on a Slurm cluster (PARAM Seva)

`slurm/` submits all 73 cases x 5 samplers as Slurm job arrays (one single-core task per run, per-problem time and
memory in `slurm/resources.tsv`, serial arrays for problems that share work files, a final comparison job).
Start with `slurm/README.md`: `slurm/check.sh`, `slurm/setup_login.sh`, `slurm/prefetch.sh`, `slurm/submit.sh`, `slurm/status.sh`.

## Per-problem notes (everything below was needed on a current setup)
| Problem | Notes |
|---|---|
| bixrayspectrum, ligo, icecube | plain `./setup.sh` + `./run.sh`. icecube (pisa) is the one most likely to break on a new numpy: add `numpy<2` to its `environment.yml`. |
| lj6 | `LJ6.py` takes the particle count (`6` in `problem.conf`); `region_class` moved from the constructor to `sampler.run()` in current UltraNest (already patched). Output in `LJ6/`. |
| exoplanet | the repo's `juliet/` fork reads `$SAMPLER` and calls `autosampler`, so any sampler name works. Run **without** `-O` (an unknown name used to exit silently). Data in `data/`. Cases `exo-transient`, `rvs-0`, `rvs-1`, `rvs-2`, `rvs-3` = transit plus 0/1/2/3-planet RV fits. The transit case uses the HATS-46 TESS Sector 2 tutorial data and downloads the public FITS light curve on first run. The `exo-transient` script is a reconstruction from the juliet HATS-46b tutorial (the original repo's script is missing), so it may differ from Buchner's version. |
| mosfit | MOSFiT 2 no longer downloads events: `setup.sh` copies `LSQ12dlf.json` out of the installed package (`post_setup.sh`). Cases `SLSN-LSQ12dlf`, `Magnetar-LSQ12dlf`. |
| crab | env needs fermitools (conda) + fermipy installed with `pip --no-deps` (plain pip replaces conda's numpy and breaks imports). Downloads Fermi-LAT data from NASA on first run (needs internet, slow). `chain_name='chains/crab'` (threeML crashes on a name with no folder). Leaves `__<hex>` fermipy folders; `clean.sh` removes them. |
| grb | runs in the crab env. First run downloads GBM data. If it dies in the catalog step with `no element found`, delete the half-written cache: `rm ~/.threeML/.cache/fermigbrst_votable.xml` and rerun. `apply_patch.sh` is an optional old threeML patch, not needed. |
| xray-agn-ciao | `./setup.sh real/xray-agn-ciao` builds the CIAO env and clones BXA; then `./run.sh real/xray-agn-ciao` (the first run downloads the 1.1 GB UXCLUMPY tables from Zenodo, md5-checked and resumable; `bash real/xray-agn-ciao/fetch_models.sh` does it separately). `run_agn.sh` applies two fixes for Sherpa 4.18 (astropy FITS reader via `~/.sherpa.rc`, and `load_xstable_model` in `xagnfitter.py`). DS9/Qt warnings are harmless. Needs a few GB of RAM. |
| posteriorstacker | clones PosteriorStacker (AGPLv3); the benchmark problem is its tutorial: `gendata.py` makes demo posteriors, then Gaussian + 11-bin histogram models are fitted. Results in `systematiclogs/posteriorstacker/`. |
| powerlaw-relation | 3-parameter line fit (Tully-Fisher style) to 42 bulge-mass/velocity-dispersion measurements, with the measurement errors integrated through deterministic 400-sample clouds, following the UltraNest line-fitting tutorial. This is a reconstruction: the original repo has no script for it, and the values in `data/tully_fisher.csv` were typed in, not downloaded; check them against the source table before relying on results. |
| cosmology | CLASS 3.2.0 is compiled and Buchner's MontePython fork cloned by `setup.sh` (needs `sudo apt install -y build-essential gfortran`). Old pins (python 3.9, numpy 1.23.5) are isolated in its own env. Runs `input/example_ns.param` with `-m UN`; the fork calls `autosampler`, so the five core samplers can be selected. Fixes already in `post_setup.sh`: CLASS 3.2.0 needs `-std=gnu17` with GCC >= 15; CLASS is cloned from the `v3.2.0` GitHub tag (the old tarball URL is dead); only the C targets are built with make (the wrapper is built into the conda env); `patch_montepython.py` defines `output = None` in the fork's `UltraNest.py`. The MontePython fork now uses the common sampler dispatcher; the startup message about UltraNest detection can be ignored when running non-UltraNest backends. Real Planck likelihood data is NOT downloaded (the example uses the built-in fake Planck likelihood). |
| synthetic | `./setup.sh synthetic`, then `./run.sh synthetic` to run every synthetic case with the five core backends. `problems.py` has the `gammaln` speed-up patched for current scipy. |

## Comparing samplers
Run the same problem with multiple inference methods:
```bash
./run.sh real/powerlaw-relation ultranest dynesty emcee pso
./compare.sh --problems real/powerlaw-relation --samplers ultranest dynesty emcee pso --ref ultranest
```
`compare_samplers.py` writes `sampler_comparison.md/.csv/.png`. Nested samplers report evidence (`logZ`, `logZerr`) and posterior agreement against the reference; emcee reports posterior agreement and run cost but no evidence; PSO reports the best log-likelihood/MLE point and run cost but does not produce posterior samples or evidence. The report therefore keeps sampler-specific quantities separate rather than treating PSO as a posterior sampler.
PSO settings can be overridden without touching its source, for example `PSO_N_RUNS=6 PSO_N_PARTICLES=40 PSO_N_ITERATIONS=1000 ./run.sh real/powerlaw-relation pso`. Native emcee uses `emcee==3.1.6`; its `EMCEE_NWALKERS`, `EMCEE_NSTEPS`, `EMCEE_BURN` and `EMCEE_SEED` settings can likewise be overridden.

## Housekeeping
- Putting this on GitHub: `.gitignore` already excludes outputs, downloads, the BXA clone and the 1.1 GB model tables.
- `bash clean.sh` lists what it would delete (runs, results, downloaded data, caches, backups); `bash clean.sh --yes` deletes. Archive results first if you want them.
- Remove an env: `conda env remove -n inf-ligo`.
- Add a problem: copy a directory; edit `environment.yml`, `problem.conf` (`ENV`, `CMD` with `${CASE}`, optional `CASES`, `FIXED_SAMPLER`) and `smoke.py`.
- Not covered: the real Planck clik likelihood data. Reconstructed (not from the original repo): `exo-transient` and `powerlaw-relation`.
