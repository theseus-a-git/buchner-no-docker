"""Native emcee adapter for the common inference interface."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def run_emcee(
    param_names,
    loglike,
    transform,
    log_dir,
    max_ncalls=400000000,
):
    """Run native emcee in the unit cube and save physical posterior samples."""
    import emcee

    ndim = len(param_names)
    nwalkers = _int_env("EMCEE_NWALKERS", max(32, 2 * ndim + 4), minimum=2)
    nsteps = _int_env("EMCEE_NSTEPS", max(200, min(5000, max_ncalls // nwalkers)), minimum=2)
    burn = _int_env("EMCEE_BURN", min(500, max(20, nsteps // 5)), minimum=0)
    burn = min(burn, nsteps - 1)
    seed = _int_env("EMCEE_SEED", 42, minimum=0)

    if nwalkers <= 2 * ndim:
        raise ValueError(
            f"EMCEE_NWALKERS must be > 2*ndim for the default stretch move; "
            f"got {nwalkers} for ndim={ndim}"
        )

    rng = np.random.default_rng(seed)
    # The posterior in unit-cube coordinates has density proportional to L,
    # because the prior is uniform in the cube. Points outside the cube are
    # rejected rather than transformed.
    def log_prob(u):
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (ndim,) or not np.all(np.isfinite(u)):
            return -np.inf
        if np.any(u <= 0.0) or np.any(u >= 1.0):
            return -np.inf
        try:
            theta = transform(u)
            value = float(loglike(theta))
        except Exception:
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    initial = 0.5 + 0.05 * rng.normal(size=(nwalkers, ndim))
    initial = np.clip(initial, 1.0e-6, 1.0 - 1.0e-6)

    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    started = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    walltime = time.time() - started

    unit_samples = sampler.get_chain(discard=burn, thin=1, flat=True)
    if unit_samples.size == 0:
        raise RuntimeError("emcee produced no post-burn-in samples")

    physical_samples = np.asarray(
        [np.asarray(transform(u), dtype=np.float64) for u in unit_samples],
        dtype=np.float64,
    )
    all_logp = sampler.get_log_prob(flat=True)
    finite_logp = all_logp[np.isfinite(all_logp)]
    best_loglike = float(np.max(finite_logp)) if finite_logp.size else float("nan")

    try:
        tau = sampler.get_autocorr_time(discard=burn)
        autocorr = tau.tolist()
    except Exception:
        autocorr = None

    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
    ncall = int(nwalkers * nsteps)

    os.makedirs(log_dir, exist_ok=True)
    result = {
        "sampler": "emcee",
        "sampler_kind": "mcmc",
        "objective": "posterior_sampling",
        "best_loglike": best_loglike,
        "paramnames": list(param_names),
        "ncall": ncall,
        "walltime_s": walltime,
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "burn": burn,
        "acceptance_fraction": acceptance_fraction,
        "autocorr_time": autocorr,
    }
    with open(Path(log_dir) / "results.json", "w") as fout:
        json.dump(result, fout, indent=4)

    np.savetxt(
        Path(log_dir) / "samples.txt.gz",
        physical_samples,
        delimiter=",",
        header=",".join(param_names),
    )
    return result
