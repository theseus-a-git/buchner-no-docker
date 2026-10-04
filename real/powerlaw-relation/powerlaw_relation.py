"""Power-law line fit (Tully-Fisher relation) benchmark.

Implements the 3-parameter real inference problem described in
Buchner's space-of-inference-spaces and in the UltraNest "fitting a line"
tutorial. The 42 Kormendy & Ho measurements have asymmetric/heteroscedastic
measurement uncertainties represented by Monte-Carlo clouds in (log M_bulge,
log sigma). The intrinsic scatter is integrated by averaging the Gaussian
likelihood over those clouds.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import scipy.stats

from autosampler import run_sampler


PARAM_NAMES = ["slope", "offset", "scatter"]
N_MC = 400
RNG_SEED = 0
DATA_FILE = Path("data/tully_fisher.csv")


def load_data(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 42:
        raise RuntimeError(f"expected 42 data points, found {len(rows)}")
    mB = np.array([float(r["mB"]) for r in rows], dtype=float)
    mBerr = np.array([float(r["mBerr"]) for r in rows], dtype=float)
    sigma = np.array([float(r["sigma"]) for r in rows], dtype=float)
    sigmaerr = np.array([float(r["sigmaerr"]) for r in rows], dtype=float)
    return mB, mBerr, sigma, sigmaerr


def make_samples(mB, mBerr, sigma, sigmaerr):
    rng = np.random.RandomState(RNG_SEED)
    samples = []
    for i in range(len(mB)):
        samples_mBi = rng.normal(mB[i], mBerr[i], size=N_MC)
        samples_sigmai = rng.normal(sigma[i], sigmaerr[i], size=N_MC)
        samples_logsigmai = np.log10(samples_sigmai)
        samples.append([samples_mBi, samples_logsigmai])
    return np.asarray(samples)


def prior_transform(cube):
    params = np.asarray(cube, dtype=float).copy()

    # slope ~ Uniform(-3, 3)
    params[0] = cube[0] * 6.0 - 3.0

    # offset ~ LogUniform(10, 1000); work in log10(km/s)
    params[1] = cube[1] * 2.0 + 1.0

    # scatter ~ LogUniform(0.001, 10) dex
    params[2] = 10.0 ** (cube[2] * 4.0 - 3.0)
    return params


def main() -> None:
    mB, mBerr, sigma, sigmaerr = load_data(DATA_FILE)
    samples = make_samples(mB, mBerr, sigma, sigmaerr)

    def log_likelihood(params):
        slope, offset, scatter = params
        y_expected = (samples[:, 0] - 10.0) * slope + offset
        probs_samples = scipy.stats.norm(y_expected, scatter).pdf(samples[:, 1])
        probs_objects = probs_samples.mean(axis=1)
        loglike = np.log(probs_objects + 1e-100).sum()
        return float(loglike)

    log_dir = Path("systematiclogs") / os.environ.get("PROBLEM", "powerlaw-relation") / os.environ["SAMPLER"]
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOGDIR"] = str(log_dir)

    run_sampler(
        PARAM_NAMES,
        log_likelihood,
        transform=prior_transform,
    )


if __name__ == "__main__":
    main()
