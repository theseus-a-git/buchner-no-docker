"""HATS-46b TESS transit benchmark from the juliet tutorial.

This is the exo-transient problem in Buchner's space-of-inference-spaces:
9 free parameters, with eccentricity, omega and dilution fixed.

The public Sector 2 TESS light curve is downloaded on first run and cached
under data/ so the repository itself remains small.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from pathlib import Path

import juliet


TESS_URL = (
    "https://archive.stsci.edu/hlsps/tess-data-alerts/"
    "hlsp_tess-data-alerts_tess_phot_00281541555-s02_tess_v1_lc.fits"
)
DEFAULT_DATA = Path("data/hats46_tess_sector02.fits")


PARAMS = [
    "P_p1", "t0_p1", "r1_p1", "r2_p1", "q1_TESS", "q2_TESS",
    "ecc_p1", "omega_p1", "rho", "mdilution_TESS", "mflux_TESS",
    "sigma_w_TESS",
]
DISTS = [
    "normal", "normal", "uniform", "uniform", "uniform", "uniform",
    "fixed", "fixed", "loguniform", "fixed", "normal", "loguniform",
]
HYPERPS = [
    [4.7, 0.1], [1358.4, 0.1], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
    0.0, 90.0, [100.0, 10000.0], 1.0, [0.0, 0.1], [0.1, 1000.0],
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "inference-envs/exo-transient (academic benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, open(tmp_path, "wb") as fout:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcfile", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--url", default=TESS_URL)
    parser.add_argument("--num-live-points", type=int, default=400)
    args = parser.parse_args()

    if not args.lcfile.exists():
        print(f"downloading HATS-46 TESS Sector 2 light curve to {args.lcfile}")
        download(args.url, args.lcfile)

    t, f, ferr = juliet.get_TESS_data(str(args.lcfile))
    times = {"TESS": t}
    fluxes = {"TESS": f}
    fluxes_error = {"TESS": ferr}

    priors = juliet.utils.generate_priors(PARAMS, DISTS, HYPERPS)

    case = os.environ.get("PROBLEM", "exo-transient")
    log_dir = Path("systematiclogs") / case / os.environ["SAMPLER"]
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOGDIR"] = str(log_dir)

    dataset = juliet.load(
        priors=priors,
        t_lc=times,
        y_lc=fluxes,
        yerr_lc=fluxes_error,
        out_folder=str(log_dir) + "/",
    )

    # The local juliet fork detects SAMPLER and delegates to common/autosampler.py.
    dataset.fit(use_ultranest=True, n_live_points=args.num_live_points)


if __name__ == "__main__":
    main()
