import os
import sys
from pathlib import Path
import numpy as np

num_particles = int(sys.argv[1]) if len(sys.argv) > 1 else 6
paramnames = ["z2"]
for i in range(3, num_particles + 1):
    if i == 3:
        paramnames += [f"y{i}", f"z{i}"]
    else:
        paramnames += [f"x{i}", f"y{i}", f"z{i}"]

sigma = 1e-3
pos0 = np.zeros(3)
pos1 = np.zeros(2)
pos2 = np.zeros(1)

def loglikelihood(param):
    if len(param) > 1:
        coordinates = np.hstack((pos0, pos1, param[:1], pos2, param[1:])).reshape((-1, 3))
    else:
        coordinates = np.hstack((pos0, pos1, param[:1])).reshape((-1, 3))
    unordered = np.diff(np.abs(coordinates[:, 2])) < 0
    if np.any(unordered):
        return float(-1e200 * np.max(-np.diff(np.abs(coordinates[:, 2]))))
    logL = 0.0
    for i, a in enumerate(coordinates):
        b = coordinates[i+1:, :]
        r = ((a.reshape((1,3)) - b)**2).sum(axis=1)**0.5
        r[r < sigma] = sigma
        sigma_r6 = (sigma / r)**6
        logL += np.log(sigma_r6 - sigma_r6**2 + 1e-100).sum()
    return float(logL)

def prior(cube):
    cube = np.asarray(cube, dtype=float)
    param = 2.0 * cube - 1.0
    param[:2] = cube[:2]
    if len(cube) > 2:
        param[3] = cube[3]
    return param

from autosampler import run_sampler
case = os.environ.get("PROBLEM", "lj6")
sampler = os.environ.get("SAMPLER", "ultranest")
log_dir = Path("systematiclogs") / case / sampler
log_dir.mkdir(parents=True, exist_ok=True)
os.environ["LOGDIR"] = str(log_dir)
run_sampler(paramnames, loglikelihood, transform=prior)
