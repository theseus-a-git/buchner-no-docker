"""Adapter for the vendored PSO project.

The files under common/PSO-master/ are the upstream PSO project and are not
modified. This module only adapts the existing inference-envs likelihood
interface to the PSO ``run_PSO`` entry point.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _load_run_pso():
    pso_root = Path(__file__).resolve().parent / "PSO-master"
    if not pso_root.is_dir():
        raise RuntimeError(f"vendored PSO directory not found: {pso_root}")
    pso_root_str = str(pso_root)
    if pso_root_str not in sys.path:
        # PSO-master deliberately remains an unchanged standalone source tree.
        # Its internal imports are resolved by putting its root on sys.path.
        sys.path.insert(0, pso_root_str)
    from PSO_main import run_PSO  # type: ignore
    return run_PSO


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def run_pso(
    param_names,
    loglike,
    transform,
    log_dir,
    max_ncalls=400000000,
):
    """Run the unchanged PSO implementation on the common unit-cube API.

    PSO minimizes a fitness function. We therefore minimize ``-loglike`` on
    the unit cube. Because the existing ``transform`` maps the unit cube to
    the physical prior space, this preserves the prior support used by the
    other samplers without changing any problem-specific code.
    """
    ndim = len(param_names)
    if ndim < 1:
        raise ValueError("PSO requires at least one parameter")

    n_runs = _int_env("PSO_N_RUNS", 4)
    n_particles = _int_env("PSO_N_PARTICLES", 40)
    n_iterations = _int_env("PSO_N_ITERATIONS", 500)

    # Optional budget override. The explicit PSO_N_ITERATIONS setting wins.
    if "PSO_N_ITERATIONS" not in os.environ and _bool_env("PSO_USE_MAX_NCALLS", True):
        per_iteration = n_runs * n_particles
        n_iterations = max(1, int(max_ncalls // per_iteration) - 1)

    run_parallel_mode = os.environ.get("PSO_RUN_PARALLEL_MODE", "series")
    fitness_parallel_mode = os.environ.get("PSO_FITNESS_PARALLEL_MODE", "series")
    run_n_jobs = os.environ.get("PSO_RUN_N_JOBS")
    fitness_n_jobs = os.environ.get("PSO_FITNESS_N_JOBS")

    kwargs = dict(
        N_particles=n_particles,
        PSO_type=os.environ.get("PSO_TYPE", "lbest"),
        boundary_type=os.environ.get("PSO_BOUNDARY_TYPE", "reflecting_wall"),
        eq_type=os.environ.get("PSO_EQ_TYPE", "inertia"),
        C1=_float_env("PSO_C1", 2.0),
        C2=_float_env("PSO_C2", 2.0),
        vmax=_float_env("PSO_VMAX", 0.5),
        seed_vals=_bool_env("PSO_SEED", True),
        run_parallel_mode=run_parallel_mode,
        fitness_parallel_mode=fitness_parallel_mode,
        verbose=_bool_env("PSO_VERBOSE", True),
        save_history=_bool_env("PSO_SAVE_HISTORY", False),
    )
    if run_n_jobs is not None:
        kwargs["run_n_jobs"] = int(run_n_jobs)
    if fitness_n_jobs is not None:
        kwargs["fitness_n_jobs"] = int(fitness_n_jobs)

    finite_penalty = np.finfo(np.float64).max

    def fitness(unit_point):
        u = np.asarray(unit_point, dtype=np.float64)
        if u.shape != (ndim,) or not np.all(np.isfinite(u)):
            return finite_penalty
        if np.any(u < 0.0) or np.any(u > 1.0):
            return finite_penalty
        try:
            theta = np.asarray(transform(u), dtype=np.float64)
            value = float(loglike(theta))
        except Exception:
            return finite_penalty
        if not np.isfinite(value):
            return finite_penalty
        return -value

    bounds = np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (ndim, 1))

    started = time.time()
    run_PSO = _load_run_pso()
    (
        fitness_vals,
        best_unit_locations,
        particle_history,
        fitness_history,
        pbest_history,
    ) = run_PSO(
        n_runs,
        n_iterations,
        bounds,
        fitness,
        **kwargs,
    )
    walltime = time.time() - started

    fitness_vals = np.asarray(fitness_vals, dtype=np.float64)
    best_unit_locations = np.asarray(best_unit_locations, dtype=np.float64)
    best_idx = int(np.argmin(fitness_vals))
    best_unit = best_unit_locations[best_idx]
    best_parameters = np.asarray(transform(best_unit), dtype=np.float64)
    best_loglike = float(-fitness_vals[best_idx])
    ncall = int(n_runs * (n_iterations + 1) * n_particles)

    os.makedirs(log_dir, exist_ok=True)
    result = {
        "sampler": "pso",
        "sampler_kind": "optimizer",
        "objective": "maximum_likelihood",
        "best_loglike": best_loglike,
        "best_parameters": best_parameters.tolist(),
        "best_unit": best_unit.tolist(),
        "paramnames": list(param_names),
        "ncall": ncall,
        "walltime_s": walltime,
        "n_runs": n_runs,
        "n_particles": n_particles,
        "n_iterations": n_iterations,
        "settings": {
            k: v for k, v in kwargs.items()
            if k not in {"position_init_premade_vals", "velocity_init_premade_vals", "update_premade_vals"}
        },
    }
    with open(Path(log_dir) / "results.json", "w") as fout:
        json.dump(result, fout, indent=4)

    np.savetxt(
        Path(log_dir) / "best_parameters.txt",
        best_parameters[None, :],
        delimiter=",",
        header=",".join(param_names),
        comments="",
    )
    np.savetxt(
        Path(log_dir) / "run_best_fitness.txt",
        fitness_vals,
        delimiter=",",
        header="best_fitness",
        comments="",
    )

    if _bool_env("PSO_SAVE_HISTORY", False):
        np.savez_compressed(
            Path(log_dir) / "history.npz",
            particle_history=particle_history,
            fitness_history=fitness_history,
            pbest_history=pbest_history,
        )

    return result
