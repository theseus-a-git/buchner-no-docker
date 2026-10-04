"""Expose a threeML model and DataList through inference-envs samplers."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

def _values(obj):
    if obj is None:
        return []
    vals = getattr(obj, "values", None)
    if callable(vals):
        return list(vals())
    return list(obj)

def make_3ml_callbacks(model, data_list):
    free = _values(getattr(model, "free_parameters", None))
    if not free:
        raise RuntimeError("threeML model exposes no free parameters")
    names = [str(getattr(p, "path", None) or getattr(p, "fullname", None) or getattr(p, "name", None) or p) for p in free]
    plugins = _values(data_list)
    if not plugins:
        raise RuntimeError("threeML DataList exposes no plugins")

    def transform(cube):
        u = np.asarray(cube, dtype=float)
        if u.shape != (len(free),):
            raise ValueError(f"expected {(len(free),)}, got {u.shape}")
        out = np.empty(len(free), dtype=float)
        for i, p in enumerate(free):
            prior = getattr(p, "prior", None)
            if prior is None or not hasattr(prior, "from_unit_cube"):
                raise RuntimeError(f"prior for {names[i]} has no from_unit_cube()")
            out[i] = float(prior.from_unit_cube(float(np.clip(u[i], 1e-12, 1-1e-12))))
        return out

    def loglike(theta):
        x = np.asarray(theta, dtype=float)
        if x.shape != (len(free),):
            raise ValueError(f"expected {(len(free),)}, got {x.shape}")
        for p, v in zip(free, x):
            p.value = float(v)
        return float(sum(float(plugin.get_log_like()) for plugin in plugins))

    return names, loglike, transform

def run_3ml(model, data_list, case_name=None, max_ncalls=None):
    from autosampler import run_sampler
    names, loglike, transform = make_3ml_callbacks(model, data_list)
    sampler = os.environ.get("SAMPLER", "ultranest")
    case = case_name or os.environ.get("PROBLEM", "threeML")
    log_dir = Path("systematiclogs") / str(case) / sampler
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOGDIR"] = str(log_dir)
    if max_ncalls is None:
        max_ncalls = int(os.environ.get("MAX_NCALLS", "100000"))
    return run_sampler(names, loglike, transform=transform, max_ncalls=max_ncalls)
