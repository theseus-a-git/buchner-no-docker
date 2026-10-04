#!/usr/bin/env python3
"""Run BXA's existing Chandra AGN setup with all common sampler backends."""
from __future__ import annotations
import os, runpy
from pathlib import Path
import numpy as np

class _Capture(RuntimeError):
    def __init__(self,solver): self.solver=solver

def main():
    here=Path(__file__).resolve().parent; script=here/"BXA"/"examples"/"sherpa"/"xagnfitter.py"
    if not script.exists(): raise RuntimeError(f"BXA example missing: {script}; rerun setup")
    from bxa.sherpa.solver import BXASolver
    original=BXASolver.run
    def intercept(self,*args,**kwargs): raise _Capture(self)
    BXASolver.run=intercept
    try:
        try: runpy.run_path(str(script),run_name="__main__")
        except _Capture as exc: solver=exc.solver
        else: raise RuntimeError("BXA example did not instantiate/run BXASolver")
    finally: BXASolver.run=original
    from autosampler import run_sampler
    names=list(solver.paramnames)
    def transform(u): return np.asarray(solver.prior_transform(np.clip(np.asarray(u,float),1e-12,1-1e-12)),float)
    def loglike(theta): return float(solver.log_likelihood(np.asarray(theta,float)))
    sampler=os.environ.get("SAMPLER","ultranest"); log_dir=here/"systematiclogs"/os.environ.get("PROBLEM", "xray-AGN-spectrum")/sampler
    log_dir.mkdir(parents=True,exist_ok=True); os.environ["LOGDIR"]=str(log_dir)
    run_sampler(names,loglike,transform=transform)
if __name__=="__main__": main()
