#!/usr/bin/env python3
"""Prepare a MOSFiT model once, then use any common inference backend."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

class _Capture(RuntimeError):
    def __init__(self, model, fitter): self.model=model; self.fitter=fitter

def capture(event_path, model_name):
    from mosfit.fitter import Fitter
    original=Fitter.fit_data
    def intercept(self,*args,**kwargs):
        raise _Capture(self._model,self)
    Fitter.fit_data=intercept
    try:
        fitter=Fitter(quiet=False,test=False)
        try:
            fitter.fit_events(events=[str(event_path)],models=[model_name],method="dynesty",iterations=1,fracking=False,burn=0,post_burn=0,write=False,return_fits=False,save_full_chain=False,quick_save=True)
        except _Capture as exc:
            return exc.fitter,exc.model
    finally:
        Fitter.fit_data=original
    raise RuntimeError("did not capture prepared MOSFiT model")

def names_for(model):
    for attr in ("_free_parameter_names","free_parameter_names","_parameter_names"):
        v=getattr(model,attr,None)
        if v is not None:
            if callable(v): v=v()
            if isinstance(v,dict): v=list(v.keys())
            else: v=list(v)
            if v: return [str(x) for x in v]
    for attr in ("free_parameters","parameters","_free_parameters","_parameters"):
        v=getattr(model,attr,None)
        if v is None: continue
        if hasattr(v,"keys"):
            k=list(v.keys())
            if k: return [str(x) for x in k]
        try: objs=list(v)
        except TypeError: continue
        out=[]
        for x in objs:
            n=getattr(x,"name",None) or getattr(x,"fullname",None) or getattr(x,"path",None)
            if n is not None: out.append(str(n))
        if out: return out
    setup=getattr(model,"_parameter_json",None)
    if isinstance(setup,dict):
        out=[]
        for n,spec in setup.items():
            if isinstance(spec,dict) and spec.get("fixed",False): continue
            out.append(str(n))
        if out: return out
    raise RuntimeError("could not determine MOSFiT free parameter names")

def main():
    root=Path(__file__).resolve().parent; event=root/"LSQ12dlf.json"
    if not event.exists(): raise RuntimeError("LSQ12dlf.json missing; run setup first")
    case=os.environ.get("PROBLEM","SLSN-LSQ12dlf"); model_name="slsn" if case=="SLSN-LSQ12dlf" else "magnetar"
    fitter,model=capture(event,model_name); del fitter
    names=names_for(model); ndim=len(names)
    def transform(u):
        x=np.asarray(u,float); return np.asarray(model.draw_from_icdf(np.clip(x,1e-12,1-1e-12)),float)
    def loglike(theta):
        v=float(model.ln_likelihood(np.asarray(theta,float))); return v if np.isfinite(v) else -1e300
    from autosampler import run_sampler
    sampler=os.environ.get("SAMPLER","ultranest"); log_dir=root/"systematiclogs"/case/sampler
    log_dir.mkdir(parents=True,exist_ok=True); os.environ["LOGDIR"]=str(log_dir)
    # Validate the adapter once so a bad parameter-name inference fails before the expensive run.
    theta=transform(np.full(ndim,0.5));
    if theta.shape!=(ndim,): raise RuntimeError(f"MOSFiT transform returned {theta.shape}, expected {(ndim,)}")
    run_sampler(names,loglike,transform=transform)
if __name__=="__main__": main()
