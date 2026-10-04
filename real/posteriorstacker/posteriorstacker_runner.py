#!/usr/bin/env python3
"""Expose PosteriorStacker tutorial models to all comparison backends."""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PSDIR=HERE/"PosteriorStacker"
DATAFILE=PSDIR/"posteriorsamples.txt"
LOW,HIGH,NBINS=-80.0,80.0,11

def ensure_data():
    if DATAFILE.exists(): return
    subprocess.run([sys.executable,"tutorial/gendata.py"],cwd=PSDIR,check=True)
    alt=PSDIR/"tutorial"/"posteriorsamples.txt"
    if not DATAFILE.exists() and alt.exists(): shutil.copy2(alt,DATAFILE)
    if not DATAFILE.exists(): raise RuntimeError(f"missing {DATAFILE} after gendata.py")

def main():
    ensure_data(); data=np.loadtxt(DATAFILE)
    if data.ndim!=2: raise RuntimeError(f"expected 2-D posterior samples, got {data.shape}")
    _,nsamples=data.shape
    bins=np.linspace(LOW,HIGH,NBINS+1)
    binned_data=np.array([np.histogram(row,bins=bins)[0] for row in data])
    case=os.environ.get("PROBLEM","dist-gauss"); sampler=os.environ.get("SAMPLER","ultranest")
    log_dir=HERE/"systematiclogs"/case/sampler; log_dir.mkdir(parents=True,exist_ok=True); os.environ["LOGDIR"]=str(log_dir)
    from autosampler import run_sampler
    if case=="dist-hist":
        names=[f"bin{i+1}" for i in range(NBINS)]
        def likelihood(params):
            p=np.asarray(params,float)
            return float(np.log(np.dot(binned_data,p)/nsamples+1e-300).sum())
        def transform(cube):
            q=np.clip(np.asarray(cube,float),1e-12,1-1e-12); g=-np.log(q); return g/g.sum()
    elif case=="dist-gauss":
        names=["mean","std"]
        def normal_pdf(x,mean,std): return np.exp(-0.5*((x-mean)/std)**2)/(std*(2*np.pi)**0.5)
        def likelihood(params):
            mean,std=params
            if std<=0: return -1e300
            return float(np.log(normal_pdf(data,mean,std).mean(axis=1)+1e-300).sum())
        def transform(cube):
            cube=np.asarray(cube,float); p=cube.copy(); p[0]=3*(HIGH-LOW)*cube[0]+LOW; p[1]=cube[1]*(HIGH-LOW)*3; return p
    else: raise ValueError(f"unknown PosteriorStacker case: {case}")
    run_sampler(names,likelihood,transform=transform)
if __name__=="__main__": main()
