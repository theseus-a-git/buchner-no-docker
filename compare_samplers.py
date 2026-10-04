#!/usr/bin/env python3
"""Compare UltraNest, Dynesty, Nestle, emcee and PSO across completed runs.

Posterior-producing methods are compared using posterior samples; nested
samplers additionally report evidence agreement. PSO is reported as an
optimizer and is never treated as if it produced posterior samples/evidence.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLERS=("ultranest","dynesty","nestle","emcee","pso")
SKIP={"testsampler"}
ALIASES={"ultranest":{"ultranest","ultranest-safe"}}

def finite_num(x):
    try:
        y=float(x); return y if np.isfinite(y) else np.nan
    except Exception: return np.nan

def sampler_kind(sampler,result=None):
    if isinstance(result,dict) and result.get("sampler_kind"): return result["sampler_kind"]
    if sampler=="pso": return "optimizer"
    if sampler in {"emcee","goodman-weare","slice"}: return "mcmc"
    return "nested"

def find_runs():
    runs={}
    pattern=os.path.join(ROOT,"*","*","systematiclogs","*","*","results.json")
    for res in glob.glob(pattern):
        rundir=os.path.dirname(res); sampler=os.path.basename(rundir)
        if sampler in SKIP: continue
        casedir=os.path.dirname(rundir); probdir=os.path.dirname(os.path.dirname(casedir))
        problem=os.path.relpath(probdir,ROOT).replace(os.sep,"/")+":"+os.path.basename(casedir)
        runs[(problem,sampler)]=rundir
    return runs

def find_expected_problems():
    """Return benchmark cases declared by problem.conf, even when no result exists yet."""
    expected=[]
    for conf in glob.glob(os.path.join(ROOT,"real","*","problem.conf")) + glob.glob(os.path.join(ROOT,"synthetic","*","problem.conf")):
        probdir=os.path.dirname(conf)
        relprob=os.path.relpath(probdir,ROOT).replace(os.sep,"/")
        text=open(conf).read()
        cases=[]
        m=re.search(r'^CASES=[\"\']([^\"\']*)[\"\']\s*$',text,re.M)
        if m:
            cases=m.group(1).split()
        elif re.search(r'^CASES=\$\(cat\s+problems\.list\)\s*$',text,re.M):
            pl=Path(probdir)/"problems.list"
            cases=pl.read_text().split() if pl.exists() else []
        else:
            cases=[os.path.basename(probdir)]
        for case in cases:
            expected.append(relprob+":"+case)
    return sorted(set(expected))

def resolve_reference(runs,requested):
    for s in [requested,*sorted(ALIASES.get(requested,set()))]:
        if any(x==s for _,x in runs): return s
    return requested

def extract_best(d):
    if not isinstance(d,dict): return np.nan
    for key in ("best_loglike","best_logl","max_loglike","max_logl"):
        v=finite_num(d.get(key))
        if np.isfinite(v): return v
    ml=d.get("maximum_likelihood")
    if isinstance(ml,dict):
        for key in ("logl","loglike","log_likelihood","L"):
            v=finite_num(ml.get(key))
            if np.isfinite(v): return v
    else:
        v=finite_num(ml)
        if np.isfinite(v): return v
    return np.nan

def load_result(rundir,sampler):
    try:
        with open(os.path.join(rundir,"results.json")) as f: d=json.load(f)
    except Exception: return None
    return dict(sampler=sampler,kind=sampler_kind(sampler,d),logz=finite_num(d.get("logz")),logzerr=finite_num(d.get("logzerr")),best_loglike=extract_best(d),ncall=finite_num(d.get("ncall_counted", d.get("ncall"))),ncall_reported=finite_num(d.get("ncall")),budget=finite_num(d.get("max_ncalls_requested")),time=finite_num(d.get("walltime_s")),acceptance=finite_num(d.get("acceptance_fraction")))

def load_samples(rundir):
    f=os.path.join(rundir,"samples.txt.gz")
    if not os.path.exists(f): return None
    try: a=np.loadtxt(f,delimiter=",",comments="#")
    except Exception: return None
    if a is None or a.size==0: return None
    if a.ndim==1: a=a[:,None]
    good=np.all(np.isfinite(a),axis=1)
    return a[good]

def posterior_metrics(s,ref,rng):
    nan=dict(meanshift=np.nan,stdratio=np.nan,wass=np.nan)
    if s is None or ref is None or s.ndim!=2 or ref.ndim!=2 or s.shape[1]!=ref.shape[1] or len(s)<10 or len(ref)<10: return nan
    from scipy.stats import wasserstein_distance
    if len(s)>5000: s=s[rng.choice(len(s),5000,replace=False)]
    if len(ref)>5000: ref=ref[rng.choice(len(ref),5000,replace=False)]
    sd=np.where(ref.std(axis=0)>0,ref.std(axis=0),1.0)
    shift=np.abs(s.mean(axis=0)-ref.mean(axis=0))/sd
    ratio=s.std(axis=0)/sd
    positive=ratio>0
    worst=float(ratio[np.nanargmax(np.abs(np.log(np.where(positive,ratio,np.nan))))]) if np.any(positive) else np.nan
    wass=float(np.mean([wasserstein_distance(s[:,i],ref[:,i])/sd[i] for i in range(s.shape[1])]))
    return dict(meanshift=float(np.max(shift)),stdratio=worst,wass=wass)

def fmt(x,f): return "-" if x is None or not np.isfinite(x) else f%x

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref",default="ultranest")
    ap.add_argument("--problems",nargs="+")
    ap.add_argument("--samplers",nargs="+",default=list(DEFAULT_SAMPLERS),help="default: ultranest dynesty nestle emcee pso")
    ap.add_argument("--out",default=os.path.join(ROOT,"sampler_comparison"))
    ap.add_argument("--no-plot",action="store_true")
    args=ap.parse_args()
    requested=list(dict.fromkeys(args.samplers))
    runs=find_runs()
    expected=find_expected_problems()
    if args.problems:
        expected=[p for p in expected if any(sel in p for sel in args.problems)]
        runs={k:v for k,v in runs.items() if any(sel in k[0] for sel in args.problems)}
    if requested:
        aliases=set(requested)
        for s in requested: aliases |= ALIASES.get(s,set())
        runs={k:v for k,v in runs.items() if k[1] in aliases}
    if not expected and not runs:
        sys.exit("no declared problems or finished runs found; run ./run.sh real or ./run.sh synthetic first")
    ref_name=resolve_reference(runs,args.ref)
    rng=np.random.default_rng(1)
    problems=sorted(set(expected) | {k[0] for k in runs})
    rows=[]
    for prob in problems:
        available={s: runs[(prob,s)] for (p,s) in runs if p==prob and s in requested}
        loaded={s: load_result(d,s) for s,d in available.items()}
        samples={s: load_samples(d) for s,d in available.items()}
        ref=loaded.get(ref_name); ref_s=samples.get(ref_name)
        finite_best=[r["best_loglike"] for r in loaded.values() if r and np.isfinite(r["best_loglike"])]
        common_best=max(finite_best) if finite_best else np.nan
        for s in requested:
            if s not in available:
                rows.append(dict(problem=prob,sampler=s,kind="not-run",status="MISSING",logz=np.nan,logzerr=np.nan,dlogz=np.nan,z=np.nan,best_loglike=np.nan,delta_best_loglike=np.nan,meanshift=np.nan,stdratio=np.nan,wass=np.nan,ncall=np.nan,time_s=np.nan,acceptance=np.nan,posterior_samples=False,is_ref=False))
                continue
            r=loaded[s]
            if r is None:
                rows.append(dict(problem=prob,sampler=s,kind="error",status="INVALID",logz=np.nan,logzerr=np.nan,dlogz=np.nan,z=np.nan,best_loglike=np.nan,delta_best_loglike=np.nan,meanshift=np.nan,stdratio=np.nan,wass=np.nan,ncall=np.nan,time_s=np.nan,acceptance=np.nan,posterior_samples=False,is_ref=False))
                continue
            row=dict(problem=prob,sampler=s,kind=r["kind"],status="OK",logz=r["logz"],logzerr=r["logzerr"],dlogz=np.nan,z=np.nan,best_loglike=r["best_loglike"],delta_best_loglike=(common_best-r["best_loglike"] if np.isfinite(common_best) and np.isfinite(r["best_loglike"]) else np.nan),meanshift=np.nan,stdratio=np.nan,wass=np.nan,ncall=r["ncall"],time_s=r["time"],acceptance=r["acceptance"],posterior_samples=samples[s] is not None,is_ref=(s==ref_name))
            if ref is not None and s!=ref_name:
                if np.isfinite(r["logz"]) and np.isfinite(ref["logz"]):
                    row["dlogz"]=r["logz"]-ref["logz"]; err=np.hypot(np.nan_to_num(r["logzerr"]),np.nan_to_num(ref["logzerr"])); row["z"]=row["dlogz"]/err if err>0 else np.nan
                row.update(posterior_metrics(samples[s],ref_s,rng))
            rows.append(row)
    cols=["problem","sampler","kind","status","logz","logzerr","dlogz","z","best_loglike","delta_best_loglike","meanshift","stdratio","wass","ncall","time_s","acceptance","posterior_samples"]
    with open(args.out+".csv","w") as f:
        f.write(",".join(cols)+"\n")
        for r in rows:
            vals=[]
            for c in cols:
                v=r[c]
                vals.append("" if isinstance(v,(float,np.floating)) and not np.isfinite(v) else str(v))
            f.write(",".join(vals)+"\n")
    lines=["| problem | sampler | kind | status | logZ | +- | dlogZ | z | best logL | ΔlogL to best found | mean shift | std ratio | W1/std | ncall | time [s] | accept | posterior |","|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"%(r["problem"],r["sampler"]," (ref)" if r["is_ref"] else "",r["kind"],r["status"],fmt(r["logz"],"%.2f"),fmt(r["logzerr"],"%.2f"),fmt(r["dlogz"],"%+.2f"),fmt(r["z"],"%+.1f"),fmt(r["best_loglike"],"%.3f"),fmt(r["delta_best_loglike"],"%.3f"),fmt(r["meanshift"],"%.2f"),fmt(r["stdratio"],"%.2f"),fmt(r["wass"],"%.3f"),fmt(r["ncall"],"%.0f"),fmt(r["time_s"],"%.1f"),fmt(r["acceptance"],"%.3f"),"yes" if r["posterior_samples"] else "no"))
    lines += ["","**Interpretation**","","- UltraNest, Dynesty and Nestle are nested samplers: compare `logZ`/`logZerr`, posterior samples, best log-likelihood, calls and time.","- emcee is MCMC: compare posterior samples, best log-likelihood, calls, time and acceptance; it does not provide evidence here.","- PSO is an optimizer: compare best log-likelihood, best-fit parameters, calls, convergence history and time; it does not provide posterior samples or evidence.","- A `MISSING` row means that sampler was requested but no completed result exists for that problem.","- `ΔlogL to best found` is relative to the best finite value reported by any available method; it is not a formal convergence diagnostic.","","**Per-sampler completeness**","","| sampler | requested problems | completed | missing | posterior runs | median time [s] | median ncall |","|---|---:|---:|---:|---:|---:|---:|"]
    for s in requested:
        rs=[r for r in rows if r["sampler"]==s]; ok=[r for r in rs if r["status"]=="OK"]; miss=[r for r in rs if r["status"]!="OK"]; times=[r["time_s"] for r in ok if np.isfinite(r["time_s"])]; calls=[r["ncall"] for r in ok if np.isfinite(r["ncall"])]
        lines.append("| %s | %d | %d | %d | %d | %s | %s |"%(s,len(rs),len(ok),len(miss),sum(r["posterior_samples"] for r in ok),fmt(np.median(times) if times else np.nan,"%.1f"),fmt(np.median(calls) if calls else np.nan,"%.0f")))
    text="\n".join(lines)+"\n"; Path(args.out+".md").write_text(text); print(text)
    if not args.no_plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            samplers=requested; probs=problems
            panels=[("time_s","time [s]",True),("ncall","likelihood calls",True),("delta_best_loglike","ΔlogL to best",False),("wass","posterior W1/std",False)]
            fig,axes=plt.subplots(1,len(panels),figsize=(4+2.8*len(samplers),1.6+0.30*len(probs)),sharey=True,squeeze=False)
            for ax,(key,title,log) in zip(axes[0],panels):
                M=np.full((len(probs),len(samplers)),np.nan)
                for r in rows:
                    v=r[key]
                    if np.isfinite(v) and (v>0 or not log): M[probs.index(r["problem"]),samplers.index(r["sampler"])]=np.log10(v) if log else v
                im=ax.imshow(M,aspect="auto",cmap="viridis"); ax.set_xticks(range(len(samplers))); ax.set_xticklabels(samplers,rotation=60,ha="right",fontsize=8); ax.set_title(title,fontsize=9)
                for i in range(len(probs)):
                    for j in range(len(samplers)):
                        if np.isfinite(M[i,j]): ax.text(j,i,"%.2f"%M[i,j],ha="center",va="center",fontsize=6,color="w")
                fig.colorbar(im,ax=ax,fraction=0.04)
            axes[0][0].set_yticks(range(len(probs))); axes[0][0].set_yticklabels(probs,fontsize=7); fig.tight_layout(); fig.savefig(args.out+".png",dpi=130); plt.close(fig)
        except Exception as e: print("plot skipped:",e)

if __name__=="__main__": main()
