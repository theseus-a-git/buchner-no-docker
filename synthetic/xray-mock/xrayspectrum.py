import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from numpy import exp
from autosampler import run_sampler

def main(args):
    powerlaw_true, nh_true, fscat_true = 2, 100, 0.04
    amplitude_true, background_true = args.contrast, args.background
    np.random.seed(int(args.contrast))
    paramnames = ["logamplitude", "photonindex", "lognh", "logfscat", "background"]
    E = np.linspace(0.5, 8, 200)
    sensitivity = exp(-np.abs((E-2)/2.0))
    model = amplitude_true * E**-powerlaw_true * (fscat_true + exp(-nh_true*E**-3))
    model_convolved = model * sensitivity + background_true
    data = np.random.poisson(model_convolved)
    plt.plot(E, model, label="intrinsic model")
    plt.plot(E, model * 0 + background_true, label="background")
    plt.plot(E, model_convolved, label="convolved model with background")
    plt.plot(E, sensitivity, color="gray", label="instrument sensitivity")
    plt.plot(E, data, "x ", label="data")
    plt.yscale("log"); plt.xscale("log"); plt.legend(loc="best")
    plt.savefig("xrayspectrum-%s.pdf" % os.environ.get("SAMPLER", "x"), bbox_inches="tight"); plt.close()

    def loglike(params):
        logamplitude, photonindex, lognh, logfscat, background = params
        model = 10**logamplitude * E**-photonindex * (10**logfscat + exp(-10**lognh * E**-3))
        model_convolved = model * sensitivity + background_true
        vals = scipy.stats.poisson.logpmf(data, model_convolved)
        vals[~np.isfinite(vals)] = -1e300
        return float(vals.sum())

    def transform(x):
        x = np.asarray(x, dtype=float).copy()
        z = x.copy()
        z[0] = x[0]*10 - 5
        z[1] = scipy.stats.norm.ppf(np.clip(x[1], 1e-12, 1-1e-12), 2.0, 0.2)
        z[2] = x[2]*6 - 3
        z[3] = -1 - x[3]*6
        z[4] = exp(scipy.stats.norm.ppf(np.clip(x[4], 1e-12, 1-1e-12), np.log(background_true), 0.1))
        return z

    case = os.environ.get("PROBLEM", "xray-mock-compton-thick-40-0.01")
    sampler = os.environ.get("SAMPLER", "ultranest")
    log_dir = Path("systematiclogs") / case / sampler
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOGDIR"] = str(log_dir)
    run_sampler(paramnames, loglike, transform=transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast", type=int, default=100)
    parser.add_argument("--background", type=float, default=0.2)
    parser.add_argument("--ndata", type=int, default=40)
    parser.add_argument("--num_live_points", type=int, default=400)
    parser.add_argument("--log_dir", default="logs/xrayspectrum")
    parser.add_argument("--reactive", action="store_true")
    parser.add_argument("--pymultinest", action="store_true")
    parser.add_argument("--slice_steps", type=int, default=100)
    parser.add_argument("--adapt_steps")
    main(parser.parse_args())
