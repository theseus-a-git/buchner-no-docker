import warnings
import os

warnings.simplefilter("ignore")
import numpy as np

np.seterr(all="ignore")
import shutil
from pathlib import Path
#import matplotlib as mpl
#from matplotlib import pyplot as plt
#from astropy.io import fits as pyfits
#import scipy as sp

from threeML import *
silence_warnings()
#set_threeML_style()

print("=== The Fermi 4FGL catalog")

lat_catalog = FermiLATSourceCatalog()

ra, dec, table = lat_catalog.search_around_source("Crab", radius=20.0)

print(table)

model = lat_catalog.get_model()

model.free_point_sources_within_radius(3.0, normalization_only=True)

model["Crab_IC.spectrum.main.Log_parabola.K"].fix = True
model.Crab_synch.spectrum.main.Log_parabola.K.fix = True

model.PSR_J0534p2200.spectrum.main.Super_cutoff_powerlaw.index.free = True

for param in model.free_parameters.values():
    if param.has_transformation():
        param.set_uninformative_prior(Log_uniform_prior)
    else:
        param.set_uninformative_prior(Uniform_prior)

model.display()

tstart = "2010-01-01 00:00:00"
tstop = "2010-02-01 00:00:00"

# Note that this will understand if you already download these files, and will
# not do it twice unless you change your selection or the outdir

evfile, scfile = download_LAT_data(
    ra,
    dec,
    20.0,
    tstart,
    tstop,
    time_type="Gregorian",
    destination_directory="Crab_data",
)

print("=== Configuration for Fermipy")
config = FermipyLike.get_basic_config(
    evfile=evfile,
    scfile=scfile,
    ra=ra,
    dec=dec,
    fermipy_verbosity=1,
    fermitools_chatter=0,
)

# See what we just got

config.display()

config["selection"]["emax"] = 300000.0

config["gtlike"] = {"edisp": False}

config.display()

print("=== FermipyLike")
LAT = FermipyLike("LAT", config)

fermipy_output_directory = Path(config["fileio"]["outdir"])
print("Fermipy Output directory: %s" % fermipy_output_directory)

# This removes the output directory, to start a fresh analysis...

#if fermipy_output_directory.exists():
#    shutil.rmtree(fermipy_output_directory)

# Here is where the fermipy processing happens (the .setup method)

data = DataList(LAT)

bayes = BayesianAnalysis(model, data)

for param in model.free_parameters.values():
    if param.has_transformation():
        print("log-uniform prior assigned to:", param)
        param.set_uninformative_prior(Log_uniform_prior)
    else:
        print("uniform prior assigned to:", param)
        param.set_uninformative_prior(Uniform_prior)

model.display()


print("sampling ...")
from threeml_adapter import run_3ml
run_3ml(model, data, case_name=os.environ.get("PROBLEM", "Crab"))
print("sampling done.")
