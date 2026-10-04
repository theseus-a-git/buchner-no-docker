#!/usr/bin/env bash
# Run ONCE on the LOGIN node after setup (it has internet; compute nodes may not). Downloads the data that
# would otherwise be fetched while a job is running, so those jobs only read local files:
#   xray-agn-ciao  UXCLUMPY tables (2 x ~520 MB)       exoplanet exo-transient  TESS light curve
#   ligo           3 GW170817 strain files (astropy cache under $XDG_CACHE_HOME)
# NOT prefetchable: real/crab and real/grb query the Fermi/HEASARC servers while they run, and real/ligo also queries
# the GWOSC event catalogue. They are marked net=1 in resources.tsv; see slurm/README.md ("Internet on compute nodes").
. "$(dirname "$0")/env.sh"
cd "$INFENV_ROOT" || exit 1
fail=0
if [ -d real/xray-agn-ciao ]; then
  echo "== xray-agn-ciao models"
  if [ -s real/xray-agn-ciao/models/uxclumpy-cutoff.fits ] && [ -s real/xray-agn-ciao/models/uxclumpy-cutoff-omni.fits ]; then echo "   present"
  else bash real/xray-agn-ciao/fetch_models.sh || fail=1; fi
fi
echo "== exoplanet TESS light curve"
( cd real/exoplanet && conda run -n inf-exoplanet --no-capture-output python -c "import transient as t; p=t.DEFAULT_DATA; (p.exists() or t.download(t.TESS_URL, p)); print('   ', p, p.stat().st_size, 'bytes')" ) || { echo "   !! failed"; fail=1; }
echo "== ligo strain data"
conda run -n inf-ligo --no-capture-output python -c "from astropy.utils.data import download_file as d; U='https://dcc.ligo.org/public/0146/P1700349/001/{}-{}1_LOSC_CLN_4_V1-1187007040-2048.gwf'; [print('   ', d(U.format(i, i), cache=True)) for i in 'HVL']" || { echo "   !! failed"; fail=1; }
[ $fail = 0 ] && echo "prefetch done" || echo "prefetch finished with errors (see above)"
exit $fail
