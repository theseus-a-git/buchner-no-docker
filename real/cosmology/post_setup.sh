#!/usr/bin/env bash
# Builds CLASS 3.2.0 (C + python wrapper "classy") and clones Buchner's MontePython fork (has the "UN" UltraNest mode).
# Needs a C compiler: sudo apt install -y build-essential gfortran
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENVN=inf-cosmology
command -v gcc >/dev/null && command -v make >/dev/null || { echo "!! need a compiler: sudo apt install -y build-essential gfortran"; exit 1; }
cd "$HERE"
# the old tarball URL on lesgourg.github.io is gone (404); use the tagged release from GitHub
if [ ! -f class_public-3.2.0/Makefile ]; then
  rm -rf class_public-3.2.0
  git clone --depth 1 --branch v3.2.0 https://github.com/lesgourg/class_public class_public-3.2.0 \
    || { echo "!! could not fetch CLASS v3.2.0 from github"; exit 1; }
fi
# CLASS 3.2.0 is pre-C23 code: with GCC >= 15 (default -std=gnu23) it fails with "too many arguments to function ... expected 0"
export CFLAGS="-std=gnu17 ${CFLAGS:-}"
make -C class_public-3.2.0 clean >/dev/null 2>&1 || true
# build only the C targets: the default "classy" make target would install the wrapper into whatever python is first on PATH (and its
# "export CC=..." line breaks when CC contains a space), so the wrapper is built into the conda env below instead
make -C class_public-3.2.0 -j4 libclass.a class CC="gcc -std=gnu17"
( cd class_public-3.2.0/python && conda run --no-capture-output -n $ENVN python setup.py build && conda run --no-capture-output -n $ENVN python setup.py install )
[ -d montepython_public ] || git clone --depth 1 https://github.com/JohannesBuchner/montepython_public montepython_public
# the fork's UltraNest.py references a variable (output) that is never assigned; see patch_montepython.py
python3 "$HERE/patch_montepython.py" "$HERE/montepython_public/montepython/UltraNest.py"
if [ ! -f montepython_public/default.conf ]; then
  [ -f montepython_public/default.conf.template ] || { echo "!! montepython_public/default.conf.template not found, create default.conf by hand"; exit 1; }
  cp montepython_public/default.conf.template montepython_public/default.conf
  sed -i "s|^path\['cosmo'\].*|path['cosmo'] = '$HERE/class_public-3.2.0'|" montepython_public/default.conf
  sed -i "s|^\(path\['clik'\]\)|#\1|" montepython_public/default.conf   # Planck clik is not used by the example
fi
echo "--- default.conf:"; grep -n "^path\|^#path" montepython_public/default.conf || true
cd montepython_public && conda run --no-capture-output -n $ENVN python montepython/MontePython.py --help | head -5
