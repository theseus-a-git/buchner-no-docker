#!/usr/bin/env bash
# fermipy from pip WITHOUT its dependencies, so pip cannot replace conda's numpy/scipy/astropy
conda run -n inf-crab pip install --no-deps fermipy
