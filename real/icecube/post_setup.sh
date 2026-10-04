#!/usr/bin/env bash
# Validate the PISA package resources needed by the benchmark.
# PISA ships the IceCube 3-year configs/data as package resources under
# pisa_examples/resources, so the benchmark should not depend on the caller's
# working directory or on a separate checked-out PISA source tree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENVN=inf-icecube

conda run --no-capture-output -n "$ENVN" python - <<'PY'
from pisa.utils.resources import find_resource

required = [
    "settings/pipeline/IceCube_3y_neutrinos.cfg",
    "settings/pipeline/IceCube_3y_muons.cfg",
    "settings/pipeline/IceCube_3y_data.cfg",
]

for name in required:
    path = find_resource(name, fail=True)
    print(f"PISA resource OK: {name} -> {path}")
PY

echo "PISA 3-year benchmark resources are available from the installed PISA package."
