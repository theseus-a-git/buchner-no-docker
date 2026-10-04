#!/usr/bin/env bash
# MOSFiT no longer downloads events by name. The installed package ships LSQ12dlf.json (test fixture); copy it here.
HERE="$(cd "$(dirname "$0")" && pwd)"
d=$(conda run -n inf-mosfit python -c "import mosfit,os; print(os.path.dirname(mosfit.__file__))" 2>/dev/null | tail -1)
if [ -f "$d/tests/LSQ12dlf.json" ]; then cp "$d/tests/LSQ12dlf.json" "$HERE/LSQ12dlf.json" && echo "copied LSQ12dlf.json"
else echo "!! LSQ12dlf.json not found in the mosfit package; put an event JSON at real/mosfit/LSQ12dlf.json"; fi
