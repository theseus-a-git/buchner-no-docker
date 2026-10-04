#!/bin/bash
# Builds the comparison tables once the run arrays have finished (submitted with --dependency=afterany by submit.sh).
. "$INFENV_ROOT/slurm/env.sh"
cd "$INFENV_ROOT" && ./compare.sh --samplers ${IE_SAMPLERS} --ref ultranest
