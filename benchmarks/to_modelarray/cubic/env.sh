#!/usr/bin/env bash
# Shared environment for the CUBIC validation ladder. Source this file.
# Everything writable lives under /cbica/home/tongt (never comp_space, never the project dir).
export OPT_ROOT=/cbica/home/tongt/modelarray_opt
export REPO=$OPT_ROOT/ModelArrayIO
export VENV=$OPT_ROOT/.venv
export SIF=/cbica/home/tongt/software/modelarray_confixel_0.1.5.sif

# uv / apptainer caches must not land in the (small, purged) default locations.
export UV_CACHE_DIR=$OPT_ROOT/.uv-cache
export UV_PYTHON_INSTALL_DIR=$OPT_ROOT/.uv-python
export APPTAINER_CACHEDIR=$OPT_ROOT/.apptainer-cache
# The production cohort CSV uses /gpfs/fs001/... absolute paths; bind both spellings.
export APPTAINER_BIND=/cbica/home/tongt,/cbica/projects/pennlinc_rbc,/gpfs/fs001

# The login node sets a 2 h CPU-time ulimit that SLURM propagates into jobs; the
# original converter needs ~3 h CPU per full CIFTI scalar and was killed by SIGXCPU.
# Lift it (soft -> hard) for every job and gate; submit with --propagate=NONE too.
ulimit -t unlimited 2>/dev/null || ulimit -t "$(ulimit -Ht)" 2>/dev/null || true
export MODELARRAYIO_BENCHMARK_RESULTS_DIR=$OPT_ROOT/results
export MODELARRAYIO_BASELINE_ROOT=$OPT_ROOT/baseline-9ab13dd
export MODELARRAYIO_PYTHON=$VENV/bin/python
export MODELARRAYIO_S3_ANON=1

export REF_TDB=/cbica/projects/pennlinc_rbc/modelarray_rbc_tiledb/derivatives/rbc_freesurferpost.tdb
export RBC_COHORT=/cbica/projects/pennlinc_rbc/modelarray_rbc_tiledb/derivatives/rbc_freesurferpost_cohort.csv

# Current ModelArray (installed from source into this user library) is the HDF5 gate;
# the container ships ModelArray 0.1.5, which reads column_names from an HDF5 attribute
# and cannot parse the current format.
export MODELARRAYIO_R_LIBS=$OPT_ROOT/Rlib
export R_LIBS_USER=$OPT_ROOT/Rlib
export GATE="$REPO/benchmarks/to_modelarray/cubic/gate.sh {output} {backend}"

# Run a command inside the ModelArray container (Ubuntu 20.04, glibc 2.31) with the
# venv first on PATH. When already inside the container, run it directly.
in_sif() {
  if [ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]; then
    PATH="$VENV/bin:/cbica/home/tongt/.local/bin:$PATH" "$@"
  else
    apptainer exec "$SIF" env PATH="$VENV/bin:/cbica/home/tongt/.local/bin:$PATH" "$@"
  fi
}
export -f in_sif
