#!/usr/bin/env bash
# Shared environment for the CUBIC validation ladder. Source this file.
#
# No site-specific paths are committed. Export these before running any step;
# sbatch passes the submitting shell's environment on to the job:
#   OPT_ROOT    writable work directory: ModelArrayIO checkout, venv, caches,
#               cohorts, results and logs (the input data is only ever read)
#   SIF         ModelArray container image
#   REF_TDB     deployed reference TileDB output (read only)
#   RBC_COHORT  production cohort CSV (read only)
# Optional:
#   REPO        ModelArrayIO checkout (default: $OPT_ROOT/ModelArrayIO)
#   DATA_BINDS  comma-separated read-only directories to bind into the container:
#               the reference output's directory and every mount path the cohort
#               CSV uses for its source files
: "${OPT_ROOT:?export OPT_ROOT, the writable work directory (see cubic/env.sh)}"
: "${SIF:?export SIF, the ModelArray container image (see cubic/env.sh)}"
: "${REF_TDB:?export REF_TDB, the deployed reference TileDB output (see cubic/env.sh)}"
: "${RBC_COHORT:?export RBC_COHORT, the production cohort CSV (see cubic/env.sh)}"
export OPT_ROOT SIF REF_TDB RBC_COHORT
export REPO=${REPO:-$OPT_ROOT/ModelArrayIO}
export VENV=$OPT_ROOT/.venv

# uv / apptainer caches must not land in the (small, purged) default locations.
export UV_CACHE_DIR=$OPT_ROOT/.uv-cache
export UV_PYTHON_INSTALL_DIR=$OPT_ROOT/.uv-python
export APPTAINER_CACHEDIR=$OPT_ROOT/.apptainer-cache
export APPTAINER_BIND=$OPT_ROOT${DATA_BINDS:+,$DATA_BINDS}

# The login node sets a 2 h CPU-time ulimit that SLURM propagates into jobs; the
# original converter needs ~3 h CPU per full CIFTI scalar and was killed by SIGXCPU.
# Lift it (soft -> hard) for every job and gate; submit with --propagate=NONE too.
ulimit -t unlimited 2>/dev/null || ulimit -t "$(ulimit -Ht)" 2>/dev/null || true
export MODELARRAYIO_BENCHMARK_RESULTS_DIR=$OPT_ROOT/results
export MODELARRAYIO_BASELINE_ROOT=$OPT_ROOT/baseline-9ab13dd
export MODELARRAYIO_PYTHON=$VENV/bin/python
export MODELARRAYIO_S3_ANON=1

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
    PATH="$VENV/bin:$HOME/.local/bin:$PATH" "$@"
  else
    apptainer exec "$SIF" env PATH="$VENV/bin:$HOME/.local/bin:$PATH" "$@"
  fi
}
export -f in_sif
