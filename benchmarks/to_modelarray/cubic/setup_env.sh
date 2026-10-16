#!/usr/bin/env bash
# Step 0 (login node): environment, baseline export, public data, cohorts.
# Route: run everything inside the ModelArray container (glibc 2.31) with a
# uv-managed Python 3.12 so the exact local library versions install as wheels.
set -euo pipefail
source "$(dirname "$0")/env.sh"
UV=/cbica/home/tongt/.local/bin/uv
mkdir -p "$OPT_ROOT"/{data,cohorts,results,logs} "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$APPTAINER_CACHEDIR"

echo "== quota =="; df -h /cbica/home/tongt | tail -1

echo "== python 3.12 (uv-managed; runs on glibc >= 2.17, host and container) =="
"$UV" python install 3.12
PY312=$("$UV" python find 3.12)
echo "python: $PY312"

echo "== venv inside the container so wheels resolve against its glibc =="
in_sif "$UV" venv --python "$PY312" "$VENV"
in_sif "$UV" pip install --python "$VENV/bin/python" -e "$REPO[s3]" pytest reportlab
in_sif python - <<'PY'
import h5py, nibabel, numpy, tiledb, modelarrayio, sys
print('in-container:', sys.version.split()[0], 'numpy', numpy.__version__, 'h5py', h5py.__version__,
      'tiledb', tiledb.__version__, 'nibabel', nibabel.__version__)
PY

echo "== current ModelArray R package into a user library (the container ships 0.1.5) =="
# ModelArray 0.1.5 reads column_names from an HDF5 *attribute*; the current format (and the
# converter) store it as a dataset, so the HDF5 gate must use the current ModelArray source,
# rsynced to $OPT_ROOT/ModelArray. Deps come from CRAN (login node has internet).
mkdir -p "$OPT_ROOT/Rlib"
in_sif Rscript -e 'lib<-Sys.getenv("R_LIBS_USER"); .libPaths(c(lib,.libPaths())); deps<-c("DelayedArray","HDF5Array","broom","crayon","dplyr","glue","hdf5r","magrittr","mgcv","pbapply","pbmcapply","rhdf5","tibble","tidyr","cheapr"); miss<-deps[!vapply(deps, requireNamespace, TRUE, quietly=TRUE)]; cat("missing:", miss, "\n"); if (length(miss)) install.packages(miss, lib=lib, repos="https://cloud.r-project.org", Ncpus=4)'
in_sif R CMD INSTALL --library="$OPT_ROOT/Rlib" "$OPT_ROOT/ModelArray"
in_sif Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); cat("ModelArray", as.character(packageVersion("ModelArray")), "from", find.package("ModelArray"), "\n")'

echo "== baseline 9ab13dd export (host git; cache under home so compute nodes see it) =="
"$REPO/benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd" --prepare-only

echo "== source identity (must equal the local tree) =="
in_sif python "$REPO/benchmarks/to_modelarray/source_identity.py" --repo "$REPO" | tee "$OPT_ROOT/source_identity.json"

echo "== public NIfTI (ABIDE-100) materialized locally =="
in_sif python "$REPO/benchmarks/to_modelarray/make_cohort.py" --subjects 100 --derivatives alff \
  --output "$OPT_ROOT/cohorts/abide-100-s3.csv"
GROUP_MASK_KEY=s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_mask/CMU_a_0050642_func_mask.nii.gz
in_sif python "$REPO/benchmarks/to_modelarray/materialize_cohort.py" --cohort "$OPT_ROOT/cohorts/abide-100-s3.csv" \
  --data-dir "$OPT_ROOT/data/abide" --output "$OPT_ROOT/cohorts/abide-100-local.csv" \
  --extra-key "$GROUP_MASK_KEY"
cp "$OPT_ROOT/data/abide/func_mask/CMU_a_0050642_func_mask.nii.gz" "$OPT_ROOT/data/group_mask.nii.gz"

echo "== public MIF (walkthrough FDC) =="
in_sif python "$REPO/benchmarks/to_modelarray/make_mif_cohort.py" --data-dir "$OPT_ROOT/data/mif-walkthrough" \
  --output "$OPT_ROOT/cohorts/mif-walkthrough.csv"

echo "== real CIFTI subset cohorts (read-only project CSV, no copying) =="
in_sif python "$REPO/benchmarks/to_modelarray/make_cifti_cohort.py" subset --input "$RBC_COHORT" \
  --output "$OPT_ROOT/cohorts/cifti-rbc-100x3ref.csv" --subjects 100 --scalars area areaPial curv

# The reference listing opens TileDB arrays; libtiledb needs a compute-node CPU
# (the login node raised SIGILL), so it runs in sbatch/00_smoke.sbatch instead.
echo "setup complete (run sbatch/00_smoke.sbatch next; then make_scalar_cohorts.sh)"
