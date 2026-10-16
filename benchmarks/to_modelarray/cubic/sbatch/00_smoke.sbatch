#!/usr/bin/env bash
#SBATCH -J ma-smoke
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o /cbica/home/tongt/modelarray_opt/logs/%x-%j.out
set -euo pipefail
source /cbica/home/tongt/modelarray_opt/ModelArrayIO/benchmarks/to_modelarray/cubic/env.sh
cd "$REPO"
echo "== reference scalar completeness (needs a compute-node CPU for libtiledb) =="
in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --list --output-csv "$OPT_ROOT/reference_scalars.csv"
echo "== compute-node imports =="; in_sif python -c "import tiledb,h5py,numpy,nibabel,modelarrayio;print('ok',tiledb.__version__,h5py.__version__,numpy.__version__)"
echo "== source identity =="; in_sif python benchmarks/to_modelarray/source_identity.py --repo "$REPO"
echo "== pytest (offline subset) =="; in_sif python -m pytest -q -m "not s3 and not downloaded_data" -p no:cacheprovider
echo "== fixture quick tier with both gate branches =="
in_sif python benchmarks/to_modelarray/run_benchmarks.py --cohort benchmarks/to_modelarray/fixtures/nifti-20.csv \
  --mask test/data_voxel_toy/group_mask_FA.nii.gz --scalar-columns FA --tier quick --repetitions 1 \
  --cache-state warm --run-id cubic-00-smoke --result-shard nifti --modelarrayio "$VENV/bin/modelarrayio" --r-gate-command "$GATE"
in_sif python benchmarks/to_modelarray/run_benchmarks.py --cohort benchmarks/to_modelarray/fixtures/cifti.csv --tier quick --repetitions 1 \
  --cache-state warm --run-id cubic-00-smoke --result-shard cifti --modelarrayio "$VENV/bin/modelarrayio" --r-gate-command "$GATE"
echo "== fixture original-vs-optimized (bitwise) =="
in_sif python benchmarks/to_modelarray/compare_revisions.py --cohort benchmarks/to_modelarray/fixtures/cifti.csv \
  --baseline-executable benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd --optimized-executable "$VENV/bin/modelarrayio" \
  --results-dir "$OPT_ROOT/results/cubic-00-fixture-compare" --repetitions 1 --warmups 0 \
  --r-gate-script benchmarks/to_modelarray/r_read_gate.R --gate-command "$GATE" --baseline-revision 9ab13dd
echo "SMOKE PASSED"
