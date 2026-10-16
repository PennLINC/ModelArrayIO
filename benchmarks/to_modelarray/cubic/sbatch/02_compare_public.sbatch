#!/usr/bin/env bash
#SBATCH -J ma-compare
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /cbica/home/tongt/modelarray_opt/logs/%x-%j.out
set -euo pipefail
source /cbica/home/tongt/modelarray_opt/ModelArrayIO/benchmarks/to_modelarray/cubic/env.sh
cd "$REPO"
BASE=benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd; OPT=$VENV/bin/modelarrayio
# area areaPial curv: scalars the original production run completed, so every CIFTI output can
# also be checked bitwise against the deployed reference.
CIFTI_COHORT=$OPT_ROOT/cohorts/cifti-rbc-100x3ref.csv
CIFTI_SCALARS=(--scalar area --scalar areaPial --scalar curv)
# Cohort provenance for the report (it cannot tell public, internal and synthetic data apart).
SOURCES=(--cohort-source 'nifti=ABIDE alff, public S3 (make_cohort.py, materialize_cohort.py)'
  --cohort-source 'mif=ModelArray walkthrough FDC, public OSF (make_mif_cohort.py)'
  --cohort-source 'cifti=RBC freesurferpost subset, internal (make_cifti_cohort.py subset)')
COMMON=(--baseline-executable $BASE --optimized-executable $OPT --repetitions 3 --warmups 1 --workers 1 --r-gate-script benchmarks/to_modelarray/r_read_gate.R --gate-command "$GATE" --baseline-revision 9ab13dd)
in_sif python benchmarks/to_modelarray/compare_revisions.py --cohort "$OPT_ROOT/cohorts/abide-100-local.csv" --mask "$OPT_ROOT/data/group_mask.nii.gz" --scalar-columns alff --results-dir "$OPT_ROOT/results/cubic-02-nifti" "${COMMON[@]}"
in_sif python benchmarks/to_modelarray/compare_revisions.py --cohort "$OPT_ROOT/cohorts/mif-walkthrough.csv" --index-file "$OPT_ROOT/data/mif-walkthrough/FDC/index.mif" --directions-file "$OPT_ROOT/data/mif-walkthrough/FDC/directions.mif" --results-dir "$OPT_ROOT/results/cubic-02-mif" "${COMMON[@]}"
in_sif python benchmarks/to_modelarray/compare_revisions.py --cohort "$CIFTI_COHORT" --results-dir "$OPT_ROOT/results/cubic-02-cifti" "${COMMON[@]}"
for d in "$OPT_ROOT"/results/cubic-02-cifti/optimized-*-rep-01; do
  in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --candidate "$d"/modelarray.* "${CIFTI_SCALARS[@]}" --output-csv "$d/reference_check.csv"
done
echo "== raw fast path vs nibabel on production headers (both must equal the reference) =="
for raw in 1 0; do
  MODELARRAYIO_CIFTI_RAW=$raw in_sif python benchmarks/to_modelarray/run_benchmarks.py --cohort "$CIFTI_COHORT" --tier quick --repetitions 1 --cache-state warm \
    --run-id cubic-02-cifti-raw$raw --backends hdf5 --modelarrayio "$OPT" --r-gate-command "$GATE"
  out=$(ls -d "$OPT_ROOT"/results/cubic-02-cifti-raw$raw/outputs/*/modelarray.h5 | head -1)
  in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --candidate "$out" "${CIFTI_SCALARS[@]}" --output-csv "$OPT_ROOT/results/cubic-02-cifti-raw$raw/reference_check.csv"
  # The raw run must read every row through the fast path; a silent fallback shows up here.
  in_sif python - "$OPT_ROOT/results/cubic-02-cifti-raw$raw" <<'PY'
import json, glob, sys
recs = [json.loads(l) for p in glob.glob(sys.argv[1] + '/outputs/*/timing.jsonl') for l in open(p) if l.strip()]
print('fast_path_rows:', [(r['scalar_name'], r.get('fast_path_rows'), r['matrix_shape'][0]) for r in recs if r.get('event') == 'scalar_complete'])
PY
done
echo "== report =="
in_sif python benchmarks/to_modelarray/summarize_comparison.py --run-dir "$OPT_ROOT/results/cubic-02-nifti" --run-dir "$OPT_ROOT/results/cubic-02-mif" --run-dir "$OPT_ROOT/results/cubic-02-cifti" --output "$OPT_ROOT/results/cubic_cross_modality_9ab13dd_vs_optimized.csv"
in_sif python benchmarks/to_modelarray/create_comparison_report.py --run-dir "$OPT_ROOT/results/cubic-02-nifti" --run-dir "$OPT_ROOT/results/cubic-02-mif" --run-dir "$OPT_ROOT/results/cubic-02-cifti" --platform "CUBIC compute node ($(nproc) CPU allocation)" "${SOURCES[@]}" --output "$OPT_ROOT/results/cubic-cross-modality-benchmark.pdf"
echo "COMPARE PASSED"
