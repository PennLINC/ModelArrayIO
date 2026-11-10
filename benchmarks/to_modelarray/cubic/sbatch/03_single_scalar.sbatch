#!/usr/bin/env bash
#SBATCH -J ma-scalar
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# One full production scalar (8,693 subjects): old vs new (1 rep), then a worker sweep.
set -euo pipefail
source "${REPO:-${OPT_ROOT:?export OPT_ROOT before sbatch (see cubic/env.sh)}/ModelArrayIO}/benchmarks/to_modelarray/cubic/env.sh"
cd "$REPO"
SCALAR=${SCALAR:-area}; COHORT="$OPT_ROOT/cohorts/per-scalar/$SCALAR.csv"
BASE=benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd; OPT=$VENV/bin/modelarrayio
echo "== prime page cache ($SCALAR) =="; cut -d, -f3 "$COHORT" | tail -n +2 | tr -d "\r" | xargs -P 8 -n 200 cat > /dev/null
echo "== old vs new, 1 rep, chunk target 32 MiB =="
in_sif python benchmarks/to_modelarray/compare_revisions.py --cohort "$COHORT" --results-dir "$OPT_ROOT/results/cubic-03-$SCALAR-compare" \
  --baseline-executable $BASE --optimized-executable $OPT --repetitions 1 --warmups 0 --workers 1 --target-chunk-mb 32 \
  --r-gate-script benchmarks/to_modelarray/r_read_gate.R --gate-command "$GATE" --baseline-revision 9ab13dd
echo "== optimized worker sweep =="
in_sif python benchmarks/to_modelarray/run_benchmarks.py --cohort "$COHORT" --tier quick --workers 1 4 8 --target-chunk-mb 32 \
  --backends hdf5 tiledb --repetitions 1 --cache-state warm --run-id cubic-03-$SCALAR-sweep --modelarrayio "$OPT" --r-gate-command "$GATE"
echo "== bitwise vs the deployed reference for every optimized output =="
for out in "$OPT_ROOT"/results/cubic-03-$SCALAR-compare/optimized-*-rep-01/modelarray.* "$OPT_ROOT"/results/cubic-03-$SCALAR-sweep/outputs/*/modelarray.*; do
  in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --candidate "$out" --scalar "$SCALAR" --output-csv "$out.reference_check.csv"
done
echo "== free space: drop bulky outputs, keep logs/timing/checks =="
find "$OPT_ROOT/results/cubic-03-$SCALAR-sweep/outputs" -maxdepth 2 \( -name 'modelarray.h5' -o -name 'modelarray.tdb' \) -exec rm -rf {} +
echo "SCALAR PASSED"
