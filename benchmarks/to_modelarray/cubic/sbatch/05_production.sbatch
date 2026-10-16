#!/usr/bin/env bash
#SBATCH -J ma-prod
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /cbica/home/tongt/modelarray_opt/logs/%x-%A_%a.out
# Full production build: every scalar -> its own HDF5 file (kept). Submit via submit_production.sh.
set -euo pipefail
source /cbica/home/tongt/modelarray_opt/ModelArrayIO/benchmarks/to_modelarray/cubic/env.sh
cd "$REPO"
SCALAR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OPT_ROOT/cohorts/per-scalar/all_scalars.txt")
COHORT="$OPT_ROOT/cohorts/per-scalar/$SCALAR.csv"; OUT="$OPT_ROOT/results/production"; mkdir -p "$OUT/checks"
start=$(date +%s.%N)
in_sif "$VENV/bin/modelarrayio" to-modelarray --cohort-file "$COHORT" --output "$OUT/${SCALAR}_modelarray.h5" \
  --backend hdf5 --workers 8 --dtype float32 --compression gzip --compression-level 4 --target-chunk-mb 32 --no-split-files \
  --timing-jsonl "$OUT/checks/${SCALAR}_timing.jsonl" --log-level WARNING
echo "$SCALAR,$(echo "$(date +%s.%N) - $start" | bc)" >> "$OUT/checks/wall_seconds.csv"
bash benchmarks/to_modelarray/cubic/gate.sh "$OUT/${SCALAR}_modelarray.h5" hdf5 | tee "$OUT/checks/${SCALAR}_gate.log"
if grep -qx "$SCALAR" "$OPT_ROOT/cohorts/per-scalar/scalars.txt"; then
  in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --candidate "$OUT/${SCALAR}_modelarray.h5" --scalar "$SCALAR" --output-csv "$OUT/checks/${SCALAR}_reference_check.csv"
fi
echo "PROD $SCALAR PASSED" | tee "$OUT/checks/${SCALAR}.PASSED"
