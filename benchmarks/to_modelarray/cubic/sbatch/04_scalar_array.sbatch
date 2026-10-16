#!/usr/bin/env bash
#SBATCH -J ma-shard
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o /cbica/home/tongt/modelarray_opt/logs/%x-%A_%a.out
# Scalar-sharded conversion: one task per scalar, its own per-scalar file, no aggregation.
# Submit with: sbatch --array=1-$(wc -l < .../per-scalar/scalars.txt) 04_scalar_array.sbatch
set -euo pipefail
source /cbica/home/tongt/modelarray_opt/ModelArrayIO/benchmarks/to_modelarray/cubic/env.sh
cd "$REPO"
SCALAR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OPT_ROOT/cohorts/per-scalar/scalars.txt")
COHORT="$OPT_ROOT/cohorts/per-scalar/$SCALAR.csv"; OUT="$OPT_ROOT/results/cubic-04-shard/$SCALAR"; mkdir -p "$OUT"
KEEP=${KEEP_OUTPUTS:-0}
for backend in hdf5 tiledb; do
  suffix=$([ $backend = hdf5 ] && echo h5 || echo tdb)
  comp=$([ $backend = hdf5 ] && echo "gzip --compression-level 4" || echo "zstd --compression-level 1")
  start=$(date +%s.%N)
  in_sif "$VENV/bin/modelarrayio" to-modelarray --cohort-file "$COHORT" --output "$OUT/${SCALAR}_modelarray.$suffix" \
    --backend $backend --workers 8 --dtype float32 --compression $comp --target-chunk-mb 32 --no-split-files \
    --timing-jsonl "$OUT/timing_$backend.jsonl" --log-level WARNING
  echo "wall_seconds,$backend,$(echo "$(date +%s.%N) - $start" | bc)" >> "$OUT/wall.csv"
  bash benchmarks/to_modelarray/cubic/gate.sh "$OUT/${SCALAR}_modelarray.$suffix" $backend | tee -a "$OUT/gate_$backend.log"
  in_sif python benchmarks/to_modelarray/compare_to_reference.py --reference "$REF_TDB" --candidate "$OUT/${SCALAR}_modelarray.$suffix" --scalar "$SCALAR" --output-csv "$OUT/reference_check_$backend.csv"
  ls "$OUT"/.*modelarrayio-tmp-* "$OUT"/.*modelarrayio-backup-* 2>/dev/null && { echo "leftover temporaries!"; exit 1; } || true
  [ "$KEEP" = 1 ] || rm -rf "$OUT/${SCALAR}_modelarray.$suffix"
done
echo "SHARD $SCALAR PASSED" | tee "$OUT/PASSED"
