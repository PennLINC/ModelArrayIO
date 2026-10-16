#!/usr/bin/env bash
# Guarded step-5 submission: refuse unless every step-4 shard PASSED, then build all scalars.
# scalars.txt stays the list of *complete reference* scalars (used for bitwise checks);
# all_scalars.txt drives the production array.
set -euo pipefail
source "$(dirname "$0")/env.sh"
n_expected=$(wc -l < "$OPT_ROOT/cohorts/per-scalar/scalars.txt")
n_passed=$(ls "$OPT_ROOT"/results/cubic-04-shard/*/PASSED 2>/dev/null | wc -l)
[ "$n_passed" -eq "$n_expected" ] || { echo "step 4: $n_passed/$n_expected shards passed; refusing to submit step 5"; exit 1; }
python3 -c "import csv;print('\n'.join(sorted({r['scalar_name'] for r in csv.DictReader(open('$RBC_COHORT'))})))" \
  > "$OPT_ROOT/cohorts/per-scalar/all_scalars.txt"
# Per-scalar cohorts for every scalar (the step-4 reference scalars already have theirs).
mapfile -t all_scalars < "$OPT_ROOT/cohorts/per-scalar/all_scalars.txt"
bash "$(dirname "$0")/make_scalar_cohorts.sh" "${all_scalars[@]}"
n=$(wc -l < "$OPT_ROOT/cohorts/per-scalar/all_scalars.txt")
echo "free space before production build:"; df -h /cbica/home/tongt | tail -1
sbatch --propagate=NONE --array=1-${n}%10 "$(dirname "$0")/sbatch/05_production.sbatch"
