#!/usr/bin/env bash
# Submit steps 0 -> 2 -> 3 -> 4 as a dependency chain (each waits for the previous to succeed).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
D="$HERE/sbatch"
# Jobs write logs/ relative to the submission directory.
cd "$OPT_ROOT" && mkdir -p logs
j0=$(sbatch --propagate=NONE --parsable "$D/00_smoke.sbatch")
j2=$(sbatch --propagate=NONE --parsable --dependency=afterok:$j0 "$D/02_compare_public.sbatch")
j3=$(sbatch --propagate=NONE --parsable --dependency=afterok:$j2 "$D/03_single_scalar.sbatch")
n=$(wc -l < "$OPT_ROOT/cohorts/per-scalar/scalars.txt")
j4=$(sbatch --propagate=NONE --parsable --dependency=afterok:$j3 --array=1-${n}%6 "$D/04_scalar_array.sbatch")
echo "submitted: smoke=$j0 compare=$j2 scalar=$j3 shard=$j4 (array 1-$n)"
