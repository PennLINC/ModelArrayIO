#!/usr/bin/env bash
# One pre-filtered long-format cohort CSV per scalar (all 8,693 subject-sessions),
# which is how a job array selects a scalar without any converter change.
# Usage: make_scalar_cohorts.sh              complete reference scalars; also writes scalars.txt
#        make_scalar_cohorts.sh SCALAR ...   just these cohorts; scalars.txt is left unchanged
# Cohorts are deterministic subsets of $RBC_COHORT, so existing ones are kept; delete one to rebuild it.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$OPT_ROOT/cohorts/per-scalar"
if [ $# -gt 0 ]; then scalars=("$@"); write_list=0; else
  mapfile -t scalars < <(python3 -c "import csv,sys;[print(r['scalar']) for r in csv.DictReader(open('$OPT_ROOT/reference_scalars.csv')) if r['complete']=='True']")
  write_list=1
fi
for s in "${scalars[@]}"; do
  out="$OPT_ROOT/cohorts/per-scalar/$s.csv"
  [ -f "$out" ] && continue
  in_sif python "$REPO/benchmarks/to_modelarray/make_cifti_cohort.py" subset --input "$RBC_COHORT" \
    --output "$out" --subjects 8693 --scalars "$s"
done
if [ "$write_list" = 1 ]; then
  printf '%s\n' "${scalars[@]}" > "$OPT_ROOT/cohorts/per-scalar/scalars.txt"
fi
echo "per-scalar cohorts ready for ${#scalars[@]} scalars"
