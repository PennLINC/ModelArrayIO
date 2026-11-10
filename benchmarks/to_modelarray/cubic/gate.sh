#!/usr/bin/env bash
# Read-gate dispatcher: `gate.sh OUTPUT BACKEND`.
# hdf5  -> R ModelArray/rhdf5/HDF5Array inside the container (r_read_gate.R)
# tiledb -> Python tiledb-py gate (the container has no R tiledb package)
set -euo pipefail
source "$(dirname "$0")/env.sh"
output=$1; backend=$2
case "$backend" in
  hdf5)   in_sif Rscript "$REPO/benchmarks/to_modelarray/r_read_gate.R" "$output" hdf5 ;;
  tiledb) in_sif python "$REPO/benchmarks/to_modelarray/tiledb_read_gate.py" "$output" tiledb ;;
  *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac
