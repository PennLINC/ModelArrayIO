# CUBIC validation ladder for `to-modelarray`

Runs the optimized converter on the CUBIC cluster against the original code (commit
`9ab13dd`) and against the production CIFTI cohort, with a correctness gate at every step.
Everything writable lives under `$OPT_ROOT`; the production cohort (`$RBC_COHORT`) and its
old-code reference output (`$REF_TDB`) are only read, in place.

## Configuration

No site-specific paths are committed. Export these on the cluster (for example in `~/.bashrc`)
before running any step; `sbatch` passes them on to every job:

```bash
export OPT_ROOT=/path/to/writable/work/dir                # checkout, venv, caches, results, logs
export SIF=/path/to/modelarray_confixel_0.1.5.sif         # ModelArray container image
export REF_TDB=/path/to/reference/rbc_freesurferpost.tdb  # deployed reference output (read only)
export RBC_COHORT=/path/to/rbc_freesurferpost_cohort.csv  # production cohort CSV (read only)
export DATA_BINDS=/path/to/project,/path/to/alternate/mount  # read-only dirs to bind (optional)
```

`DATA_BINDS` must cover the reference output and every mount path that the cohort CSV uses for its
source files. `env.sh` stops with a message naming any required variable that is missing. On the
local machine, set `CUBIC_HOST` to your `user@login-node` and `OPT_ROOT` to the same value as on
the cluster.

## Environment route

The ModelArray container (`$SIF`, `modelarray_confixel` 0.1.5, Ubuntu 20.04, glibc 2.31)
runs *everything*: a uv-managed Python 3.12 venv created inside it installs the exact local
library versions as `manylinux_2_28` wheels (the CentOS 7 host, glibc 2.17, cannot), and its
R ModelArray/rhdf5/HDF5Array gate HDF5 outputs natively. TileDB outputs are gated in Python
(`benchmarks/to_modelarray/tiledb_read_gate.py`) because the container has no R `tiledb`. `env.sh` defines
`in_sif`, which runs a command in the container — or directly when already inside one, so the
gate dispatcher works from within a containerised job without nesting.

## One-time setup (login node; it has internet, compute nodes may not)

```bash
# local machine: copy the working tree (with .git, for the baseline export) — no commit needed
rsync -a --delete --exclude .venv --exclude 'benchmarks/*/data' --exclude 'benchmarks/*/results' \
  --exclude 'benchmarks/*/cohorts' --exclude .DS_Store \
  ModelArrayIO/ "$CUBIC_HOST:$OPT_ROOT/ModelArrayIO/"
python benchmarks/to_modelarray/source_identity.py          # note source_tree_sha256

# local machine: the current ModelArray R source, for the HDF5 gate
rsync -a --delete --exclude .git --exclude docs ModelArray/ "$CUBIC_HOST:$OPT_ROOT/ModelArray/"

# cubic login node (jobs write logs/ relative to the directory they are submitted from)
CUBIC=$OPT_ROOT/ModelArrayIO/benchmarks/to_modelarray/cubic
bash "$CUBIC/setup_env.sh"
cd "$OPT_ROOT" && sbatch --propagate=NONE "$CUBIC/sbatch/00_smoke.sbatch"   # also writes reference_scalars.csv
bash "$CUBIC/make_scalar_cohorts.sh"                                       # after the smoke job
```

`setup_env.sh` prints the container's `source_tree_sha256`; it must equal the local one — this
is the rsync-fidelity gate, and every result file records it.

## The ladder

| Step | Job | Gate |
|---|---|---|
| 0 | `sbatch/00_smoke.sbatch` (short) | imports on a compute node, offline pytest, quick-tier fixtures through both gate branches, fixture old-vs-new bitwise |
| 2 | `sbatch/02_compare_public.sbatch` (all, exclusive) | old-vs-new on ABIDE-100, MIF walkthrough-100, RBC CIFTI 100×3: bitwise + gates; raw fast path vs nibabel both equal the deployed reference; comparison report (PDF, not committed) |
| 3 | `sbatch/03_single_scalar.sbatch` (all, exclusive) | one full scalar, 8,693 subjects: old-vs-new (1 rep), worker sweep, every output bitwise vs `rbc_freesurferpost.tdb` |
| 4 | `sbatch/04_scalar_array.sbatch` `--array` | one task per complete reference scalar, per-scalar file, bitwise vs reference, no aggregation |
| 5 | `submit_production.sh` → `05_production.sbatch` `--array` | all scalars → kept per-scalar HDF5; runs only when every step-4 shard PASSED |

`submit_ladder.sh` chains steps 0→2→3→4 with `--dependency=afterok`.

## Pinning the results

Step 2 writes the cross-modality summary CSV and an interim PDF without the production-scale
section. Once steps 3–5 have passed, build the production-scale CSV and the final report.
`summarize_scale.py` reads the step-3 comparison and worker sweep, the step-4 shard walls and the
step-5 production walls (it skips any step not yet run). The report opens one TileDB output per
backend, so run both commands on a compute node, not the login node:

```bash
# cubic login node: open a shell on a compute node
srun -p short -c 2 --mem=8G -t 1:00:00 --propagate=NONE --pty bash

source "$OPT_ROOT/ModelArrayIO/benchmarks/to_modelarray/cubic/env.sh"
cd "$REPO"
in_sif python benchmarks/to_modelarray/summarize_scale.py --results-root "$OPT_ROOT/results" \
  --scalar area --output "$OPT_ROOT/results/cubic_production_scale_9ab13dd_vs_optimized.csv"
# The note is the report's hand-written "Production scale" summary; this is the text of the pinned
# 2026-09-22 report. Rewrite it from the new numbers whenever the ladder is re-run.
NOTE='The original production run (code/02_create_tiledb.sh, --workers 15) needed 112 h 23 m for its first scalar and stalled at 15 of 70 scalars after about 119 h. On the same data, the optimized converter builds one full 8,693-subject scalar in about 7 minutes at one loader worker and in 69-91 seconds at eight, and the scalar-sharded job array converted the 15 reference scalars in 21 minutes of wall-clock time, every output bitwise-equal to the deployed reference. The full production build - all 70 scalars, 608,510 files, 484 GB of per-scalar HDF5 - completed in about 22 minutes of job-array wall-clock (median 138 s per scalar), 70 of 70 gated.'
in_sif python benchmarks/to_modelarray/create_comparison_report.py \
  --run-dir "$OPT_ROOT/results/cubic-02-nifti" --run-dir "$OPT_ROOT/results/cubic-02-mif" \
  --run-dir "$OPT_ROOT/results/cubic-02-cifti" \
  --scale-csv "$OPT_ROOT/results/cubic_production_scale_9ab13dd_vs_optimized.csv" \
  --original-run-note "$NOTE" --measured-at 2026-09-22 \
  --platform 'CUBIC compute nodes (Xeon Gold 6130, 8-CPU allocations, GPFS)' \
  --cohort-source 'nifti=ABIDE alff, public S3 (make_cohort.py, materialize_cohort.py)' \
  --cohort-source 'mif=ModelArray walkthrough FDC, public OSF (make_mif_cohort.py)' \
  --cohort-source 'cifti=RBC freesurferpost subset, internal (make_cifti_cohort.py subset)' \
  --output "$OPT_ROOT/results/cubic-cross-modality-benchmark.pdf"
```

Then copy the two CSVs into the repository, naming each with its measurement date. Replace the
previous pinned files rather than keeping both, and update the links and numbers in
[`../README.md`](../README.md#cubic-hpc-validation). The PDF is not committed: it is regenerated
from the pinned CSVs with `create_comparison_report.py`, so keep it with the run's results or
attach it to the pull request.

```bash
# local machine, from the ModelArrayIO root
R="$CUBIC_HOST:$OPT_ROOT/results"
scp "$R/cubic_cross_modality_9ab13dd_vs_optimized.csv" \
  benchmarks/to_modelarray/baselines/cubic_cross_modality_9ab13dd_vs_optimized_YYYYMMDD.csv
scp "$R/cubic_production_scale_9ab13dd_vs_optimized.csv" \
  benchmarks/to_modelarray/baselines/cubic_production_scale_9ab13dd_vs_optimized_YYYYMMDD.csv
```

## Why this is the HPC pattern

ModelArray's `element-splitting` shards the *element* axis across jobs and stitches chunk files
back together because HDF5 cannot be written concurrently. Building the matrix inverts that:
inputs are whole per-subject files (gzip is not seekable, so element shards would re-decode every
subject), while each **scalar's** output is already a separate file. So the job array shards by
scalar and there is nothing to aggregate. Within a task, `--workers` threads decode subjects
concurrently and the HDF5 compression pool / TileDB internal pool compress in parallel.

Production-scale runs use `--target-chunk-mb 32` (the original run's tile geometry) and `--mem=48G`
(an 11.4 GB float32 matrix plus TileDB write buffers); calibrate from `peak_memory_bytes` in step 3.

## Defects the ladder caught (and how the scripts now avoid them)

- **Stale reader in the container.** ModelArray 0.1.5 reads `column_names` from an HDF5
  *attribute*; every converter version writes the dataset, so 0.1.5 rejects old and new outputs
  alike. The HDF5 gate uses the current ModelArray installed into `$OPT_ROOT/Rlib`.
- **CRLF cohort files.** `csv.DictWriter`'s default `\r\n` broke shell consumers such as the
  cache-priming `xargs cat` (`path\r` → "No such file"). All cohort writers now emit `\n`; the
  priming step also strips `\r` defensively.
- **CPU-time rlimit.** The login node's 2 h `ulimit -t` propagates into jobs
  (`PropagateResourceLimits=ALL`) and killed the original converter with SIGXCPU two-thirds of the
  way through a full scalar (it needs ~3 h CPU). Every submission uses `sbatch --propagate=NONE`
  (probe-verified unlimited inside jobs); `env.sh` also raises the soft limit to the hard limit.
- **Login node cannot open TileDB arrays** (Ivy Bridge, no AVX2 → SIGILL in libtiledb). Anything
  that touches an array runs in a SLURM job; compute nodes are Xeon Gold with AVX-512.
- **Silent fast-path fallback.** Production dscalars carry per-subject provenance in the CIFTI
  XML, shifting `vox_offset`; the exact-size check rejected 91/100 files. The fast path now
  verifies the on-disk dims, dtype and (absent) scaling plus a digest of every non-scalar
  `MatrixIndicesMap`, so the offset may differ; watch `fast_path_rows` in `timing.jsonl`.
