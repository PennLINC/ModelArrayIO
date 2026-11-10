# `to-modelarray` benchmarks

This folder benchmarks the complete CLI conversion. Results stay under
`benchmarks/to_modelarray/results/<run-id>/` (or `$MODELARRAYIO_BENCHMARK_RESULTS_DIR/<run-id>/`)
and are intentionally not deleted: correctness comparisons and the R read gate need the
produced arrays.

## Cohorts

- `fixture`: the committed NIfTI and CIFTI test data, used only for correctness and smoke runs.
- `small`: 100 public ABIDE subjects and one derivative.
- `medium`: 1,102 ABIDE subjects and one derivative (`alff`).
- `wide`: 1,102 subjects and nine derivatives (`alff`, `falff`, `reho`, `vmhc`, `lfcd`,
  `degree_binarize`, `degree_weighted`, `eigenvector_binarize`, and
  `eigenvector_weighted`).
- `production confirmation`: internal data, run only for the final candidates.

The tracked fixture manifests are `fixtures/nifti-20.csv` and `fixtures/cifti.csv`. ABIDE is read
anonymously from `s3://fcp-indi`. Generate every public ABIDE cohort as follows:

```bash
# Small: 100 subjects x 1 derivative
python benchmarks/to_modelarray/make_cohort.py \
  --subjects 100 \
  --derivatives alff \
  --repeat-factor 1 \
  --output benchmarks/to_modelarray/cohorts/abide-small.csv

# Medium: 1,102 subjects x 1 derivative
python benchmarks/to_modelarray/make_cohort.py \
  --subjects 1102 \
  --derivatives alff \
  --repeat-factor 1 \
  --output benchmarks/to_modelarray/cohorts/abide-medium.csv

# Wide: 1,102 subjects x 9 derivatives
python benchmarks/to_modelarray/make_cohort.py \
  --subjects 1102 \
  --derivatives alff falff reho vmhc lfcd degree_binarize degree_weighted \
    eigenvector_binarize eigenvector_weighted \
  --repeat-factor 1 \
  --output benchmarks/to_modelarray/cohorts/abide-wide.csv
```

The generator lists public keys, intersects subjects having every requested derivative and
`func_mask`, sorts the subject IDs, and takes the first requested count. Pass
`--subject-manifest subjects.txt` to use a frozen newline-delimited subject list without an S3
listing. Repeats reuse the real source URIs and receive unique cohort IDs; they are valid for
decode-throughput tests but not compression-ratio conclusions.

Use one downloaded mask as the group mask. For example, download the first mask in the generated
small cohort with the AWS CLI:

```bash
mkdir -p benchmarks/to_modelarray/data
aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_mask/CMU_a_0050642_func_mask.nii.gz \
  benchmarks/to_modelarray/data/group_mask.nii.gz \
  --no-sign-request
```

`cohorts/`, `data/` and `results/` hold generated or downloaded files and are ignored by Git (see
[`../README.md`](../README.md#suite-layout)). Set `MODELARRAYIO_S3_ANON=1` for manual CLI runs;
the harness sets it when absent.

### MIF (fixel) cohorts

MIF has a fully public, reproducible cohort: the ModelArray walkthrough FDC demo (100 fixel
`.mif` subjects with 602,229 fixels each, plus `index.mif` and `directions.mif`), hosted on OSF.
`make_mif_cohort.py` downloads and extracts it (once) and writes a normalized long-format
`subject_id,scalar_name,source_file` cohort with absolute paths:

```bash
# Downloads ~212 MiB from https://osf.io/tce9d/download into benchmarks/to_modelarray/data/mif-walkthrough/
python benchmarks/to_modelarray/make_mif_cohort.py \
  --data-dir benchmarks/to_modelarray/data/mif-walkthrough \
  --output benchmarks/to_modelarray/cohorts/mif-walkthrough.csv
```

Run it on both backends with the R gate; MIF needs `--index-file` and `--directions-file` and no
`--mask`:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/mif-walkthrough.csv \
  --index-file benchmarks/to_modelarray/data/mif-walkthrough/FDC/index.mif \
  --directions-file benchmarks/to_modelarray/data/mif-walkthrough/FDC/directions.mif \
  --tier quick --workers 1 4 --dtypes float32 \
  --backends hdf5 tiledb --repetitions 3 --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

This is real diffusion fixel data, so its runtime, `--workers` scaling and compression ratios are
all quotable — the MIF analogue of the public ABIDE NIfTI cohort.

### CIFTI cohorts

ABIDE has no public CIFTI equivalent, so the CIFTI path is benchmarked with `make_cifti_cohort.py`,
which emits a long-format `subject_id,scalar_name,source_file` cohort the harness runs with combined
output (no `--mask`, no `--scalar-columns`).

CIFTI matters because its bottleneck is unlike NIfTI's: uncompressed `.dscalar.nii` files carry a
large CIFTI XML header that nibabel parses at roughly one second per file while holding the GIL, so
thread workers cannot speed it up. The loader's raw greyordinate fast path
(`MODELARRAYIO_CIFTI_RAW`, on by default) reads the data block directly and sidesteps that parse;
benchmark both settings to measure it.

The generator has three modes:

```bash
# 1. Default synthesize: reproducible anywhere with no real data. Uses scalar-appropriate marginal
#    ranges and a medial-wall zero block, so it is valid for load/runtime timing but NOT compression
#    ratios (its per-vertex values carry no real spatial autocorrelation).
python benchmarks/to_modelarray/make_cifti_cohort.py synthesize \
  --output-dir benchmarks/to_modelarray/data/cifti-synth \
  --subjects 40 --scalars thickness area sulc \
  --grayordinates 327684 --seed 0 \
  --cohort benchmarks/to_modelarray/cohorts/cifti-synth.csv

# 2. Reference-profile synthesize: derives the medial-wall mask and per-vertex mean/std from a
#    small set of real dscalars and draws fresh noise around the real mean map, so compression is
#    realistic (measured 1.38x vs 1.41x for the real reference) while no real subject row is copied.
python benchmarks/to_modelarray/make_cifti_cohort.py synthesize \
  --output-dir benchmarks/to_modelarray/data/cifti-synth-real \
  --subjects 40 --scalars thickness area sulc \
  --reference-dir /path/to/real/dscalars --seed 0 \
  --cohort benchmarks/to_modelarray/cohorts/cifti-synth-real.csv

# 3. Subset: slice an existing long-format internal CIFTI cohort without copying files.
python benchmarks/to_modelarray/make_cifti_cohort.py subset \
  --input /path/to/rbc_freesurferpost_cohort.csv \
  --output benchmarks/to_modelarray/cohorts/cifti-real-small.csv \
  --subjects 100 --scalars thickness area sulc

# Run it (fast path on vs. off) to quantify the raw-read win.
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/cifti-synth-real.csv \
  --tier quick --backends tiledb hdf5 \
  --cache-state warm --run-id cifti-synth-raw \
  --modelarrayio .venv/bin/modelarrayio
MODELARRAYIO_CIFTI_RAW=0 python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/cifti-synth-real.csv \
  --tier quick --backends tiledb hdf5 \
  --cache-state warm --run-id cifti-synth-nibabel \
  --modelarrayio .venv/bin/modelarrayio
```

The `subset` mode reads any long-format CIFTI cohort, keeps the first `--subjects` subjects (by
first appearance, using `subject_id` and `session_id` when present) for the requested scalars, and
never lists or copies files, so it is safe to point at a large internal cohort CSV. The
`--reference-dir` mode reads only aggregate per-vertex statistics, never an individual subject row.

## Running

The cohort controls input size and scalar count; `--tier` independently controls the parameter
grid. The complete grids are:

| Tier | Loader workers | Dtypes | Chunk targets (MiB) | Codecs | Cases/repetition |
| --- | --- | --- | --- | --- | ---: |
| `quick` | 1 | float32 | 2 | HDF5 gzip-4; TileDB zstd-5 | 2 |
| `medium` | 1, 4, 8 | float32, float64 | 2 | HDF5 gzip-4; TileDB zstd-5 | 12 |
| `full` | 1, 2, 4, 8, allocated CPUs | float32, float64 | 8, 32, 64, 128 | HDF5 gzip-1/gzip-4/none; TileDB zstd-1/zstd-3/zstd-5/none | hardware-dependent |

The full tier also sweeps HDF5 gzip compression workers over 1, 4, and allocated CPUs. If `W` is
the number of distinct loader-worker settings and `P` is the number of distinct compression-worker
settings, it runs `8 * W * (2 * P + 5)` cases per repetition: 440 cases on a 14-CPU host. Three
repetitions therefore launch 1,320 conversions before accounting for multiple scalars. Use it only
for an intentional overnight or cluster sweep.

### Full validation suite

Run the complete Python regression and runtime-integration suite before benchmarking:

```bash
python -m pytest -q
ruff check src test benchmarks docs/examples
ruff format --check src test benchmarks docs/examples
Rscript -e "invisible(parse(file='benchmarks/to_modelarray/r_read_gate.R')); invisible(parse(file='benchmarks/to_modelarray/plot_results.R'))"
```

Run every benchmark parameter combination once on the committed NIfTI fixture. This is an
exhaustive correctness/configuration test, not a throughput result:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/fixtures/nifti-20.csv \
  --mask test/data_voxel_toy/group_mask_FA.nii.gz \
  --scalar-columns FA \
  --tier full \
  --repetitions 1 \
  --cache-state warm \
  --run-id fixture-nifti-full \
  --modelarrayio .venv/bin/modelarrayio \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

On the 14-CPU reference host this is 440 cases: 280 HDF5 and 160 TileDB. The 2026-09-19 run
passed all 440 converter and R-gate checks; its tracked audit is
[`baselines/full_fixture_validation_20260919.csv`](baselines/full_fixture_validation_20260919.csv).
The run was split into worker-specific shards to execute independent R gates concurrently. To do
the same, launch the command once per worker in `1 2 4 8 14`, add `--workers WORKER`, and use a
unique `--result-shard worker-WORKER` with a shared run ID.

Wide cohorts split by default and long-format cohorts combine, matching the converter. The
explicit `--split-files` and `--no-split-files` flags remain available for compatibility tests.
Every reportable command below includes the R read gate.

### Fixture smoke and correctness

Run both committed fixture modalities before public or production sweeps:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/fixtures/nifti-20.csv \
  --mask test/data_voxel_toy/group_mask_FA.nii.gz \
  --scalar-columns FA \
  --tier quick \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/fixtures/cifti.csv \
  --tier quick \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

### Small iteration case

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/abide-small.csv \
  --mask benchmarks/to_modelarray/data/group_mask.nii.gz \
  --scalar-columns alff \
  --tier quick \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

### Medium comparison case

The medium tier is the default public performance comparison. It runs both backends at loader
worker counts 1, 4, and 8 and both storage dtypes:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/abide-medium.csv \
  --mask benchmarks/to_modelarray/data/group_mask.nii.gz \
  --scalar-columns alff \
  --tier medium \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

### Wide scalar case

First validate the nine-scalar path with the quick tier:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/abide-wide.csv \
  --mask benchmarks/to_modelarray/data/group_mask.nii.gz \
  --scalar-columns alff falff reho vmhc lfcd degree_binarize degree_weighted \
    eigenvector_binarize eigenvector_weighted \
  --tier quick \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

Run the exhaustive grid only when its storage and scheduling cost is acceptable:

```bash
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort benchmarks/to_modelarray/cohorts/abide-wide.csv \
  --mask benchmarks/to_modelarray/data/group_mask.nii.gz \
  --scalar-columns alff falff reho vmhc lfcd degree_binarize degree_weighted \
    eigenvector_binarize eigenvector_weighted \
  --tier full \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

Use `--backends`, `--workers`, and `--result-shard` to divide a full sweep into explicit,
non-overlapping jobs. A shard label isolates artifacts but does not partition the grid for you.

### Production confirmation case

Production data are site-specific. Set these two paths, then run the final public-cohort settings
on the internal cohort. This template uses the quick-tier codecs with an explicit finalist worker
count and dtype:

```bash
COHORT_CSV=/absolute/path/to/production-cohort.csv
GROUP_MASK=/absolute/path/to/production-group-mask.nii.gz
python benchmarks/to_modelarray/run_benchmarks.py \
  --cohort "$COHORT_CSV" \
  --mask "$GROUP_MASK" \
  --scalar-columns alff \
  --tier quick \
  --workers 8 \
  --dtypes float32 \
  --repetitions 3 \
  --cache-state warm \
  --r-gate-command 'Rscript benchmarks/to_modelarray/r_read_gate.R {output} {backend}'
```

For CIFTI cohorts, omit `--mask`. For MIF cohorts, omit `--mask` and pass
`--index-file` plus `--directions-file` instead. Long-format cohorts record
their actual `scalar_name` values in result rows and R-gate commands.

Important comparisons must use the same cohort, cache state, filesystem, and empty case output.
The harness records `--cache-state` as metadata but does not warm or clear caches itself.
Run at least three repetitions and compare medians. `--result-shard node-01` writes
`results.node-01.csv`; its outputs, logs, timing, and metadata are also stored below a private
`shard-node-01` directory, so parallel workers cannot overwrite one another. Reusing the same
run ID with the same configuration skips completed case keys and writes an interrupted case to a
new retry directory, preserving all prior artifacts. A changed configuration requires a new run
ID. The plotting script merges all shard result files. `run_meta.<shard>.json` records commands,
versions, the cohort hash, CPU/RAM allocation, and TileDB concurrency settings. Each result row includes phase counters, output/file/fragment
counts, input bytes, dtype, codec, loader workers, compression workers, and resource usage.
Before a production sweep, set an absolute stopping rule with
`--acceptance-files-per-minute`; a case below that target makes the harness exit nonzero. Use
`--compression-workers 1 2 4 8` to sweep the internal HDF5 gzip pool without adding another
public converter option.

Runs without an R gate are recorded as `unverified`, remain useful as smoke tests, and are
excluded from the plotting script's candidate set. A failed gate marks the case failed and makes
the sweep exit nonzero. The default two-backend commands require the R packages `rhdf5`,
`HDF5Array`, `ModelArray`, and `tiledb`.

For HDF5, the gate constructs a `ModelArray` through `rhdf5`/`HDF5Array` and reads a representative
slice. For TileDB, it opens and reads the array through the R `tiledb` package. A failed gate makes
the sweep fail. Deployment environments should extend the TileDB branch to construct the local
ModelArray TileDB adapter when one is available.

Create plots with:

```bash
RUN_DIR=$(ls -dt benchmarks/to_modelarray/results/*/ | head -n 1)
Rscript benchmarks/to_modelarray/plot_results.R "$RUN_DIR"
```

## CUBIC (HPC) validation

The cluster ladder — container-based environment, SLURM job templates, the old-code reference
comparison, and the scalar-sharded job-array recipe — lives in [`cubic/`](cubic/README.md).

Pinned CUBIC results (original `9ab13dd` vs optimized, real RBC freesurferpost data):
[`baselines/cubic_cross_modality_9ab13dd_vs_optimized_20260922.csv`](baselines/cubic_cross_modality_9ab13dd_vs_optimized_20260922.csv)
(step 2: NIfTI 1.40×/1.46×, MIF 3.70×/4.11×, CIFTI 31.8×/31.6×) and
[`baselines/cubic_production_scale_9ab13dd_vs_optimized_20260923.csv`](baselines/cubic_production_scale_9ab13dd_vs_optimized_20260923.csv)
(one full 8,693-subject scalar 27.6×/27.5× at one worker, 69–91 s at eight; 15-scalar job array in
21 min; all 70 production scalars in ≈22 min of array wall-clock, every output bitwise-equal to the
deployed reference). The CUBIC report PDF is not committed; regenerate it from these CSVs with
`create_comparison_report.py` (see [`cubic/README.md`](cubic/README.md#pinning-the-results)).

## Original versus optimized comparison

`compare_revisions.py` runs two `modelarrayio` executables against the same materialized cohort,
inputs, environment, backend settings, and warm-cache policy. It infers the modality from the
cohort (so it drives NIfTI, CIFTI and MIF), discards one warm-up per implementation and backend,
counterbalances the remaining runs, applies the R gate, and compares every optimized output with
the baseline for shape, dtype, source order, NaN placement, and values.

The original code is pinned to commit `9ab13dd` (the last commit before the optimization branch),
exported with `git archive` into an isolated cache and run under the current checkout's Python
environment by `benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd`. Override the cache or
interpreter with `MODELARRAYIO_BASELINE_ROOT` or `MODELARRAYIO_PYTHON`.

The reportable result is the CUBIC run above (step 2 of the ladder, on real CIFTI). The commands
below reproduce the same comparison from public data on any machine; CIFTI uses the realistic
synthesizer there because no real CIFTI cohort is public. Run one comparison per modality (each
writes its own results directory):

```bash
BASELINE=benchmarks/to_modelarray/baselines/modelarrayio-baseline-9ab13dd
OPTIMIZED=.venv/bin/modelarrayio
GATE=benchmarks/to_modelarray/r_read_gate.R

# NIfTI (100 ABIDE subjects)
python benchmarks/to_modelarray/compare_revisions.py \
  --cohort benchmarks/to_modelarray/cohorts/abide-nifti100-local.csv \
  --mask benchmarks/to_modelarray/data/group_mask.nii.gz --scalar-columns alff \
  --baseline-executable "$BASELINE" --optimized-executable "$OPTIMIZED" \
  --results-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-nifti \
  --repetitions 3 --warmups 1 --r-gate-script "$GATE" --baseline-revision 9ab13dd

# MIF (100 walkthrough FDC subjects)
python benchmarks/to_modelarray/compare_revisions.py \
  --cohort benchmarks/to_modelarray/cohorts/mif-walkthrough.csv \
  --index-file benchmarks/to_modelarray/data/mif-walkthrough/FDC/index.mif \
  --directions-file benchmarks/to_modelarray/data/mif-walkthrough/FDC/directions.mif \
  --baseline-executable "$BASELINE" --optimized-executable "$OPTIMIZED" \
  --results-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-mif \
  --repetitions 3 --warmups 1 --r-gate-script "$GATE" --baseline-revision 9ab13dd

# CIFTI (20 realistic-synth subjects x 3 scalars)
python benchmarks/to_modelarray/compare_revisions.py \
  --cohort benchmarks/to_modelarray/cohorts/cifti-cmp20.csv \
  --baseline-executable "$BASELINE" --optimized-executable "$OPTIMIZED" \
  --results-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-cifti \
  --repetitions 3 --warmups 1 --r-gate-script "$GATE" --baseline-revision 9ab13dd
```

Then render the cross-modality findings PDF from the three run directories. It is written to
`benchmarks/to_modelarray/results/` by default and is not committed:

```bash
python benchmarks/to_modelarray/create_comparison_report.py \
  --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-nifti \
  --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-mif \
  --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-cifti \
  --cohort-source 'nifti=ABIDE alff, public S3 (make_cohort.py, materialize_cohort.py)' \
  --cohort-source 'mif=ModelArray walkthrough FDC, public OSF (make_mif_cohort.py)' \
  --cohort-source 'cifti=synthetic, real reference profile (make_cifti_cohort.py --reference-dir)'
```

The report shows the original-vs-optimized median wall time, speedup, downstream R-read time,
output file count and bitwise correctness for every modality on both backends. Its text comes from
the run metadata (date, platform, repetitions, warm-ups, workers, chunk target, cohort file and
hash); `--cohort-source` adds where each cohort came from, which the metadata cannot tell. It needs ReportLab,
which is not a declared dependency.

## What the results show about the design

The optimized design and the alternatives it rejects are described in
[`design/TO_MODELARRAY_RUNTIME_OPTIMIZATION_PLAN.md`](../../design/TO_MODELARRAY_RUNTIME_OPTIMIZATION_PLAN.md).
The benchmarks measure its two main choices:

- **One dense TileDB write per scalar** instead of one fragment per stripe: on the CUBIC comparison
  this reduced the output from 350 files to 8 for MIF and from 596 to 38 for CIFTI while preserving
  values and metadata.
- **Concurrent loader workers.** Loader-worker scaling is pinned by the CUBIC step-3 sweep in
  [`baselines/cubic_production_scale_9ab13dd_vs_optimized_20260923.csv`](baselines/cubic_production_scale_9ab13dd_vs_optimized_20260923.csv):
  one full production CIFTI scalar (8,693 subjects), one repetition per setting. Workers 1/4/8
  measured 403.9/120.9/90.6 seconds for HDF5 and 186.3/83.2/68.6 seconds for TileDB, while peak
  memory stayed flat (about 12 GB for HDF5 and 20 GB for TileDB) because it is dominated by the
  staging matrix. Other modalities were not swept at production scale, so the CLI keeps the
  conservative default of one worker; each production workload should validate `--workers 4` or
  `--workers 8` with the medium tier and multiple repetitions.

## Interpretation and limits

The optimization target is end-to-end elapsed time while preserving values, NaN locations,
source order, metadata, and R readability. `scheduler_wait_seconds` and the in-flight
source counters describe the bounded loader itself. HDF5 `compression_workers` is an internal runtime policy;
`--workers` controls only concurrent input loads. TileDB's compute and I/O pools are recorded and
never capped to `--workers`.

`baselines/baseline_fixture.csv` pins three-run medians from the committed 20-subject fixture.
It is a schema/size and gross-regression reference, not a production throughput target. The
reportable original-versus-optimized results are the CUBIC CSVs above: real data, the original
source tree at commit `9ab13dd`, float32 and warm local data. Step 2 uses one loader worker and
step 3 sweeps workers; remote S3 throughput remains a separate measurement.

ABIDE's roughly 65k selected voxels and small compressed files emphasize per-file overhead more
than production CIFTI/fixel inputs. MIF has a public equivalent cohort (the walkthrough FDC demo,
via `make_mif_cohort.py`); CIFTI has no public real cohort, so it uses the reproducible synthesizer
(`make_cifti_cohort.py`), optionally seeded from a small real reference for realistic compression.
Repeated local rows measure a warm-cache decode ceiling, and repeated values compress
unrealistically well. Never compare warm and cold cache rows or quote synthetic/layout-sweep rows
as end-to-end results.
