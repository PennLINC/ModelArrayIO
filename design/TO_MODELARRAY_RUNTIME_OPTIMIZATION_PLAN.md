# `to-modelarray` runtime optimization plan

## Goal

Make `modelarrayio to-modelarray` finish as fast as possible without changing
its output schema or breaking ModelArray reads. Memory use, output size and
downstream read speed are things to measure and check, not things to optimize.

Where a choice trades speed against precision or output size, the fast option
is the default and the slower, more precise option is an explicit, documented
CLI flag.

## Terms used in this document

- **chunk** (HDF5) / **tile** (TileDB) — the fixed-size block a file format
  stores and compresses as a unit. Its shape decides how much data gets
  compressed at once, and therefore how finely compression can be split across
  threads.
- **fragment** (TileDB) — one write commit. Many small commits mean many files
  and slower reads.
- **staging matrix** — the in-memory `subjects × elements` array we fill from
  the input images and then hand to the backend.
- **ceiling** — the fastest a step could possibly go given the hardware, e.g.
  how fast the machine can decode `.nii.gz` files at all. Useful as a stopping
  rule: once we are near the ceiling there is nothing left to win.

## Measured starting point

These numbers come from a scouting run on a 14-core macOS workstation with a
warm page cache. They exist only to rank the work; the benchmark plan below
replaces them with measurements on real data.

```text
HDF5 write, gzip-4 + shuffle
  h5py dataset assignment                      73 MB/s
  parallel direct chunk write, 14 threads     695 MB/s
HDF5 write, lzf + shuffle                     336 MB/s
HDF5 write, no compression                   1611 MB/s

TileDB write, zstd + shuffle
  current column-stripe path                  299 MB/s   92 fragments, 278 files
  one full-matrix query                      1992 MB/s    1 fragment,    5 files
  100-subject blocked writes                  922 MB/s    4 fragments

nibabel .nii.gz decode, 1 thread              108 MB/s decoded
nibabel .nii.gz decode, 14 threads           1011 MB/s decoded
get_fdata(float32) vs float64 default         no measurable difference

NIfTI row assembly, current flattened_image    52 ms/subject
NIfTI row assembly, cached mask + float32      34 ms/subject
```

Three conclusions follow, and they set the shape of the plan.

**Writing HDF5 is slow because of compression, not disk speed.** `h5py`
compresses each chunk one at a time, on the single thread that writes the file.
That caps HDF5 output at roughly 70 MB/s no matter how fast the loader feeds
it. Compressing chunks on several threads at once removes the cap — 70 MB/s
becomes 695 MB/s — and is the single largest win available.

**Writing TileDB is fast once we stop reopening the array.** The current code
reopens the array once per spatial stripe. Writing the whole matrix in one
query is 6.7 times faster, and turns 92 write commits across 278 files into
1 commit across 5 files.

**Loading dominates, so overlapping load and write buys little.** With 8 loader
threads the example scalar takes about 1.7 s to load and about 0.10 s to write
to TileDB. Even hiding the write completely behind the load would save about
five percent, and once compression runs on several threads the writer and the
loader compete for the same cores anyway. Overlapping is a late, optional step
whose real value is hiding S3 network latency.

## What is slow currently

1. HDF5 output is limited by compressing one chunk at a time on the writer
   thread.
2. The stripe writers reopen the TileDB array for every spatial stripe, and
   `create_empty_scalar_matrix_array` opens it one more time just to write
   `column_names`. One scalar becomes 92 array opens and 92 write commits.
3. `load_cohort_cifti`, `load_cohort_voxels` and `load_cohort_mif` each build
   the **entire cohort across all scalars** in memory — as dictionaries of
   per-subject row lists — before any writing starts, and they submit one task
   per cohort item when workers > 1.
4. CIFTI-to-TileDB starts several scalar jobs at once under its own
   `--workers`, so a run can hold several full scalar matrices in memory.
5. `flattened_image` loads a mask file for every subject even when the whole
   cohort shares one mask, and it builds two full-volume `float64` temporary
   arrays per row.
6. CIFTI axis validation builds and compares large label arrays for every input
   file, and S3 loading creates a new client for every file.

Relevant code: `src/modelarrayio/cli/{to_modelarray,cifti_to_h5}.py`,
`src/modelarrayio/utils/{cifti,nifti,mif}.py`,
`src/modelarrayio/storage/{h5_storage,tiledb_storage}.py`.

## Decisions

### One public worker option

`--workers N` means one thing: *the maximum number of input sources being
loaded at the same time*. A source is one scalar image plus any companion input
it needs, such as its NIfTI mask. It applies to local paths and `s3://` URLs,
every modality, and both backends. It does not control HDF5 writers, TileDB
writers, or how many scalars are held in memory.

This changes the meaning of an existing option. `--workers` is currently
documented as "maximum number of parallel TileDB write workers ... has no
effect when `--backend=hdf5`". Someone passing `--workers 8 --backend tiledb`
today is asking for write concurrency and will now get load concurrency. An
alias cannot cover this because the option name does not change, so: say it in
the release notes and the CLI help, and log the resolved meaning ("loading N
sources concurrently") on the first run with `--workers > 1`, so the change
shows up in existing job logs.

`--s3-workers` is replaced by `--workers` everywhere. Accept it as a deprecated
alias for one minor release with a warning; give both options `default=None` so
argparse can tell whether the user supplied both; reject conflicting values and
accept matching ones; remove the alias in the next breaking release.

Do not add separate public options for source, scalar and write workers.
Writer lanes stay an internal runtime policy.

### Loader concurrency uses threads

Use a `ThreadPoolExecutor`. This is settled: nibabel `.nii.gz` decoding scaled
from 108 to 1011 MB/s between 1 and 14 threads, because zlib lets other threads
run while it decompresses. Do not build a process pool or a shared-memory ring
buffer.

### Compress HDF5 chunks on several threads

Instead of letting `h5py` compress chunks on the writer thread, compress them
in a thread pool and hand the finished bytes straight to HDF5:

```text
chunk-compression pool (N threads)       writer thread
  byte-shuffle + zlib.compress(chunk) ──→  write_direct_chunk(offset, raw)
```

Measured at gzip-4 with shuffle: 70, 139, 269, 507 and 695 MB/s at 1, 2, 4, 8
and 14 threads. The results read back correctly through ordinary `h5py`.

Two properties make this safe. The dataset is still created with the standard
`compression='gzip'`, `shuffle=True` settings, and each worker produces bytes
that this standard pipeline can decode — so `rhdf5` and `HDF5Array` read the
file with no plugin and no schema change. And there is still exactly one writer
thread, so a shared output file still has exactly one h5py handle.

Requirements:

1. Apply the byte-shuffle step exactly as HDF5's shuffle filter defines it for
   the element size in use, covered by a round-trip test against a dataset
   written the normal way.
2. Pad the last partial chunk out to the full chunk size before compressing,
   because `write_direct_chunk` writes whole chunks only, and check that the
   padded region reads back correctly.
3. Skip the direct-chunk path entirely for `--compression none`; uncompressed
   writes already run at memory speed.
4. Fall back to ordinary dataset assignment if a worker cannot reproduce the
   requested filter settings, and log the fallback instead of silently writing
   a slow file.

### Write each scalar matrix in one submission

Write each staged scalar matrix in a single backend call instead of as a list
of rows. Keep every source-name and CIFTI metadata item exactly where and how
it lives today — in particular, do **not** replace the existing
`scalars/<name>/column_names` dataset with an HDF5 attribute, and keep the
TileDB column-name arrays the CIFTI outputs rely on. The current
`create_scalar_matrix_dataset` writes an attribute, so fix or wrap it first.

For TileDB, use one dense write query per scalar. Keeping the array open does
not merge the stripe queries into one commit, and TileDB-Py 0.36 offers only
`write_direct` and `write_subarray` with no incremental global-order write, so
a single-commit write has no streaming alternative from Python.

Writing in one call means the whole scalar matrix has to be in memory, which
rules out overlapping loader and writer *within* one scalar. That is
acceptable: one matrix is about 1 GiB even at 500 subjects by 500,000 elements.
Allocate it once per scalar as `np.empty((n_subjects, n_elements),
dtype=storage_dtype, order='C')`, in the layout the backend wants, so neither
side needs a second full-size cast or stack copy. Because of this the plan
needs no peak-memory admission checks, no `--staging-dir`, and no
memory-mapped staging.

If some future workload cannot hold one matrix, the documented fallback is
writing in subject blocks with the subject tile size shrunk to the block size —
measured at 922 MB/s for 100-subject blocks, three times better than today and
two times worse than one big write.

Treat the per-subject extent of the chunk or tile as a tunable, since it sets
both the compression unit and how finely compression can be parallelized. Make
whichever compression and layout wins the new default, rather than keeping
gzip-4 because it happened to be the old default.

### Default to the fast dtype, with precision as an explicit opt-in

`--dtype float32` stays the default and is the fast path. `--dtype float64` is
the supported opt-in for workloads that need double precision end to end.

Be precise about where float32 actually wins. *Decoding* in float32 is not
faster: `get_fdata(dtype=np.float32, caching='unchanged')` measured no faster
than the float64 default, because decoding dominates. The win is everything
after decoding — half the bytes to shuffle, half to compress, half to write,
and half the staging memory and cache pressure. Since compression is the HDF5
bottleneck, halving what it has to compress saves real runtime, not just
memory.

Implementation:

- On the default path, read numerical data directly as `np.float32` through the
  nibabel API without filling the image cache, and allocate the staging matrix
  as C-contiguous `float32`. The CIFTI and NIfTI readers currently call
  `get_fdata()` at its float64 default and cast afterwards, which creates a
  large avoidable temporary array.
- On `--dtype float64`, load and stage as float64 throughout so precision is
  exactly preserved. Do not silently downcast.
- Cast MIF data once, at staging, and only when its requested storage type
  differs.
- Benchmark both dtypes at least once on the medium cohort so the CLI help can
  state the real cost of `--dtype float64` instead of an estimate. Expect
  roughly double the write time and double the output size; confirm it.
- CLI help must say that float32 is the default because it is faster and
  smaller, and that float64 preserves full source precision at a measured cost.

### Cache masks and select with flat indices

`flattened_image` builds two full-volume temporaries per row: a float64 mask
thresholded at `> 0`, and a float64 scalar volume that it modifies in place and
then fancy-indexes. In practice the per-subject mask is usually one file
repeated across the whole cohort, and often the group mask itself.

Compute `np.flatnonzero(group_mask)` once; cache masks by file path in a
bounded dictionary; reduce each cached mask down to the selected elements once;
and fill the typed staging row by flat selection, then apply the subject mask
to the already-selected row. Skip the mask load entirely when the subject's
mask path is the group mask path. This dropped row assembly from 52 to 34 ms
per subject with bitwise-identical output, including NaN positions.

Cache the reduced boolean selection — not the nibabel image and not its
`get_fdata()` array. `flattened_image` writes NaN into the decoded volume in
place, which is only safe because every call decodes a fresh copy; a cache that
handed back a reused decoded array would let one subject's NaNs corrupt the
next subject's data. Test that loading the same path twice through the cache
leaves the second row unaffected by the first.

Do not build a float64 mask just to test `> 0`, and do not keep a second full
scalar volume after selection.

### Reuse CIFTI setup work and validate cheaply

Load, decode and validate the first CIFTI source once. Reuse its decoded row as
the first row of the first staging matrix, and reuse its metadata to create the
greyordinate/parcel output metadata without reopening the file.

Build a short, stable signature of the spatial axis from that reference header
and check later sources against the signature instead of comparing full label
arrays element by element. The signature must cover every spatial property the
current comparison protects, and must leave out scalar labels that legitimately
differ between inputs. Cache signatures that pass, and build full label arrays
only for the reference metadata or for a failure message.

### Do not throttle TileDB's internal thread pools

TileDB writes measured 791 MB/s against `h5py`'s 73 at the same codec and
level, and the reason is that TileDB compresses on many threads internally.
Capping `sm.compute_concurrency_level` or `sm.io_concurrency_level` to
`--workers` would give that away. Leave the compute pool at the allocated core
count; if contention shows up in measurements, reduce *loader* threads while a
write is in flight instead. Record TileDB's concurrency settings in every
benchmark run.

For S3, reuse one configured client whose connection pool can serve `--workers`
loads at once.

### Preserve split-output behaviour

`--scalar-columns` selects wide-cohort input and produces one output per
scalar; a long cohort keeps the scalar datasets under one output root. Preserve
this exactly. There is no `--no-split-files` option, and split behaviour is not
a performance knob.

## Implementation plan

### 1. Measurement before behaviour

Add per-scalar structured timing at INFO level, plus optional JSON Lines output
for benchmark automation. Record source load time, CIFTI validation time,
staging allocation time, write time with compression separated from
submission, per-scalar and total time, matrix shape and dtype, chunk/tile
shape, compression settings, peak memory, and TileDB commit and output-byte
counts per write.

Also record *waiting*, not just duration. Phase durations alone cannot tell you
which side to widen, and they will not reveal the loader/writer core
competition that parallel compression introduces. So record writer idle time,
loader blocked time, sampled queue occupancy, and per-phase CPU utilization.

Keep it cheap. Do not instrument every one of hundreds of thousands of files
during production runs.

### 2. Shared bounded source loader

Build one modality-neutral bounded-executor helper that:

- takes the ordered list of sources for the scalar being staged;
- keeps at most `workers` loads submitted or running at any time, rather than
  one task per file;
- writes each row straight into its own index in the preallocated staging
  matrix, so completion order does not matter and no results dictionary is
  needed;
- includes the source path and scalar name in errors; and
- never builds a list of rows or a second full-size `stack`/cast copy.

Use it for local files and S3 alike; only the scheduling becomes shared, and
each loader still picks its own nibabel/S3/MRtrix reader. Avoid pools inside
pools — CIFTI-to-TileDB must use this loader instead of launching one
long-running outer task per scalar. Loader threads hold no HDF5 handles: for a
combined HDF5 output, every h5py write goes through its single writer lane. For
S3, release the response body and any compressed or decompressed byte buffers
as soon as the typed row is copied into staging.

### 3. Replace the stripe writers

Implement the full-matrix writers, the parallel HDF5 chunk-compression path,
and the removal of the redundant array opens. Steps 2 and 3 are the bulk of the
expected win and must be benchmarked before any overlap work begins.

Use one writer lane for a combined HDF5 file. For TileDB, benchmark one lane
first and add a small number of lanes only for scalar-split outputs that can be
written independently; concurrent writers must never modify shared
output-group metadata unless that is proven safe. Expect the compression pool,
not extra writer lanes, to be the effective lever.

### 4. Overlap load and write only if the measurement justifies it

Read the writer-idle and loader-blocked counters. If writing is a small
fraction of wall time on local input, as the scouting numbers suggest, stop
here for local workloads.

Add overlap only where it pays, which is S3 input where network latency
dominates: start loading the next scalar as soon as the current one is handed to
a queue that holds one item. Keep the depth at one; a deeper queue only
multiplies how many matrices sit in memory. The first CIFTI row is copied into
the staging matrix rather than decoded again.

### 5. Recoverable output completion

For scalar-split output, write each scalar to a temporary name in the
destination parent and publish it only after its data and metadata validate.
For a combined HDF5 file or TileDB root, write the whole conversion to one
temporary root on the same filesystem and publish after every scalar validates.
Use atomic rename where supported; never turn publication into a hidden copy
whose duration is left out of the benchmark.

Published scalar outputs let a user target missing work manually, but the
converter must not implicitly skip them on a later invocation. Resume behaviour
stays an explicit opt-in, decided separately.

## Benchmark plan

### Benchmark data: real ABIDE images

Benchmark on real images, not generated arrays. The primary cohort is the open
ABIDE derivatives already used by `test/test_nifti_s3.py`:

```text
s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/
  alff/<subject>_alff.nii.gz
  func_mask/<subject>_func_mask.nii.gz
  falff/, reho/, vmhc/, lfcd/, degree_binarize/, degree_weighted/,
  eigenvector_binarize/, eigenvector_weighted/   (same 1102 subjects)
```

The bucket is public, so `MODELARRAYIO_S3_ANON=1` with unsigned requests is
enough and no credentials appear in any recorded command. The properties that
matter: 1102 subjects per derivative; 61×73×61 volumes with about 65,193
in-group-mask voxels, so 0.249 MiB per `float32` row and 274 MiB for a full
1102-subject matrix; about 285 KB per gzipped file and 0.29 GiB per derivative;
and the files store **`float64` on disk**, so the float32 change gets tested
against a genuinely double-precision source rather than data that was already
float32.

Cohorts:

- **fixture** — the 20 committed `test/data_voxel_toy/FA` volumes plus
  `test/data_cifti_toy`, for correctness tests only.
- **small** — 100 ABIDE subjects, 1 scalar, for iteration.
- **medium** — 1102 ABIDE subjects, `alff`. The default comparison cohort.
- **wide** — 1102 ABIDE subjects × 9 derivatives, about 2.4 GiB staged. The
  only cohort that exercises the per-scalar loop, `--scalar-columns` split
  outputs, and writer lanes.
- **production confirmation** — the real internal workload, run once on the
  final two or three candidates.

To go beyond 1102 input files, repeat real subjects as extra cohort rows rather
than synthesizing values: each repeated row is a genuine open, decode and
mask-apply, so loader throughput, `--workers` scaling and S3 latency stay real.
Record the repeat factor. Two caveats: repeats read warm from the page cache
locally, so a repeated-row local run measures the decode ceiling rather than
the filesystem; and duplicate rows compress unrealistically well, so never draw
output-size or compression-ratio conclusions from them.

Two limits to state wherever ABIDE numbers are reported. First, 65 k voxels and
285 KB files make per-file overhead matter more than it does in production
fixel and CIFTI workloads — which is exactly what the production confirmation
run exists to catch. Second, ABIDE has no public CIFTI or MIF equivalent, so
those modalities are benchmarked on internal data with the same harness and
only the NIfTI numbers are publicly reproducible.

Use the same cohort file and an empty output location for every comparison.
Label warm-cache and cold-cache runs separately and never compare across that
line. Repeat important candidates at least three times and compare medians.

### Benchmarking lives in its own folder

Everything about benchmarking goes in one new self-contained folder so it can
be added, run and updated without touching the library or the main test suite:

```text
benchmarks/
  README.md                 what each cohort is, how to run, how to read results
  make_cohort.py            deterministic ABIDE cohort-CSV generator
  run_benchmarks.py         builds the CLI commands, times them, appends result rows
  plot_results.R            figures from the results CSV
  results/
    <run-id>/
      results.csv           one row per run
      run_meta.json         exact commands, environment, versions, cohort hash
      plots/                generated figures
  baselines/
    baseline_<cohort>.csv   pinned reference numbers, committed
```

Benchmark output is written under `benchmarks/results/<run-id>/` and nowhere
else — not into the package, not into the test data folders, not into `docs/`.
An environment variable (`MODELARRAYIO_BENCHMARK_RESULTS_DIR`) overrides the
results root so a cluster job can write to scratch. The harness documentation
lives in `benchmarks/README.md`; the main docs link to it rather than
duplicating it.

### Reuse the conventions, not the code, from earlier benchmarking

An earlier round of HDF5 benchmarking in this project produced result CSVs,
plots and a pytest-based runner. Do not port that code forward: it was written
before `to-modelarray` existed, it benchmarks the writer rather than the CLI,
and adapting it costs more than writing the small harness above. Its
conventions are worth keeping, though, because they worked:

- Append one row per run to a CSV, so an interrupted sweep keeps the runs it
  already finished and later runs accumulate instead of overwriting.
- Keep a `run_meta.json` beside the CSV with the exact commands and environment
  for every run, so any row can be reproduced later.
- Select size tiers by flag (quick / medium / full) so the same harness serves
  both a 30-second sanity check and an overnight sweep.
- Add columns over time instead of renaming them, and fill missing columns when
  loading, so old rows stay plottable by the current plotting script.
- Keep a `run_kind` column that separates end-to-end runs from synthetic layout
  sweeps, so the two are never quoted interchangeably.
- When the sweep runs in parallel, write one result file per parallel worker and
  merge them at plot time.

The earlier *numbers* are reusable too, as hypotheses rather than results. Its
chunk-geometry and gzip-level sweeps point the layout and codec sweep below at
the promising region instead of a blind grid. But they came from a writer-only
harness fed synthesized values, so no figure from them counts as a result here;
every number this plan reports comes from a run of the new harness.

### What the harness must do differently

1. **Benchmark the CLI end to end, not the writer.** Drive `to-modelarray`
   against an ABIDE cohort CSV and record the phase counters from step 1.
   Timing a single call to `create_empty_scalar_matrix_dataset` filled with
   generated values covers about five percent of what this plan optimizes.
2. **Keep the output.** Do not delete the converted file after measuring its
   size; the R read gate and the correctness comparison both need it.
3. **Cover both backends.** Record the backend, TileDB fragment count, output
   file count and TileDB concurrency settings.
4. **Record the new knobs**: `--workers`, dtype, compression-pool size, decode
   time, writer-idle and loader-blocked time, peak memory and repeat factor.

Keep a small seeded data generator for the one thing synthetic data is good at:
sweeping chunk geometry and codecs at 10,000 and 40,000 rows, where holding
that many real decoded subjects is not the point. Mark those rows in `run_kind`
and never quote them as end-to-end runtimes.

### Sequence

1. **Measure the ceilings first, before writing pipeline code.** Real ABIDE
   decode throughput, serially and at several thread counts, per modality;
   single-matrix compress-and-write throughput per backend and candidate codec,
   including the parallel direct-chunk path at several thread counts; and raw
   source-read and output-write bandwidth plus S3 first-byte latency. From
   those, write down the predicted floor and an **absolute acceptance target**
   — for example a sustained input-files-per-minute figure, or a stated
   fraction of the decode ceiling. "The fastest candidate" has no stopping
   rule; a target does.
2. **Current baseline** on the medium cohort at current production settings.
   Do not spend a production-scale run on the unmodified converter.
3. **Architecture comparison** on the same cohort with identical dtype,
   compression, tile target and storage. This isolates bounded loading, float32
   loading, mask caching, full-matrix writing and parallel compression. ABIDE
   tests the mask-caching change well, because every subject has its own
   `func_mask` and none of them is the group mask, so it measures the
   cache-miss path rather than the best case.
4. **Loader sweep.** `--workers` at 1, 2, 4, 8 and higher, while filesystem
   throughput and memory stay healthy. Run local and `s3://` separately; one is
   CPU-bound on zlib and the other is latency-bound, so their best settings
   differ.
5. **Compression-thread sweep.** For HDF5, sweep the chunk-compression pool
   against a fixed `--workers`, then sweep both together, since they compete
   for the same cores. Record where throughput stops improving.
6. **dtype check.** Medium cohort at `--dtype float32` and `--dtype float64`,
   recording runtime, write time, output size and peak memory, to document the
   real cost of the precision opt-in.
7. **Overlap check.** Sequential versus bounded overlapping pipeline, local and
   S3 separately. Expect a small local gain and a larger S3 gain; keep overlap
   only where it pays.
8. **Writer-lane sweep** on the wide cohort. One lane for combined HDF5; a
   small number of bounded lanes for TileDB scalar-split outputs that can be
   written independently.
9. **Layout and codec sweep.** Item extents of 8, 32, 64 and 128 MiB. For
   TileDB, `zstd` at levels 1, 3 and 5 with shuffle, plus none; if output is
   still the bottleneck, try alternative cell/tile orders and TileDB internal
   concurrency — and confirm that constraining TileDB's pools makes things
   worse before shipping any such setting. For HDF5, gzip 1 and 4 through the
   direct-chunk path, plus none, plus any plugin codec that clears the R gate.
   Always pass and record an explicit `--compression`. This step and step 8 may
   use seeded synthetic data to reach 10,000 and 40,000 rows, but must also run
   once at each real cohort size, because compression ratio depends on the real
   value distribution.
10. **Full-scale confirmation** of the two or three finalists on the production
    workload, reporting time per output unit, input files per minute, output
    size, peak memory and total duration.

Every run records: ModelArrayIO commit, Python, nibabel, h5py and TileDB
versions, allocated CPUs and RAM, TileDB concurrency settings, source and
output filesystem types, cohort hash and repeat factor, scalar names, input
byte total, and the exact CLI command; plus wall time, CPU utilization, peak
memory, output bytes, phase times, writer-idle and loader-blocked time, TileDB
fragment and file counts, peak open file descriptors and S3 pool settings. On
Linux, `/usr/bin/time -v` covers the outer resource record.

Comparing node-local scratch storage is optional and low priority now that
staging lives in RAM. If it is run, include staging and copy-back time.

### Every HDF5 codec candidate must pass an R read gate

ModelArray reads HDF5 through `rhdf5` and `HDF5Array`. LZF ships with `h5py`;
zstd and blosc need a filter plugin registered against whatever libhdf5 those
Bioconductor packages were built with. A codec that Python writes quickly but R
cannot open is not a candidate at all, however good its number looks: LZF
measured 336 MB/s against gzip-4's 73, which is exactly the kind of result that
gets adopted by accident.

Read every candidate output back through ModelArray in R, on the deployment
environment, as a pass/fail gate before its timing counts. This is why the
harness must keep its output. The parallel direct-chunk path makes the question
largely moot for HDF5, since it reaches 695 MB/s while writing a standard
gzip+shuffle file that needs no plugin.

Record R-side read time for a representative element range alongside the gate.
A candidate that halves conversion time by turning off compression and then
doubles every downstream read is a bad trade that a write-only benchmark cannot
see.

## Acceptance criteria

Compare every optimized output against the current converter on the same
inputs:

- scalar names, source ordering, shapes, dtypes, and the candidate's declared
  chunk/tile metadata — not equality with the baseline layout, when the sweep
  picked a different compatible one;
- CIFTI greyordinate and parcel metadata;
- NaN positions, and values bitwise equal where the source transform permits;
  otherwise an explicitly documented float32 rounding tolerance;
- successful ModelArray construction and representative reads in R for the
  selected codec and layout; and
- no unexpected TileDB fragment growth and no leftover temporary output.

Ship the fastest candidate that produces correct, compatible output, clears the
R read gate, and fits the allocated resources. Report output size and
downstream read observations, but do not use them to reject a faster correct
conversion unless the R-side read regression is large enough to be recorded as
a deliberate trade.

## Deliverables

Code and tests:

- Unit tests for bounded scheduling, ordered rows, worker validation, and
  float32/float64 reader behaviour.
- A round-trip test proving the parallel direct-chunk path produces a dataset
  whose declared filter settings and read-back values match one written the
  normal way, including the padded final partial chunk and the
  `--compression none` bypass.
- A test that the direct-chunk fallback triggers and logs when a worker cannot
  reproduce the filter settings.
- Mask-caching and flat-index tests covering the group-mask case, the
  differing-mask case, and the cached-row mutation hazard.
- Integration tests for CIFTI, NIfTI and MIF at `--workers 1` and `>1`, on both
  backends, and at both dtypes.
- Tests that `--s3-workers` warns and aliases `--workers`, including the
  conflicting-value case.
- Regression tests that scalar-split versus long-cohort behaviour is unchanged,
  that full-matrix writers preserve every existing HDF5 and TileDB metadata
  location including CIFTI column-name arrays, and that one scalar commits as
  one TileDB fragment.
- CIFTI tests that reuse the first decoded row, accept an equivalent later
  header without building label arrays, and still reject every spatial-axis
  mismatch the current validation catches.

Documentation and benchmarking:

- CLI help, README and usage docs covering the single worker meaning, the
  redefinition of `--workers`, the `--s3-workers` removal schedule, and the
  measured cost of `--dtype float64`.
- The `benchmarks/` folder described above, containing the harness, the
  plotting script, `benchmarks/README.md`, and a deterministic ABIDE cohort-CSV
  generator that takes a subject count, derivative list and repeat factor, so
  anyone can reproduce the small, medium and wide cohorts from the public
  bucket with no credentials.
- A pinned baseline results CSV committed under `benchmarks/baselines/`, so a
  future regression is detectable by rerunning one command.

## Non-goals

- No independent public knobs for source, scalar and write workers.
- No output schema change; chunk/tile layout is tunable only while the output
  stays compatible and readable from R.
- No process-based or shared-memory loader.
- No peak-memory admission checks, memory-mapped staging, or `--staging-dir`.
- No mandatory node-local scratch.
- No exhaustive sweep of every combination of formats, backends and settings
  before the target architecture is fixed.
