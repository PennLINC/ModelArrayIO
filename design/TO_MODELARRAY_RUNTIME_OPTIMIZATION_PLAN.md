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

**Production baseline (CUBIC, real run).** The original converter was run on the
full RBC freesurferpost cohort — 8,693 subject-sessions × 70 scalars, 608,510
`fs_LR den-164k` dscalars — with `--backend tiledb --compression zstd
--compression-level 1 --target-chunk-mb 32 --workers 15`
(`modelarray_rbc_tiledb/code/02_create_tiledb.sh`). Its log shows the **first
scalar took 112 h 23 m** (the old loader parsed all 608,510 CIFTI XML headers
up front, GIL-bound, so extra workers made it slower), and the run stalled at
15/70 scalars after ~119 h; 18 scalars were completed. That output is now the
bitwise correctness reference for the optimized converter.

The scouting numbers below come from a 14-core macOS workstation with a warm
page cache; they rank the work, and the CUBIC ladder in the benchmark plan
replaces them with production measurements.

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

## What was slow in the original code

The measurements above explain each bottleneck; the fix for each is in
*Decisions*.

1. **HDF5 writes were limited by compression, not disk speed.** `h5py`
   compresses each chunk one at a time on the single thread that writes the
   file, capping output at roughly 70 MB/s no matter how fast the loader feeds
   it. Compressing chunks on several threads removes the cap (695 MB/s) and is
   the single largest write-side win.
2. **TileDB writes reopened the array once per spatial stripe**, and
   `create_empty_scalar_matrix_array` opened it one more time just to write
   `column_names`: one scalar became 92 array opens and 92 write commits across
   278 files. One full-matrix query is 6.7 times faster and leaves 1 commit
   across 5 files.
3. `load_cohort_cifti`, `load_cohort_voxels` and `load_cohort_mif` each built
   the **entire cohort across all scalars** in memory — as dictionaries of
   per-subject row lists — before any writing started, and submitted one task
   per cohort item when workers > 1.
4. CIFTI-to-TileDB started several scalar jobs at once under its own
   `--workers`, so a run could hold several full scalar matrices in memory.
5. `flattened_image` decoded each scalar and its individual mask into
   full-volume `float64` temporaries, thresholded the mask at `> 0` into another
   float64 array, and mutated the decoded scalar volume in place before
   fancy-indexing it — several full-volume temporaries per row at double the
   precision the output needs.
6. CIFTI axis validation built and compared large label arrays for every input
   file, and S3 loading created a new client for every file.

**Loading dominates, so overlapping load and write buys little.** With 8 loader
threads the example scalar takes about 1.7 s to load and about 0.10 s to write
to TileDB. Even hiding the write completely behind the load would save about
five percent, and once compression runs on several threads the writer and the
loader compete for the same cores anyway. Overlap is therefore deferred
(implementation step 4); its real value would be hiding S3 network latency.

Relevant code: `src/modelarrayio/cli/{to_modelarray,cifti_to_h5}.py`,
`src/modelarrayio/utils/{cifti,nifti,mif}.py`,
`src/modelarrayio/storage/{h5_storage,tiledb_storage}.py`.

## Decisions

### One public worker option

*Status: implemented — cli/runtime.py:resolve_loader_workers, cli/to_modelarray.py.*

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

*Status: implemented — utils/loading.py:stage_source_rows (bounded ThreadPoolExecutor).*

Use a `ThreadPoolExecutor`. This is settled: nibabel `.nii.gz` decoding scaled
from 108 to 1011 MB/s between 1 and 14 threads, because zlib lets other threads
run while it decompresses. Do not build a process pool or a shared-memory ring
buffer.

### CIFTI loading bypasses the XML parse instead of adding threads

*Status: implemented — utils/cifti.py:_CiftiRawReadPlan / _read_cifti_row_raw; rows served this way are counted in `fast_path_rows`.*

The thread scaling above is a NIfTI/gzip result and does **not** transfer to
uncompressed CIFTI. On real `fs_LR den-164k` `.dscalar.nii` files (327,684
greyordinates, ~3.4 MB, of which ~2 MB is the CIFTI XML header) nibabel spends
about 0.78 s parsing the header and 0.48 s in `get_axis` while reading the data
block takes 0.5 ms. That parse holds the GIL, so it does not thread: measured on
cubic, the loader ran 0.8 files/s at one worker and *0.3 files/s at eight*. A
608,510-file freesurferpost conversion under this path took nearly seven days
with `--workers 15` pinned to roughly one core.

Because every source in a cohort shares one surface template, its values are a
contiguous floating block after the XML; the XML only needs parsing once. The
loader therefore decodes each scalar's first source with nibabel to fix the
reference row, brain-model axis and layout, then reads later sources straight
from their data block (`np.fromfile` at each file's own `vox_offset`) whenever a
cheap check passes — falling back to a full nibabel decode otherwise. The plan
self-validates against the decoded reference row, so any layout it cannot
reproduce (parcellated-connectivity ordering, on-disk scaling, byte order,
remote or compressed inputs) disables it rather than risking a wrong row.
`MODELARRAYIO_CIFTI_RAW=0` forces the nibabel path.

**What the cheap check must be (learned on CUBIC).** The first version required
the file's `vox_offset` and size to equal the reference's. On the production RBC
files that rejected 91 of 100 `area` files: the pipeline embeds per-subject
provenance strings (`wb_command` paths with job/subject/session IDs) in the CIFTI
metadata, so the XML length — and `vox_offset` — differs by a few bytes between
subjects while the brain-model `MatrixIndicesMap` is byte-identical
(2,072,200 bytes across every size class). Those rows silently fell back to the
1 s nibabel parse and the CUBIC step-2 CIFTI comparison showed no speedup at
all. The check is now: NIfTI-2 dims, dtype and scaling equal the reference, the
file is large enough to hold the block at its own offset, and a SHA-256 of the
raw XML bytes of the dimension-1 (brain-model) `MatrixIndicesMap` equals the
reference's — a few milliseconds per file, and strictly safer than the size
check, since identical map bytes are what guarantee identical greyordinate
order. `fast_path_rows` in the per-scalar telemetry counts rows served this
way; a value below the row count means silent fallbacks and must be treated as
a performance bug, not noise.

Measured effect on real data (16 subjects × 3 scalars, warm cache): output
bitwise-identical to nibabel, and 448 s → 6.3 s at `--workers 8` (71×). Because
the raw read releases the GIL, threads help again; the per-scalar reference
decode (~70 nibabel loads for the full cohort) is negligible. This is the
CIFTI-specific answer to "loading dominates" — for uncompressed CIFTI the fix is
to stop parsing the header, not to add loader threads or a process pool.

### Compress HDF5 chunks on several threads

*Status: implemented — storage/h5_storage.py:_write_direct_gzip_chunks (pool size `MODELARRAYIO_HDF5_COMPRESSION_WORKERS`, default all CPUs).*

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

*Status: implemented — storage/tiledb_storage.py (one dense write, fragment_count 1); storage/h5_storage.py:write_full_matrix.*

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

*Status: implemented — cli/to_modelarray.py `--dtype` (float32 default); readers decode directly in the storage dtype.*

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

*Status: implemented — utils/nifti.py:_flat_group_indices, ReducedMaskCache, _selected_mask_values.*

`flattened_image` builds two full-volume temporaries per row: a `float64` mask
thresholded at `> 0`, and a `float64` scalar volume that it modifies in place
and then fancy-indexes.

The group mask fixes the output columns — the set of voxels every subject's row
is reported at. Each subject also has its **own** individual mask, defined by
that subject's anatomy, which may cover only part of the group mask and differs
from subject to subject. We cannot assume any subject's mask matches the group
mask or another subject's. Voxels that are inside the group mask but outside a
given subject's individual mask are written as `NaN` for that subject; ModelArray
reads those per-subject `NaN`s and handles the differing coverage downstream. So
the per-subject mask is real, per-subject work, not a shared file that can be
loaded once and reused.

Compute `np.flatnonzero(group_mask)` once; fill the typed staging row by flat
selection at those group indices; reduce each subject's individual mask to the
same group indices; and set the out-of-mask positions of the already-selected
row to `NaN`. The scalar is decoded directly in the storage dtype (`float32` by
default) via `get_fdata(dtype=storage_dtype, caching='unchanged')`, and the mask
is read from its native `dataobj` and thresholded `> 0` — neither is ever
promoted to `float64`. Cache reduced masks by file path in a bounded dictionary
so cohorts that *do* repeat a mask (for example repeated rows, or sessions that
share one mask) avoid re-decoding it, but never skip applying a subject's mask on
the assumption that the group mask covers it. The one safe short-circuit is
when a subject's mask path is literally the group-mask file: every selected voxel
is then in-mask by construction, so applying it is a no-op and is skipped. This
dropped row assembly from 52 to 34 ms per subject with bitwise-identical output,
including `NaN` positions.

Cache the reduced boolean selection — not the nibabel image and not its
`get_fdata()` array. Each row is a fresh `np.take` result that gets its
out-of-mask positions overwritten with NaN, which is only safe because every
call decodes and selects a fresh copy; a cache that handed back a reused
scalar array would let one subject's NaNs corrupt the next subject's data. The
cached reduced mask is read-only and never mutated. Test that loading the same
path twice through the cache leaves the second row unaffected by the first.

Do not build a float64 mask just to test `> 0`, do not keep a second full
scalar volume after selection, and stage in `float32` unless the user asks for
`--dtype float64`.

### Reuse CIFTI setup work and validate cheaply

*Status: implemented — utils/cifti.py (first row reused; cifti_spatial_signature instead of label compare).*

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

*Status: implemented — concurrency levels are only read into telemetry (cli/runtime.py:_tiledb_runtime_settings), never set.*

TileDB writes measured 791 MB/s against `h5py`'s 73 at the same codec and
level, and the reason is that TileDB compresses on many threads internally.
Capping `sm.compute_concurrency_level` or `sm.io_concurrency_level` to
`--workers` would give that away. Leave the compute pool at the allocated core
count; if contention shows up in measurements, reduce *loader* threads while a
write is in flight instead. Record TileDB's concurrency settings in every
benchmark run.

For S3, reuse one configured client whose connection pool can serve `--workers`
loads at once.

### Per-scalar output is the primary target

*Status: implemented — cli/to_modelarray.py `--split-files`/`--no-split-files`; per-scalar atomic publish in every CLI.*

Optimize for **one output file per scalar** — one `.h5` file, or one `.tdb`
root, per scalar. This is what we want in practice: ModelArray reads per-scalar
files directly, and separate files make a cohort easier to share, copy, and
reprocess one scalar at a time. `--split-files` produces per-scalar outputs and
is the default for wide `--scalar-columns` input; `--no-split-files` keeps every
scalar under one combined output root and stays supported for callers that want
a single file. Split is a data-layout choice, not a performance knob — but
because per-scalar files are fully independent, the scalar is also the unit the
converter parallelizes and shards on (next section), so the fast path is built
around it.

Each scalar is staged lazily and written to its own atomically published output
with no cross-scalar coupling, so one scalar can be built, validated, and
published without touching another. Preserve the greyordinate/parcel and
fixel/voxel metadata each per-scalar file needs.

### Scalar-sharded conversion for HPC

*Status: implemented — recipe and job templates in benchmarks/to_modelarray/cubic/ (README, sbatch/04_scalar_array.sbatch); `--only-scalar` deferred — per-scalar cohort CSVs select the scalar.*

Because each scalar's file is independent, the scalar is the natural unit of
coarse parallelism, and on a cluster the productive pattern is **one job per
scalar** rather than more threads inside one job. This mirrors ModelArray's HPC
sharding (`vignette("element-splitting")` and its `run_chunk.R`), with one
important difference that comes from building the matrix rather than analysing
one.

ModelArray shards the **element** axis: each array task analyses a disjoint
range of fixels/voxels, writes its own `result_chunk_k.h5`, and a final
single-process step concatenates the chunk files into one output. That works
because it reads element slices lazily from an already-built matrix, and it needs
the combine step only because the shards are disjoint element ranges and HDF5
forbids concurrent writes.

Building the matrix is the opposite shape, so the axes map differently:

- **Element-range sharding does not transfer.** To read elements *j..k* of a
  subject you must decode that subject's whole `.nii.gz`/`.mif` — gzip is not
  seekable — so *E* element-shards re-decode all *N* subjects (*E×N* decodes).
  This is the one piece of ModelArray's approach that does not carry over to the
  build.
- **Scalar sharding is the win, with no aggregation.** One array task per scalar,
  each decoding its subjects once and writing its own final per-scalar file. The
  shards are already the desired output granularity, so there is nothing to
  combine and no concurrent-write hazard — "shard and aggregate" collapses to
  just "shard." The CLI already supports this (one `--scalar-columns X`, or a
  scalar-filtered cohort, per `$SLURM_ARRAY_TASK_ID`); an explicit
  `--only-scalar` selector would let a long-format job array map cleanly to the
  task index without pre-filtering the CSV. Document this array recipe as the
  build-side analogue of `element-splitting`.
- **Subject-block sharding is a niche add-on** for a single scalar too large for
  one node. Decode each subject once inside its block, then recombine along the
  subject (row) axis. The backend decides the cost: HDF5 inherits ModelArray's
  no-concurrent-write rule, so it needs a single-process combine that copies the
  row blocks into the final dataset — one extra full pass, worth it only when
  decode dominates a plain copy, which it does. TileDB does not inherit it —
  subject-block jobs can write disjoint row ranges as concurrent fragments into
  one array with no copy, trading the single-fragment write for a bounded number
  of fragments (one per block, not per stripe). For the sharded build this is a
  real TileDB advantage over HDF5.

Scalar-sharded jobs and the in-node thread parallelism (`--workers` decode
threads plus the HDF5 compression pool / TileDB internal pool) are orthogonal
and compose: shard scalars across nodes, thread within each job. Neither adds a
public worker knob; `--only-scalar` selects work, it does not size a pool.

## Implementation plan

### 1. Measurement before behaviour

*Status: implemented — cli/runtime.py:TimingRecorder, `--timing-jsonl`.*

Per-scalar structured timing at INFO level plus optional JSON Lines output for
benchmark automation records: source load time, CIFTI validation time, staging
allocation time, write time with compression separated from submission,
per-scalar and total time, matrix shape and dtype, chunk/tile shape,
compression settings, peak memory, TileDB fragment counts and output bytes,
CPU utilization, and the bounded-loader counters that actually exist —
`scheduler_wait_seconds`, `in_flight_sources_mean/max`, and `fast_path_rows`
(rows served by the CIFTI raw read; a value below the source count means some
rows silently fell back to nibabel). Loader/writer *overlap* counters
(`loader_blocked_seconds`, queue occupancy) are not measured because the
pipeline is sequential per scalar; they are emitted as zeros and must not be
used as evidence. Keep instrumentation cheap: never per-file at production scale.

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

In-process writer lanes are for building several per-scalar files within one
invocation. On HPC, prefer the coarser and simpler split: one process (one job
array task) per scalar, each writing its own per-scalar file, as in
*Scalar-sharded conversion for HPC*. That needs no shared-handle coordination
and no aggregation, and it uses the scheduler's cores instead of competing with
the in-node loader and compression threads.

### 4. Deferred: overlap load and write

Not implemented, and not justified by measurement: on local and GPFS input the
write phase is a small fraction of wall time and the two phases run
sequentially per scalar. Revive it only for S3-latency-bound input, with a
one-item queue (a deeper queue only multiplies resident matrices). Subject-block
sharding is likewise deferred until a single scalar no longer fits one node.

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
run exists to catch. Second, ABIDE is NIfTI-only, so the other two modalities
use their own reproducible cohorts, described next.

### MIF and CIFTI benchmark data

MIF has a fully public, reproducible cohort: the ModelArray walkthrough FDC
demo — 100 fixel `.mif` subjects (602,229 fixels each) plus `index.mif` and
`directions.mif`, downloaded from OSF (`https://osf.io/tce9d/download`, mirrored
by `benchmarks/to_modelarray/make_mif_cohort.py`). This is real diffusion fixel data, so its
runtime, `--workers` scaling and compression ratios are all quotable, and the
shipped `cohort_FDC_n100.csv` is already long-format and drives the harness
directly. It is the MIF analogue of ABIDE.

CIFTI has no public `.dscalar.nii` cohort, so it is benchmarked with
`benchmarks/to_modelarray/make_cifti_cohort.py`, which has two reproducible modes. The default
`synthesize` writes seeded `fs_LR den-164k` dscalars with scalar-appropriate
marginal ranges and a medial-wall zero block — reproducible anywhere and valid
for load/runtime timing, but its per-vertex values are independent, so it is not
quoted for compression ratios. Passing `--reference-dir` derives, per scalar,
the shared medial-wall mask and the per-vertex mean and standard deviation from
a small set of real dscalars and generates each subject as
`clip(mean + std·noise, min, max)`; only aggregate statistics are read and fresh
noise is drawn, so no real subject row is reproduced, while the real mean map's
spatial smoothness makes compression realistic (measured 1.38× versus 1.41× for
the real reference at gzip-4, against 1.17× for the independent-noise default).
The reference axes are copied verbatim so the header and `vox_offset` match the
real files exactly, which is what the raw-read fast path depends on.

Use the same cohort file and an empty output location for every comparison.
Label warm-cache and cold-cache runs separately and never compare across that
line. Repeat important candidates at least three times and compare medians.

### Benchmarking lives in its own folder

Everything about benchmarking goes in one new self-contained folder so it can
be added, run and updated without touching the library or the main test suite:

```text
benchmarks/
  README.md                 index of suites and their shared conventions
  to_modelarray/            this suite; a future export-results suite gets a sibling folder
    README.md               what each cohort is, how to run, how to read results
    make_cohort.py          deterministic ABIDE (NIfTI) cohort-CSV generator
    make_mif_cohort.py      downloads/normalizes the public walkthrough FDC cohort
    make_cifti_cohort.py    seeded/realistic CIFTI dscalar cohort generator
    run_benchmarks.py       builds the CLI commands, times them, appends result rows
    r_read_gate.R           per-candidate R read gate (rhdf5/HDF5Array/tiledb)
    plot_results.R          figures from the results CSV
    compare_revisions.py    original (pinned commit) vs optimized, per modality, with a bitwise check
    harness_common.py       cohort, CSV, output-naming and provenance helpers shared by the scripts
    output_compare.py       block-wise HDF5/TileDB output comparison (revisions and reference)
    comparison_runs.py      reads compare_revisions run directories for the report and summary
    create_comparison_report.py  renders the cross-modality original-vs-optimized findings PDF
    fixtures/               committed fixture cohort CSVs
    baselines/
      baseline_<cohort>.csv pinned reference numbers, committed
    reports/                rendered findings PDFs, committed
    cubic/                  HPC validation ladder (SLURM templates, container env)
    data/  cohorts/         downloaded/generated inputs, gitignored
    results/                gitignored
      <run-id>/
        results.csv         one row per run
        run_meta.json       exact commands, environment, versions, cohort hash
        plots/              generated figures
```

Benchmark output is written under `benchmarks/to_modelarray/results/<run-id>/` and nowhere
else — not into the package, not into the test data folders, not into `docs/`.
An environment variable (`MODELARRAYIO_BENCHMARK_RESULTS_DIR`) overrides the
results root so a cluster job can write to scratch. The harness documentation
lives in `benchmarks/to_modelarray/README.md`; the main docs link to it rather than
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

### Sequence: the CUBIC validation ladder

The local original-vs-optimized comparison (2026-09-21, superseded by the CUBIC step-2 results
pinned in `benchmarks/to_modelarray/baselines/cubic_cross_modality_9ab13dd_vs_optimized_20260922.csv`)
fixes the architecture; what remains is validating it where it runs. Each step is
gated on correctness before the next starts; scripts live in `benchmarks/to_modelarray/cubic/`.

0. **Environment and smoke** — container venv (glibc 2.31) with the exact local
   library versions, offline pytest on a compute node, quick-tier fixtures
   through both gate branches (R ModelArray for HDF5, Python for TileDB), and a
   fixture old-vs-new bitwise comparison. `source_tree_sha256` must equal the
   local tree's.
1. **Production-data preflight** — reference-scalar completeness
   (`compare_to_reference.py --list`), per-scalar cohort CSVs, header-size
   uniformity of one scalar's files (the raw fast path needs it).
2. **Original vs optimized on the public and RBC-subset cohorts** — ABIDE-100,
   MIF walkthrough-100, RBC CIFTI 100 × 3; both backends; 3 counterbalanced
   repetitions after a warm-up; bitwise equality, both gates; the raw fast path
   and the nibabel path must both equal the deployed reference.
3. **One full production scalar** (8,693 subjects) — old vs new on the same
   node, a `--workers` sweep, every output bitwise against the deployed
   reference (`rbc_freesurferpost.tdb`), `fragment_count == 1`, no leftover
   temporaries, an R read of a representative element range.
4. **Scalar-sharded job array** — one task per complete reference scalar, its
   own per-scalar file, no aggregation; every task bitwise against its
   reference scalar. Aggregate wall versus the serial sum is the HPC result.
5. **Full production build** — every scalar as a job array, outputs kept, the 18
   cross-checkable scalars bitwise, every file through the HDF5 gate. Submitted
   only when every step-4 shard passed.

Every run records: ModelArrayIO commit *and* `source_tree_sha256`, Python,
nibabel, h5py and TileDB versions, allocated CPUs and RAM, TileDB concurrency
settings, source and output filesystem types, cohort hash, scalar names, input
byte total, and the exact CLI command; plus wall time, CPU utilization, peak
memory, output bytes, phase times, `fast_path_rows`, TileDB fragment and file
counts, and peak open file descriptors.

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

## CUBIC results (ladder completed 2026-09-23)

Environment: CUBIC compute nodes (Xeon Gold 6130), everything inside the ModelArray
container with a uv-managed Python 3.12 venv (numpy 2.5.3, h5py 3.16.0, tiledb
0.36.1, nibabel 5.4.2); HDF5 gated by the current ModelArray R package installed
into a user library, TileDB gated in Python; `source_tree_sha256` recorded in every
run. Warm GPFS cache; medians of three counterbalanced repetitions after one
warm-up; one loader worker; identical inputs for both converters.

| Step | Result |
|---|---|
| 0 Env + smoke | 415 tests pass on a compute node; quick-tier NIfTI and CIFTI fixtures pass both gate branches; fixture original-vs-optimized bitwise on both backends. |
| 2 Original (`9ab13dd`) vs optimized | NIfTI ABIDE-100: **1.40× HDF5, 1.46× TileDB**. MIF walkthrough-100: **3.70× HDF5, 4.11× TileDB**. RBC CIFTI 100 × 3 (area, areaPial, curv): **31.8× HDF5 (385.8 → 12.1 s), 31.6× TileDB (396.5 → 12.6 s)**. All 36 runs passed their gate; every optimized output bitwise-equal to the original (max abs diff 0.0). |
| 2b Raw fast path on production headers | After the digest fix, `fast_path_rows` = 99/100 per scalar (was 8/100); raw path 27.2 s vs nibabel path 659.2 s for 100 × 3; **both** paths bitwise-equal to the deployed old-code TileDB reference (rows aligned by source name, NaN maps equal). |
| 3 One full scalar (8,693 subjects), old vs new | `area`, 8,693 × 327,684, `--target-chunk-mb 32`, one loader worker, 1 rep each: **HDF5 11,854.8 s (3 h 17 m) → 429.9 s (7.2 min), 27.6×; TileDB 11,501.6 s (3 h 12 m) → 418.1 s (7.0 min), 27.5×** (output files 1,031 → 14). Both gates passed; optimized output bitwise-equal to the original on the full matrix (max abs diff 0.0). Optimized worker sweep on the same scalar (`fast_path_rows` = 8,692, i.e. every non-reference row): HDF5 403.9 / 120.9 / **90.6 s** at 1 / 4 / 8 workers (5,756 files/min), peak RSS ≈ 12 GB; TileDB 186.3 / 83.2 / **68.6 s** (7,603 files/min), `fragment_count` 1, peak RSS ≈ 20 GB — about 130× (HDF5) and 170× (TileDB) the original at 8 workers. All eight optimized outputs bitwise-equal to the deployed old-code reference for all 8,693 rows. |
| 4 Scalar-sharded job array (15 reference scalars) | `--array=1-15%6`, 8 CPUs / 48 GB per task, one scalar per task, HDF5 then TileDB, no aggregation: **15/15 shards passed**; per-scalar wall 112–141 s (HDF5) and 70–118 s (TileDB) at 8 workers; serial-equivalent sum **1,947 s (32 min) HDF5 / 1,466 s (24 min) TileDB** for 15 scalars that took the original run ~119 h; **array wall-clock 21.0 min** (first task start to last task end, rerun included). All 30 per-scalar outputs bitwise-equal to the deployed old-code reference. One task was cancelled by the node (signal, 3 min in, on the node step 3 had just vacated) and passed on rerun. |
| 5 Full production build (70 scalars) | `--array=1-70%10`, 8 CPUs / 48 GB per task, per-scalar HDF5 gzip-4, `--target-chunk-mb 32`, 8 workers: **70/70 scalars passed** (converter validation + current-ModelArray R gate on every file; the 15 cross-checkable scalars bitwise-equal to the deployed reference). Per-scalar wall 107–192 s, median 138 s; serial-equivalent 9,800 s (2.7 h); **array wall-clock 22.2 min** for the entire 608,510-file dataset (484 GB of per-scalar files). The original run needed 112 h for its first scalar and never finished. |

Three harness/cluster defects surfaced and were fixed by the ladder before any
production number was trusted: the container's ModelArray 0.1.5 cannot read the
current HDF5 format (attribute- vs dataset-stored `column_names`); the cohort
writers emitted CRLF line endings that broke shell consumers; and the login node's
2 h CPU-time `ulimit` propagates into SLURM jobs, killing the original converter
with SIGXCPU two-thirds of the way through a full scalar (jobs now submit with
`--propagate=NONE`). All are documented in `benchmarks/to_modelarray/cubic/README.md`.

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

The CUBIC ladder above is the acceptance path: each step's gate must pass on
production data before the next runs. Ship the fastest candidate that produces
correct, compatible output, clears the read gates, and fits the allocated
resources. Report output size and
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
- The `benchmarks/to_modelarray/` folder described above, containing the harness, the
  plotting script, `benchmarks/to_modelarray/README.md`, and a deterministic ABIDE cohort-CSV
  generator that takes a subject count, derivative list and repeat factor, so
  anyone can reproduce the small, medium and wide cohorts from the public
  bucket with no credentials.
- A pinned baseline results CSV committed under `benchmarks/to_modelarray/baselines/`, so a
  future regression is detectable by rerunning one command.
- An HPC scalar-sharding recipe — the build-side analogue of ModelArray's
  `element-splitting` — showing a SLURM job array that converts one scalar per
  task into its own per-scalar file with no aggregation step, plus a note on the
  TileDB concurrent-fragment option for a single scalar too large for one node.

Optional, decided separately:

- An `--only-scalar <name>` selector so a long-format cohort maps one scalar to
  one job-array task without pre-filtering the CSV, with a regression test that
  it produces exactly the same per-scalar output as converting that scalar
  alone.

## Non-goals

- No independent public knobs for source, scalar and write workers. (A
  `--only-scalar` selector for scalar-sharded HPC job arrays selects work, not
  worker counts, and is a separate, optional addition.)
- No output schema change; chunk/tile layout is tunable only while the output
  stays compatible and readable from R.
- No process-based or shared-memory loader *inside one conversion*. A
  scalar-sharded HPC job array — one ordinary threaded process per scalar,
  writing its own file — is a supported deployment pattern, not a process-pool
  loader, and does not violate this.
- No peak-memory admission checks, memory-mapped staging, or `--staging-dir`.
- No mandatory node-local scratch.
- No exhaustive sweep of every combination of formats, backends and settings
  before the target architecture is fixed.
