# ModelArrayIO benchmarks

Each benchmark suite lives in its own folder, named after the `modelarrayio` subcommand it
measures (with underscores). A suite is self-contained: its scripts, committed fixtures, pinned
results (`baselines/`), its own `README.md`, and the gitignored inputs and outputs it produces.
Rendered reports (PDFs) are not committed; they are regenerated from the pinned CSVs.

| Suite | Measures | Guide |
| --- | --- | --- |
| [`to_modelarray/`](to_modelarray/) | `modelarrayio to-modelarray` runtime and output correctness across NIfTI, CIFTI and MIF on HDF5 and TileDB, including the original-vs-optimized comparison and the CUBIC (HPC) validation ladder | [`to_modelarray/README.md`](to_modelarray/README.md) |

## Suite layout

Each suite keeps the same three workspace folders, ignored by Git through the
`benchmarks/*/{data,cohorts,results}/` patterns in `.gitignore`:

- `data/`: downloaded or synthesized input files.
- `cohorts/`: generated cohort CSVs. Locally materialized cohorts contain absolute paths into
  `data/`, so regenerate them (or rewrite the path prefix) after moving the checkout.
- `results/<run-id>/`: per-run outputs, logs, timings and `run_meta` files. Set
  `MODELARRAYIO_BENCHMARK_RESULTS_DIR` to write them elsewhere (for example cluster scratch).

Run every command from the repository root.

## Adding a suite

Create `benchmarks/<subcommand>/` with its own `README.md`, `fixtures/` and `baselines/`, and
add a row to the table above. When a second suite needs a tool or input that
currently lives in `to_modelarray/` (the shared modules `harness_common.py` and
`output_compare.py`, the cohort generators, `source_identity.py`, the old-commit launcher in
`baselines/`, or a downloaded cohort), move it to `benchmarks/common/` and update both suites;
until then, keep each tool and its data with the suite that uses it.
