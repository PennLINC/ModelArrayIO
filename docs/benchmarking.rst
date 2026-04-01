############
Benchmarking
############

HDF5 Benchmarking and Plots
===========================

The repository includes a pytest-based benchmark suite for HDF5 write settings
with persisted artifacts and diagnostic plots.

The benchmarks use open ABIDE ALFF + func-mask volumes from S3 as templates
(``test/test_voxels_s3.py`` dataset) and then add seeded, data-adaptive
variation based on per-voxel mean/SD estimated from the downloaded data.
For a fixed benchmark row seed, generated values are deterministic and
independent of chunk/stripe geometry, so chunk-size comparisons are
apples-to-apples.
Within a given cohort size (``num_input_files``), benchmark rows now reuse the
same deterministic seed across chunk/compression/shuffle settings so storage
trade-offs compare identical synthetic values.

Quick benchmark subset (small + fast)
-------------------------------------

Runs an expanded mini-grid (chunk size, compression, gzip level, shuffle, and
``num_input_files`` at 100 and 1000) for fast local comparison.
Because template data are loaded from public S3, keep internet access enabled
when running these benchmarks. Since the ABIDE bucket is public, set
``MODELARRAYIO_S3_ANON=1`` to use unsigned S3
requests (no AWS credentials required).
All benchmark runs use the full real group-mask voxel set (no voxel downsampling)
so file size and throughput trends are directly comparable across run kinds.
Benchmarks now fail fast if S3 templates are unavailable, instead of silently
falling back to a tiny local synthetic mask.

.. code-block:: console

   # From repository root
   MODELARRAYIO_S3_ANON=1 PYTHONPATH=src pytest -m benchmark_quick test/test_h5_benchmarks.py -q

Medium benchmark sweep (exclude largest cohort)
------------------------------------------------

.. code-block:: console

   # Includes cohort sizes: 100, 1000, 10000
   MODELARRAYIO_S3_ANON=1 PYTHONPATH=src pytest -m benchmark_medium test/test_h5_benchmarks.py -q

Full benchmark sweep
--------------------

.. code-block:: console

   # Includes cohort sizes: 100, 1000, 10000, 40000
   MODELARRAYIO_S3_ANON=1 PYTHONPATH=src pytest -m benchmark_full test/test_h5_benchmarks.py -q

Parallel full sweep (faster on multi-core machines)
----------------------------------------------------

.. code-block:: console

   # Requires pytest-xdist; writes per-worker CSV/meta files to avoid write races
   MODELARRAYIO_S3_ANON=1 PYTHONPATH=src pytest -n auto -m benchmark_full test/test_h5_benchmarks.py -q

Default benchmark outputs:

* Serial runs:
  * ``benchmark_results/h5_benchmark_results.csv``
  * ``benchmark_results/run_meta.json``
* Parallel runs (``pytest -n ...``):
  * ``benchmark_results/h5_benchmark_results_<worker>.csv``
  * ``benchmark_results/run_meta_<worker>.json``

You can override output location by setting
``MODELARRAYIO_BENCHMARK_RESULTS_DIR``. Benchmark writers and the plotting
script's default CSV auto-discovery both use this directory.

Timing columns in the CSV:

* ``elapsed_seconds``: end-to-end benchmark time for the row
* ``data_generation_seconds``: synthetic data generation time
* ``hdf5_write_seconds``: HDF5 dataset creation + writes

For storage-setting comparisons (chunking/compression), prefer
``hdf5_write_seconds``; use ``elapsed_seconds`` when you want total runtime.

Generate diagnostic plots from saved CSV
----------------------------------------

.. code-block:: console

   Rscript test/plot_h5_benchmarks.R

The plotting script is implemented in R with ``ggplot2`` and currently requires
``ggplot2``, ``dplyr``, ``tidyr``, and ``patchwork``.
By default, plotting filters to a single comparable slice of results
(``run_kind`` auto-prefers full, and ``sampled_voxels`` auto-picks the max
present in that run kind). Use ``--run-kind`` and ``--sampled-voxels`` to
override, and invalid explicit filters fail with a clear error.
When ``--results-csv`` is omitted, plotting automatically loads all
``h5_benchmark_results*.csv`` files in the benchmark results directory
(``benchmark_results/`` by default, or
``MODELARRAYIO_BENCHMARK_RESULTS_DIR`` when set), including per-worker outputs
from parallel runs.
All scaling and trade-off plots include all available compression variants; the
compression program is encoded by color and variants from the same program
(for example ``gzip-1``, ``gzip-4``, ``gzip-9``) are differentiated by marker
shape.

The plotting script now generates a curated faceted SVG summary:

* ``benchmark_results/plots/h5_benchmark_summary.svg`` (analysis output)
* ``docs/_static/h5_benchmark_summary.svg`` (README figure)

These can be regenerated later without rerunning benchmarks. The CSV includes
file size in bytes and GiB (``output_size_bytes`` and ``output_size_gb``),
plus split timing columns for generation vs write time.
