============
ModelArrayIO
============

.. image:: https://img.shields.io/pypi/v/modelarrayio.svg
   :target: https://pypi.python.org/pypi/modelarrayio/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/modelarrayio.svg
   :target: https://pypi.python.org/pypi/modelarrayio/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

.. image:: https://readthedocs.org/projects/modelarrayio/badge/?version=latest
   :target: http://modelarrayio.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/PennLINC/ModelArrayIO/actions/workflows/tox.yml/badge.svg
   :target: https://github.com/PennLINC/ModelArrayIO/actions/workflows/tox.yml
   :alt: GitHub Actions: Tox

.. image:: https://codecov.io/gh/pennlinc/modelarrayio/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/pennlinc/modelarrayio
   :alt: Codecov

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: Code style: ruff

**ModelArrayIO** is a Python package that converts between neuroimaging formats (fixel ``.mif``, voxel NIfTI, CIFTI-2 dscalar) and the HDF5 (``.h5``) layout used by the R package `ModelArray <https://pennlinc.github.io/ModelArray/>`_. It can also write ModelArray statistical results back to imaging formats.

**Relationship to ConFixel:** The earlier project `ConFixel <https://github.com/PennLINC/ConFixel>`_ is superseded by ModelArrayIO. The ConFixel repository is retained for history (including links from publications) and will be archived; new work should use this repository.

Documentation for installation and usage: `ModelArrayIO on GitHub <https://github.com/PennLINC/ModelArrayIO#installation>`_ (this README). For conda, HDF5 libraries, and installing the ModelArray R package, see the ModelArray vignette `Installation <https://pennlinc.github.io/ModelArray/articles/installations.html>`_.

.. readme-overview-figure-placeholder

.. image:: docs/_static/overview_structure.png
   :align: center
   :alt: Overview

.. readme-overview-figure-end

ModelArrayIO provides three converter areas, each with import and export commands:

Once ModelArrayIO is installed, these commands are available in your terminal:

* **Fixel-wise** data (MRtrix ``.mif``):

  * ``.mif`` → ``.h5``: ``confixel`` (CLI name kept for compatibility with earlier ConFixel workflows)
  * ``.h5`` → ``.mif``: ``fixelstats_write``

* **Voxel-wise** data (NIfTI):

  * NIfTI → ``.h5``: ``convoxel``
  * ``.h5`` → NIfTI: ``volumestats_write``

* **Greyordinate-wise** data (CIFTI-2):

  * CIFTI-2 → ``.h5``: ``concifti``
  * ``.h5`` → CIFTI-2: ``ciftistats_write``

Installation
============

MRtrix (required for fixel ``.mif`` only)
-----------------------------------------

For fixel-wise ``.mif`` conversion, the ``confixel`` / ``fixelstats_write`` tools use MRtrix ``mrconvert``. Install MRtrix from `MRtrix’s webpage <https://www.mrtrix.org/download/>`_ if needed. Run ``mrview`` in the terminal to verify the installation.

If your data are voxel-wise or CIFTI only, you can skip this step.

Install ModelArrayIO
--------------------

You may want a conda environment first—see `ModelArray: Installation <https://pennlinc.github.io/ModelArray/articles/installations.html>`_. If MRtrix is installed in that environment, install ModelArrayIO in the same environment.

Install from GitHub:

.. code-block:: console

   git clone https://github.com/PennLINC/ModelArrayIO.git
   cd ModelArrayIO
   pip install .   # build via pyproject.toml

Editable install for development:

.. code-block:: console

   # From the repository root
   pip install -e .

With ``hatch`` installed, you can build wheels/sdist locally:

.. code-block:: console

   hatch build
   pip install dist/*.whl

How to use
==========

We provide a `walkthrough for fixel-wise data <notebooks/walkthrough_fixel-wise_data.md>`_ (``confixel`` / ``fixelstats_write``) and a `walkthrough for voxel-wise data <notebooks/walkthrough_voxel-wise_data.md>`_ (``convoxel`` / ``volumestats_write``).

Together with `ModelArray <https://pennlinc.github.io/ModelArray/>`_, see the `combined walkthrough <https://pennlinc.github.io/ModelArray/articles/walkthrough.html>`_ with example fixel-wise data (ModelArray + ModelArrayIO).

CLI help:

.. code-block:: console

   confixel --help

Use the same pattern for ``convoxel``, ``concifti``, ``fixelstats_write``, ``volumestats_write``, and ``ciftistats_write``.

Storage backends: HDF5 and TileDB
=================================

ModelArrayIO supports two on-disk backends for the subject-by-element matrix:

* HDF5 (default), implemented in ``modelarrayio/h5_storage.py``
* TileDB, implemented in ``modelarrayio/tiledb_storage.py``

Both backends expose a similar API:

* create a dense 2D array ``(subjects, items)`` and write all values at once
* create an empty array with the same shape and write by column stripes
* write/read column names alongside the data

Notes and minor differences:

* Chunking vs tiling: HDF5 uses chunks; TileDB uses tiles. We compute tile sizes analogous to chunk sizes to keep write/read patterns similar.
* Compression: HDF5 uses ``gzip`` by default; TileDB defaults to ``zstd`` with shuffle for better speed/ratio. You can switch to ``gzip`` for parity.
* Metadata: HDF5 stores ``column_names`` as a dataset attribute; TileDB stores names as JSON metadata on the array/group.
* Layout: Both backends keep dimensions in the same order and use zero-based indices.

HDF5 Benchmarking and Plots
===========================

Benchmark commands, environment setup, output artifacts, and plotting workflow
are documented in the dedicated benchmarking guide:
`docs/benchmarking.rst <docs/benchmarking.rst>`_.

Faceted HDF5 benchmark summary:

.. image:: docs/_static/h5_benchmark_summary.svg
   :align: center
   :alt: Faceted HDF5 benchmark summary (write time, output size, throughput, and Pareto trade-off)
