#######
Outputs
#######

This page describes what each CLI command writes, how files are named, and what data
is stored inside each output artifact.


*****************
Commands Overview
*****************

The commands fall into two groups:

- ``to-modelarray``: convert input neuroimaging data into either:
  - one or more HDF5 files (``--backend hdf5``), or
  - one or more TileDB directories (``--backend tiledb``).
  The modality (NIfTI, CIFTI, or MIF/fixel) is autodetected from the source file
  extensions in the cohort file.
- ``export-results``: convert analysis results stored in an HDF5 file into image files.
  The modality is inferred from which arguments are provided (``--mask`` for NIfTI,
  ``--index-file``/``--directions-file`` for MIF, ``--cohort-file``/``--example-file``
  for CIFTI).


***********************
to-modelarray (volumes)
***********************

Triggered when source files in the cohort have ``.nii`` or ``.nii.gz`` extensions.
Requires ``--mask``.

HDF5 output contents:

- ``voxels`` dataset:
  - transposed voxel table with rows for ``voxel_id``, ``i``, ``j``, ``k``.
  - attribute ``column_names = ['voxel_id', 'i', 'j', 'k']``.
- Per scalar:
  - ``scalars/<scalar_name>/values`` with shape ``(n_subjects, n_voxels)``.
  - ``scalars/<scalar_name>/column_names`` listing source file names.

TileDB output contents:

- Per scalar dense array at ``scalars/<scalar_name>/values`` with shape
  ``(n_subjects, n_voxels)``.
- Column names are stored in array metadata (``column_names``).

When ``--scalar-columns`` is provided:

- Output is split by scalar column name.
- Example: ``--scalar-columns alpha beta --output modelarray.h5`` writes:
  - ``alpha_modelarray.h5``
  - ``beta_modelarray.h5``
- The same prefix rule also applies to TileDB output paths.


*********************
to-modelarray (CIFTI)
*********************

Triggered when source files in the cohort have a CIFTI compound extension
(e.g. ``.dscalar.nii``, ``.pscalar.nii``, ``.pconn.nii``).

HDF5 output contents:

- ``greyordinates`` dataset (dscalar):
  - transposed table with rows for ``vertex_id`` and ``structure_id``.
  - attribute ``column_names = ['vertex_id', 'structure_id']``.
  - attribute ``structure_names`` listing CIFTI brain structures.
- ``parcels/parcel_id`` string dataset (pscalar), or
  ``parcels/parcel_id_from`` and ``parcels/parcel_id_to`` (pconn).
- Per scalar:
  - ``scalars/<scalar_name>/values`` with shape ``(n_subjects, n_greyordinates)``.
  - ``scalars/<scalar_name>/column_names`` listing source file names.

TileDB output contents:

- Per scalar dense array at ``scalars/<scalar_name>/values`` with shape
  ``(n_subjects, n_greyordinates)``.
- Column names metadata is written on each scalar matrix.
- An explicit TileDB array is also written at ``scalars/<scalar_name>/column_names``.

When ``--scalar-columns`` is provided:

- Output is split by scalar column name.
- Example: ``--scalar-columns alpha beta --output modelarray.h5`` writes:
  - ``alpha_modelarray.h5``
  - ``beta_modelarray.h5``
- The same prefix rule also applies to TileDB output paths.


**************************
to-modelarray (MIF/fixels)
**************************

Triggered when source files in the cohort have a ``.mif`` extension.
Requires ``--index-file`` and ``--directions-file``.

HDF5 output contents:

- ``fixels`` dataset:
  - transposed fixel table (``fixel_id``, coordinates/directions metadata from input fixel DB).
  - attribute ``column_names`` containing table column names.
- ``voxels`` dataset:
  - transposed voxel table with ``voxel_id``, ``i``, ``j``, ``k``.
  - attribute ``column_names`` containing table column names.
- Per scalar:
  - ``scalars/<scalar_name>/values`` with shape ``(n_subjects, n_fixels)``.
  - ``scalars/<scalar_name>/column_names`` listing source file names.

TileDB output contents:

- Per scalar dense array at ``scalars/<scalar_name>/values`` with shape
  ``(n_subjects, n_fixels)``.
- Column names are stored in array metadata (``column_names``).

When ``--scalar-columns`` is provided:

- Output is split by scalar column name.
- Example: ``--scalar-columns alpha beta --output modelarray.h5`` writes:
  - ``alpha_modelarray.h5``
  - ``beta_modelarray.h5``
- The same prefix rule also applies to TileDB output paths.


*********************************
export-results (result exporters)
*********************************

This command reads statistical results from:

- ``results/<analysis_name>/results_matrix`` (shape: ``(n_results, n_elements)``).

Result names are read in this order:

- ``results_matrix.attrs['colnames']`` (if present),
- ``results/<analysis_name>/column_names`` dataset,
- ``results/<analysis_name>/results_matrix/column_names`` dataset,
- fallback names: ``component001``, ``component002``, ...

Any spaces or ``/`` in result names are replaced with ``_`` in filenames.


export-results (NIfTI)
======================

Triggered by providing ``--mask``.

Writes one file per result to ``--output-dir``:

- ``<analysis_name>_<result_name>.nii.gz``.
- If a result name contains ``p.value``, an additional file is written:
  ``<analysis_name>_<result_name_with_1m.p.value>.nii.gz``,
  containing ``1 - p.value``.

Each output volume uses ``--mask`` to map vectorized results back into 3D space.
Pass ``--no-compress`` to write uncompressed ``.nii`` files instead.


export-results (CIFTI)
======================

Triggered by providing ``--cohort-file`` or ``--example-file`` (without
``--mask`` or ``--index-file``/``--directions-file``).

Writes one CIFTI file per result to ``--output-dir``, using the extension that
matches the example file (e.g. ``.dscalar.nii``, ``.pscalar.nii``, ``.pconn.nii``):

- ``<analysis_name>_<result_name>.<ext>``.
- If a result name contains ``p.value``, also writes the ``1 - p.value`` companion file
  with ``1m.p.value`` in its name.

The header is taken from ``--example-file`` (or from the first cohort ``source_file`` if
``--cohort-file`` is used instead).


export-results (MIF/fixels)
===========================

Triggered by providing ``--index-file`` and ``--directions-file``.

Writes one MIF file per result to ``--output-dir``:

- ``<analysis_name>_<result_name>.mif.gz``.
- If a result name contains ``p.value``, also writes the ``1 - p.value`` companion file
  with ``1m.p.value`` in its name.

Also copies these files into ``--output-dir``:

- ``--index-file``
- ``--directions-file``

The output MIF geometry/header template is taken from ``--example-file`` (or from the first
cohort ``source_file`` if ``--cohort-file`` is used instead).
Pass ``--no-compress`` to write uncompressed output where applicable.
