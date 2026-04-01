#######
Outputs
#######

This page describes what each CLI command writes, how files are named, and what data
is stored inside each output artifact.


*****************
Commands Overview
*****************

The commands fall into two groups:

- ``*-to-h5`` commands: convert input neuroimaging data into either:
  - one or more HDF5 files (``--backend hdf5``), or
  - one or more TileDB directories (``--backend tiledb``).
- ``h5-to-*`` commands: convert analysis results stored in an HDF5 file into image files.


*********************
nifti-to-h5 (volumes)
*********************

Default output name (HDF5 backend): ``voxelarray.h5``.

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
- Example: ``--scalar-columns alpha beta --output voxelarray.h5`` writes:
  - ``alpha_voxelarray.h5``
  - ``beta_voxelarray.h5``
- The same prefix rule also applies to TileDB output paths.


********************
cifti-to-h5 (CIFTI)
********************

Default output name (HDF5 backend): ``greyordinatearray.h5``.

HDF5 output contents:

- ``greyordinates`` dataset:
  - transposed table with rows for ``vertex_id`` and ``structure_id``.
  - attribute ``column_names = ['vertex_id', 'structure_id']``.
  - attribute ``structure_names`` listing CIFTI brain structures.
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
- Example: ``--scalar-columns alpha beta --output greyordinatearray.h5`` writes:
  - ``alpha_greyordinatearray.h5``
  - ``beta_greyordinatearray.h5``
- The same prefix rule also applies to TileDB output paths.


******************
mif-to-h5 (fixels)
******************

Default output name (HDF5 backend): ``fixelarray.h5``.

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
- Example: ``--scalar-columns alpha beta --output fixelarray.h5`` writes:
  - ``alpha_fixelarray.h5``
  - ``beta_fixelarray.h5``
- The same prefix rule also applies to TileDB output paths.


***********************************
h5-to-* commands (result exporters)
***********************************

These commands read statistical results from:

- ``results/<analysis_name>/results_matrix`` (shape: ``(n_results, n_elements)``).

Result names are read in this order:

- ``results_matrix.attrs['colnames']`` (if present),
- ``results/<analysis_name>/column_names`` dataset,
- ``results/<analysis_name>/results_matrix/column_names`` dataset,
- fallback names: ``component001``, ``component002``, ...

Any spaces or ``/`` in result names are replaced with ``_`` in filenames.


h5-to-nifti
===========

Writes one file per result to ``--output-dir``:

- ``<analysis_name>_<result_name><output_ext>`` (default extension ``.nii.gz``).
- If a result name contains ``p.value``, an additional file is written:
  ``<analysis_name>_<result_name_with_1m.p.value><output_ext>``,
  containing ``1 - p.value``.

Each output volume uses ``--group-mask-file`` to map vectorized results back into 3D space.


h5-to-cifti
===========

Writes one CIFTI dscalar file per result to ``--output-dir``:

- ``<analysis_name>_<result_name>.dscalar.nii``.
- If a result name contains ``p.value``, also writes the ``1 - p.value`` companion file
  with ``1m.p.value`` in its name.

The header is taken from ``--example-cifti`` (or from the first cohort ``source_file`` if
``--cohort-file`` is used instead).


h5-to-mif
=========

Writes one MIF file per result to ``--output-dir``:

- ``<analysis_name>_<result_name>.mif``.
- If a result name contains ``p.value``, also writes the ``1 - p.value`` companion file
  with ``1m.p.value`` in its name.

Also copies these files into ``--output-dir``:

- ``--index-file``
- ``--directions-file``

The output MIF geometry/header template is taken from ``--example-mif`` (or from the first
cohort ``source_file`` if ``--cohort-file`` is used instead).
