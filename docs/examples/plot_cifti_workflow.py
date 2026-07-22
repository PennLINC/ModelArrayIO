"""
CIFTI (Greyordinate-wise) Data Conversion
=========================================

For imaging data in CIFTI format, use the ``modelarrayio to-modelarray`` command to convert
the CIFTI files to the HDF5 format (``.h5``) used by **ModelArray**,
and ``modelarrayio export-results`` to export results back to CIFTI.
The CIFTI workflow is very similar to the MIF workflow
(:ref:`sphx_glr_auto_examples_plot_mif_workflow.py`).
"""

# %%
# Prepare data
# ------------
#
# To convert a list of CIFTI files to ``.h5`` format, you need:
#
# 1. **A cohort CSV** describing every CIFTI file to include.
#
# The cohort CSV may use either of these layouts:
#
# * **Long format:** one row per subject and scalar, with the fixed columns ``scalar_name`` and
#   ``source_file``.
# * **Wide format:** one row per subject, with a separate file-path column for each scalar
#   (for example, ``FA`` and ``MD``). Pass those column names to ``--scalar-columns``.
#
# Other columns, such as subject IDs and demographics, may appear alongside the path columns.

# %%
# Example folder structure
# ------------------------
#
# .. code-block:: text
#
#     /home/username/myProject/data
#     |
#     ├── cohort_FA.csv
#     ├── cohort_wide.csv
#     │
#     ├── FA
#     │   ├── sub-01_FA.dscalar.nii
#     │   ├── sub-02_FA.dscalar.nii
#     │   ├── sub-03_FA.dscalar.nii
#     │   └── ...
#     │
#     ├── MD
#     │   ├── sub-01_MD.dscalar.nii
#     │   ├── sub-02_MD.dscalar.nii
#     │   ├── sub-03_MD.dscalar.nii
#     │   └── ...
#     │
#     └── ...
#
# Long-format cohort CSV
# ----------------------
#
# Corresponding ``cohort_FA.csv`` for scalar FA:
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - **scalar_name** *(required)*
#      - **source_file** *(required)*
#      - subject_id
#      - age
#      - sex
#    * - FA
#      - /home/username/myProject/data/FA/sub-01_FA.dscalar.nii
#      - sub-01
#      - 10
#      - F
#    * - FA
#      - /home/username/myProject/data/FA/sub-02_FA.dscalar.nii
#      - sub-02
#      - 20
#      - M
#    * - FA
#      - /home/username/myProject/data/FA/sub-03_FA.dscalar.nii
#      - sub-03
#      - 15
#      - F
#    * - ...
#      - ...
#      - ...
#      - ...
#      - ...
#
# Notes:
#
# * Column order does not matter.
# * Values are case-sensitive — folder names, file names, and scalar names must match exactly
#   between the CSV and disk.

# %%
# Wide-format cohort CSV
# ----------------------
#
# A wide CSV stores all scalar paths for a subject on one row. For example,
# ``cohort_wide.csv`` can contain both FA and MD:
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - subject_id
#      - **FA**
#      - **MD**
#      - age
#      - sex
#    * - sub-01
#      - /home/username/myProject/data/FA/sub-01_FA.dscalar.nii
#      - /home/username/myProject/data/MD/sub-01_MD.dscalar.nii
#      - 10
#      - F
#    * - sub-02
#      - /home/username/myProject/data/FA/sub-02_FA.dscalar.nii
#      - /home/username/myProject/data/MD/sub-02_MD.dscalar.nii
#      - 20
#      - M
#    * - ...
#      - ...
#      - ...
#      - ...
#      - ...
#
# ``FA`` and ``MD`` are user-defined scalar column names. All files for a scalar must use
# compatible CIFTI axes.

# %%
# Convert CIFTI files to HDF5
# ---------------------------
#
# Using the FA dataset from the example above:
#
# .. code-block:: console
#
#     # activate your conda environment first
#     conda activate <env_name>
#
#     modelarrayio to-modelarray \
#         --cohort-file     /home/username/myProject/data/cohort_FA.csv \
#         --output          /home/username/myProject/data/FA.h5
#
# This produces ``FA.h5`` in ``/home/username/myProject/data``.  You can then use
# `ModelArray <https://pennlinc.github.io/ModelArray/>`_ to run statistical analyses on it.

# %%
# Convert a wide-format cohort
# ----------------------------
#
# Name each scalar path column with ``--scalar-columns``:
#
# .. code-block:: console
#
#     modelarrayio to-modelarray \
#         --cohort-file    /home/username/myProject/data/cohort_wide.csv \
#         --scalar-columns FA MD \
#         --output         /home/username/myProject/data/modelarray.h5
#
# Wide cohorts write one output per scalar by default. This command creates
# ``FA_modelarray.h5`` and ``MD_modelarray.h5``. Add ``--no-split-files`` to store both
# scalars in the single ``modelarray.h5`` output instead.

# %%
# Convert result .h5 back to CIFTI
# --------------------------------
#
# After running **ModelArray** and obtaining statistical results inside ``FA.h5`` (suppose the
# analysis name is ``"mylm"``), use ``modelarrayio export-results`` to export them as CIFTI files.
#
# Supply either ``--cohort-file`` (the first ``source_file`` entry is used as a header template)
# or ``--example-file`` (an explicit template path) — these two flags are mutually exclusive.
#
# .. code-block:: console
#
#     modelarrayio export-results \
#         --cohort-file     /home/username/myProject/data/cohort_FA.csv \
#         --analysis-name   mylm \
#         --input-hdf5      /home/username/myProject/data/FA.h5 \
#         --output-dir      /home/username/myProject/data/FA_stats
#
# All converted volume data are saved as ``float32``.  Results in ``FA_stats`` can be viewed
# with any CIFTI image viewer.
#
# .. warning::
#
#    If ``--output-dir`` already exists, ``modelarrayio export-results`` will not delete it — you will
#    see ``WARNING: Output directory exists``.  Existing files that are **not** part of the
#    current output list are left unchanged.  Existing files that **are** part of the current
#    output list will be overwritten.  To avoid confusion, consider manually deleting the output
#    directory before re-running ``modelarrayio export-results``.

# %%
# Number-of-observations image
# ----------------------------
#
# If you requested ``nobs`` during model fitting in ModelArray, after conversion you will find
# an image called ``*_model.nobs.dscalar.nii*``.  With subject-specific masks, this image may be
# inhomogeneous across voxels.
#
# Voxels that did not have sufficient subjects (due to subject-specific masking) are stored as
# ``NaN`` in the HDF5 file.  How different viewers display these greyordinates:
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - Viewer
#      - Regular voxel
#      - Voxel without sufficient subjects
#    * - nibabel (Python)
#      - e.g. ``nobs: 209.0``, ``p.value: 0.01``
#      - ``NaN``
#    * - MRtrix mrview
#      - e.g. ``nobs: 209``, ``p.value: 0.01``
#      - ``?``
#    * - ITK-SNAP
#      - e.g. ``nobs: 209``, ``p.value: 0.01``
#      - ``0`` (displayed, but excluded when thresholding)

# %%
# Additional help
# ---------------
#
# Full argument documentation is available from the command line:
#
# .. code-block:: console
#
#     modelarrayio to-modelarray --help
#     modelarrayio export-results --help
#
# or in the :doc:`/usage` page of this documentation.
