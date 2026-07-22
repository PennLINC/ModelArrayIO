"""
NIfTI (Voxel-wise) Data Conversion
==================================

For imaging data in NIfTI format, use the ``modelarrayio to-modelarray`` command to convert
the NIfTI files to the HDF5 format (``.h5``) used by **ModelArray**,
and ``modelarrayio export-results`` to export results back to NIfTI.
The voxel workflow is very similar to the fixel workflow
(:ref:`sphx_glr_auto_examples_plot_mif_workflow.py`).
"""

# %%
# Prepare data
# ------------
#
# To convert a list of NIfTI files to ``.h5`` format, you need:
#
# 1. **A cohort CSV** describing every NIfTI file to include.
# 2. **A group mask** — only voxels inside the group mask are kept during conversion.
# 3. **Subject-specific masks** — voxels outside each subject's mask are set to ``NaN`` after
#    conversion. Every cohort row must provide a mask path in ``source_mask_file``. If you do
#    not have distinct subject masks, use the group mask path for each row.
#
# The cohort CSV may use either of these layouts:
#
# * **Long format:** one row per subject and scalar, with the fixed columns ``scalar_name`` and
#   ``source_file``.
# * **Wide format:** one row per subject, with a separate file-path column for each scalar
#   (for example, ``FA`` and ``MD``). Pass those column names to ``--scalar-columns``.
#
# In either layout, the ``source_mask_file`` column and a valid mask path for every row are
# required. Other columns, such as subject IDs and demographics, may appear alongside the path
# columns.

# %%
# Example folder structure
# ------------------------
#
# .. code-block:: text
#
#     /home/username/myProject/data
#     |
#     ├── cohort_long.csv
#     ├── cohort_wide.csv
#     ├── group_mask.nii.gz
#     │
#     ├── FA
#     │   ├── sub-01_FA.nii.gz
#     │   ├── sub-02_FA.nii.gz
#     │   ├── sub-03_FA.nii.gz
#     │   └── ...
#     │
#     ├── MD
#     │   ├── sub-01_MD.nii.gz
#     │   ├── sub-02_MD.nii.gz
#     │   ├── sub-03_MD.nii.gz
#     │   └── ...
#     │
#     ├── individual_masks
#     │   ├── sub-01_mask.nii.gz
#     │   ├── sub-02_mask.nii.gz
#     │   ├── sub-03_mask.nii.gz
#     │   └── ...
#     └── ...
#
# Long-format cohort CSV
# ----------------------
#
# Corresponding ``cohort_long.csv`` for scalars FA and MD:
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - **scalar_name** *(required)*
#      - **source_file** *(required)*
#      - **source_mask_file** *(required)*
#      - subject_id
#      - age
#      - sex
#    * - FA
#      - /home/username/myProject/data/FA/sub-01_FA.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-01_mask.nii.gz
#      - sub-01
#      - 10
#      - F
#    * - FA
#      - /home/username/myProject/data/FA/sub-02_FA.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-02_mask.nii.gz
#      - sub-02
#      - 20
#      - M
#    * - FA
#      - /home/username/myProject/data/FA/sub-03_FA.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-03_mask.nii.gz
#      - sub-03
#      - 15
#      - F
#    * - MD
#      - /home/username/myProject/data/MD/sub-01_MD.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-01_mask.nii.gz
#      - sub-01
#      - 10
#      - F
#    * - MD
#      - /home/username/myProject/data/MD/sub-02_MD.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-02_mask.nii.gz
#      - sub-02
#      - 20
#      - M
#    * - MD
#      - /home/username/myProject/data/MD/sub-03_MD.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-03_mask.nii.gz
#      - sub-03
#      - 15
#      - F
#    * - ...
#      - ...
#      - ...
#      - ...
#      - ...
#      - ...
#
# Notes:
#
# * Column order does not matter.
# * ``source_mask_file`` must contain a valid subject-mask path for every row.
# * ``--mask`` separately supplies the required group mask.
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
#      - **source_mask_file** *(required)*
#      - age
#      - sex
#    * - sub-01
#      - /home/username/myProject/data/FA/sub-01_FA.nii.gz
#      - /home/username/myProject/data/MD/sub-01_MD.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-01_mask.nii.gz
#      - 10
#      - F
#    * - sub-02
#      - /home/username/myProject/data/FA/sub-02_FA.nii.gz
#      - /home/username/myProject/data/MD/sub-02_MD.nii.gz
#      - /home/username/myProject/data/individual_masks/sub-02_mask.nii.gz
#      - 20
#      - M
#    * - ...
#      - ...
#      - ...
#      - ...
#      - ...
#      - ...
#
# ``FA`` and ``MD`` are user-defined scalar column names. The required ``source_mask_file``
# value is applied to every scalar for that subject.

# %%
# Convert a long-format cohort
# ----------------------------
#
# Long-format cohorts combine all scalars into one output by default. The
# ``--no-split-files`` flag below makes that choice explicit:
#
# .. code-block:: console
#
#     # activate your conda environment first
#     conda activate <env_name>
#
#     modelarrayio to-modelarray \
#         --mask           /home/username/myProject/data/group_mask.nii.gz \
#         --cohort-file    /home/username/myProject/data/cohort_long.csv \
#         --no-split-files \
#         --output         /home/username/myProject/data/modelarray.h5
#
# This creates one ``modelarray.h5`` containing both ``scalars/FA`` and ``scalars/MD``.
# To write one file per scalar instead, use ``--split-files`` with the same output basename:
#
# .. code-block:: console
#
#     modelarrayio to-modelarray \
#         --mask        /home/username/myProject/data/group_mask.nii.gz \
#         --cohort-file /home/username/myProject/data/cohort_long.csv \
#         --split-files \
#         --output      /home/username/myProject/data/modelarray.h5
#
# The split command creates ``FA_modelarray.h5`` and ``MD_modelarray.h5``. You can then use
# `ModelArray <https://pennlinc.github.io/ModelArray/>`_ to run statistical analyses on either
# the combined output or the scalar-specific outputs.

# %%
# Convert a wide-format cohort
# ----------------------------
#
# Name each scalar path column with ``--scalar-columns``:
#
# .. code-block:: console
#
#     modelarrayio to-modelarray \
#         --mask           /home/username/myProject/data/group_mask.nii.gz \
#         --cohort-file    /home/username/myProject/data/cohort_wide.csv \
#         --scalar-columns FA MD \
#         --output         /home/username/myProject/data/modelarray.h5
#
# Wide cohorts write one output per scalar by default. This command creates
# ``FA_modelarray.h5`` and ``MD_modelarray.h5``. To override that default, add
# ``--no-split-files``:
#
# .. code-block:: console
#
#     modelarrayio to-modelarray \
#         --mask            /home/username/myProject/data/group_mask.nii.gz \
#         --cohort-file     /home/username/myProject/data/cohort_wide.csv \
#         --scalar-columns  FA MD \
#         --no-split-files \
#         --output          /home/username/myProject/data/modelarray.h5
#
# This creates one ``modelarray.h5`` containing both scalar groups.

# %%
# Convert result .h5 back to NIfTI
# --------------------------------
#
# After running **ModelArray** and obtaining statistical results inside ``FA_modelarray.h5``
# (suppose the analysis name is ``"mylm"``), use ``modelarrayio export-results`` to export them
# as NIfTI files.
#
# .. code-block:: console
#
#     modelarrayio export-results \
#         --mask            /home/username/myProject/data/group_mask.nii.gz \
#         --analysis-name   mylm \
#         --input-hdf5      /home/username/myProject/data/FA_modelarray.h5 \
#         --output-dir      /home/username/myProject/data/FA_stats
#
# All converted volume data are saved as ``float32`` and compressed (``.nii.gz``) by default.
# Pass ``--no-compress`` to write uncompressed ``.nii`` files instead.
# Results in ``FA_stats`` can be viewed with any NIfTI image viewer.
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
# an image called ``*_model.nobs.nii*``.  With subject-specific masks, this image may be
# inhomogeneous across voxels.
#
# Voxels that did not have sufficient subjects (due to subject-specific masking) are stored as
# ``NaN`` in the HDF5 file.  How different viewers display these voxels:
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
