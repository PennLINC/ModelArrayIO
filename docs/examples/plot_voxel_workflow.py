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
# 1. **A cohort CSV** describing every NIfTI file to include (one CSV per scalar recommended).
# 2. **A group mask** — only voxels inside the group mask are kept during conversion.
# 3. **Subject-specific masks** *(optional)* — voxels outside each subject's mask are set to
#    ``NaN`` after conversion.  If you do not have per-subject masks, supply the group mask for
#    every subject (see the CSV example below).
#
# Cohort CSV columns (names are fixed, not user-defined):
#
# * ``scalar_name`` — which metric is being analysed (e.g., ``FA``)
# * ``source_file`` — path to the subject's NIfTI file
# * ``source_mask_file`` — path to the subject-specific mask (or the group mask if none exists)

# %%
# Example folder structure
# ------------------------
#
# .. code-block:: text
#
#     /home/username/myProject/data
#     |
#     ├── cohort_FA.csv
#     ├── group_mask.nii.gz
#     │
#     ├── FA
#     │   ├── sub-01_FA.nii.gz
#     │   ├── sub-02_FA.nii.gz
#     │   ├── sub-03_FA.nii.gz
#     │   └── ...
#     │
#     ├── individual_masks
#     │   ├── sub-01_mask.nii.gz
#     │   ├── sub-02_mask.nii.gz
#     │   ├── sub-03_mask.nii.gz
#     │   └── ...
#     └── ...
#
# Corresponding ``cohort_FA.csv`` for scalar FA:
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
# * Values are case-sensitive — folder names, file names, and scalar names must match exactly
#   between the CSV and disk.

# %%
# Convert NIfTI files to HDF5
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
#         --mask /home/username/myProject/data/group_mask.nii.gz \
#         --cohort-file     /home/username/myProject/data/cohort_FA.csv \
#         --output          /home/username/myProject/data/FA.h5
#
# This produces ``FA.h5`` in ``/home/username/myProject/data``.  You can then use
# `ModelArray <https://pennlinc.github.io/ModelArray/>`_ to run statistical analyses on it.

# %%
# Convert result .h5 back to NIfTI
# --------------------------------
#
# After running **ModelArray** and obtaining statistical results inside ``FA.h5`` (suppose the
# analysis name is ``"mylm"``), use ``modelarrayio export-results`` to export them as NIfTI files.
#
# .. code-block:: console
#
#     modelarrayio export-results \
#         --mask            /home/username/myProject/data/group_mask.nii.gz \
#         --analysis-name   mylm \
#         --input-hdf5      /home/username/myProject/data/FA.h5 \
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
