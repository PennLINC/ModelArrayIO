"""
CIFTI (Greyordinate-wise) Data Conversion
=========================================

For imaging data in CIFTI format, use the ``modelarrayio cifti-to-h5`` command to convert
the CIFTI files to the HDF5 format (``.h5``) used by **ModelArray**,
and ``modelarrayio h5-to-cifti`` to export results back to CIFTI.
The CIFTI workflow is very similar to the MIF workflow
(:ref:`sphx_glr_auto_examples_plot_mif_workflow.py`).
"""

# %%
# Prepare data
# ------------
#
# To convert a list of CIFTI files to ``.h5`` format, you need:
#
# 1. **A cohort CSV** describing every CIFTI file to include (one CSV per scalar recommended).
#
# Cohort CSV columns (names are fixed, not user-defined):
#
# * ``scalar_name`` — which metric is being analysed (e.g., ``FA``)
# * ``source_file`` — path to the subject's CIFTI file

# %%
# Example folder structure
# ------------------------
#
# .. code-block:: text
#
#     /home/username/myProject/data
#     |
#     ├── cohort_FA.csv
#     │
#     ├── FA
#     │   ├── sub-01_FA.dscalar.nii
#     │   ├── sub-02_FA.dscalar.nii
#     │   ├── sub-03_FA.dscalar.nii
#     │   └── ...
#     │
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
#     modelarrayio cifti-to-h5 \
#         --cohort-file     /home/username/myProject/data/cohort_FA.csv \
#         --output          /home/username/myProject/data/FA.h5
#
# This produces ``FA.h5`` in ``/home/username/myProject/data``.  You can then use
# `ModelArray <https://pennlinc.github.io/ModelArray/>`_ to run statistical analyses on it.

# %%
# Convert result .h5 back to CIFTI
# --------------------------------
#
# After running **ModelArray** and obtaining statistical results inside ``FA.h5`` (suppose the
# analysis name is ``"mylm"``), use ``modelarrayio h5-to-cifti`` to export them as CIFTI files.
#
# You must also provide an example CIFTI file to use as a template for the output.
#
# .. code-block:: console
#
#     modelarrayio h5-to-cifti \
#         --cohort-file     /home/username/myProject/data/cohort_FA.csv \
#         --analysis-name   mylm \
#         --input-hdf5      /home/username/myProject/data/FA.h5 \
#         --output-dir      /home/username/myProject/data/FA_stats \
#         --example-cifti   /home/username/myProject/data/FA/sub-01_FA.dscalar.nii
#
# All converted volume data are saved as ``float32``.  Results in ``FA_stats`` can be viewed
# with any CIFTI image viewer.
#
# .. warning::
#
#    If ``--output-dir`` already exists, ``modelarrayio h5-to-cifti`` will not delete it — you will
#    see ``WARNING: Output directory exists``.  Existing files that are **not** part of the
#    current output list are left unchanged.  Existing files that **are** part of the current
#    output list will be overwritten.  To avoid confusion, consider manually deleting the output
#    directory before re-running ``modelarrayio h5-to-cifti``.

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
#     modelarrayio cifti-to-h5 --help
#     modelarrayio h5-to-cifti --help
#
# or in the :doc:`/usage` page of this documentation.
