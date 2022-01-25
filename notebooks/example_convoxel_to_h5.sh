#!/bin/bash
# TO SETUP:
# git clone <repo>
# cd ConFixel
# pip install -e .

# for grmpy project, on cubic:
nsubj=3
cmd="convoxel --group-mask-file aslprep_unzipped_convoxel/code/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz --cohort-file aslprep_unzipped_convoxel/GRMPY_convoxel_meanCBF_n${nsubj}.csv --relative-root /cbica/projects/GRMPY/project/curation/testing/ --output-hdf5 aslprep_unzipped_convoxel/voxeldb_meanCBF_n${nsubj}.h5"
echo $cmd
$cmd