#!/usr/bin/env python
import nibabel as nb
import numpy as np



def flattened_image(scalar_image, scalar_mask, group_mask_matrix):
    scalar_mask_img = nb.load(scalar_mask)
    scalar_mask_matrix = scalar_mask_img.get_fdata() > 0
    
    scalar_img = nb.load(scalar_image)
    scalar_matrix = scalar_img.get_fdata()

    scalar_matrix[np.logical_not(scalar_mask_matrix)] = np.nan
    return scalar_matrix[group_mask_matrix]
    

def get_group_matrix(cohort_df, group_mask_file):
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0

    for xx in cohort_df:
        pass


def back_to_3d(group_mask_file, results_array, out_file):

    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0

    output = np.zeros(group_mask_matrix.shape)
    output[group_mask_matrix] = results_array
    output_img = nb.Nifti1Image(output, affine=group_mask_img.affine,
                                header=group_mask_img.header)
    output_img.to_filename(out_file)


