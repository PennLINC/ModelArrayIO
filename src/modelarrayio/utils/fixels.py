"""Utility functions for fixel-wise data."""

import os
import os.path as op
import shutil
import subprocess
import tempfile

import nibabel as nb
import numpy as np
import pandas as pd


def find_mrconvert():
    program = 'mrconvert'

    def is_exe(fpath):
        return op.exists(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ['PATH'].split(os.pathsep):
        path = path.strip('"')
        exe_file = op.join(path, program)
        if is_exe(exe_file):
            return program
    return None


def nifti2_to_mif(nifti2_image, mif_file):
    """Convert a .nii file to a .mif file.

    Parameters
    ----------
    nifti2_image : :obj:`nibabel.Nifti2Image`
        Nifti2 image
    mif_file : :obj:`str`
        Path to a .mif file
    """
    # Note: because -force is not turned on in "mrconvert", the output files won't be overwritten!
    mrconvert = find_mrconvert()
    if mrconvert is None:
        raise Exception('The mrconvert executable could not be found on $PATH')

    nii_file = mif_file.replace('.mif', '.nii')
    nifti2_image.to_filename(nii_file)  # save as .nii first

    # convert .nii to .mif
    proc = subprocess.Popen(
        [mrconvert, nii_file, mif_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, err = proc.communicate()

    if not op.exists(mif_file):
        raise Exception(err)

    os.remove(nii_file)  # remove temporary .nii file


def mif_to_nifti2(mif_file):
    """Convert a .mif file to a .nii file.

    Parameters
    ----------
    mif_file : :obj:`str`
        Path to a .mif file

    Returns
    -------
    nifti2_img : :obj:`nibabel.Nifti2Image`
        Nifti2 image
    data : :obj:`numpy.ndarray`
        Data from the nifti2 image
    """
    if not mif_file.endswith('.nii'):
        dirpath = tempfile.mkdtemp()
        mrconvert = find_mrconvert()
        if mrconvert is None:
            raise Exception('The mrconvert executable could not be found on $PATH')
        nii_file = op.join(dirpath, 'mif.nii')
        proc = subprocess.Popen(
            [mrconvert, mif_file, nii_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, err = proc.communicate()
    else:
        nii_file = mif_file
        dirpath = None

    if not op.exists(nii_file):
        raise Exception(err)

    nifti2_img = nb.load(nii_file)
    data = nifti2_img.get_fdata(dtype=np.float32).squeeze()
    # ... do stuff with dirpath
    if dirpath:
        shutil.rmtree(dirpath)
    return nifti2_img, data


def gather_fixels(index_file, directions_file):
    """Load the index and directions files to get lookup tables.

    Parameters
    ----------
    index_file : :obj:`str`
        Path to a Nifti2 index file
    directions_file : :obj:`str`
        Path to a Nifti2 directions file

    Returns
    -------
    fixel_table : :obj:`pandas.DataFrame`
        DataFrame with fixel_id, voxel_id, x, y, z
    voxel_table : :obj:`pandas.DataFrame`
        DataFrame with voxel_id, i, j, k
    """
    _index_img, index_data = mif_to_nifti2(index_file)
    count_vol = index_data[..., 0].astype(
        np.uint32
    )  # number of fixels in each voxel; by index.mif definition
    id_vol = index_data[
        ..., 1
    ]  # index of the first fixel in this voxel, in the list of all fixels (in directions.mif, FD.mif, etc)
    max_id = id_vol.max()
    max_fixel_id = max_id + int(
        count_vol[id_vol == max_id]
    )  # = the maximum id of fixels + 1 = # of fixels in entire image
    voxel_mask = count_vol > 0  # voxels that contains fixel(s), =1
    masked_ids = id_vol[voxel_mask]  # 1D array, len = # of voxels with fixel(s), value see id_vol
    masked_counts = count_vol[voxel_mask]  # dim as masked_ids; value see count_vol
    id_sort = np.argsort(
        masked_ids
    )  #  indices that would sort array masked_ids value (i.e. first fixel's id in this voxel) from lowest to highest; so it's sorting voxels by their first fixel id
    sorted_counts = masked_counts[id_sort]
    voxel_coords = np.column_stack(
        np.nonzero(count_vol)
    )  # dim: [# of voxels with fixel(s)] x 3, each row is the subscript i.e. (i,j,k) in 3D image of a voxel with fixel

    fixel_id = 0
    fixel_ids = np.arange(max_fixel_id, dtype=np.int32)
    fixel_voxel_ids = np.zeros_like(fixel_ids)
    for voxel_id, fixel_count in enumerate(sorted_counts):
        for _ in range(fixel_count):
            fixel_voxel_ids[fixel_id] = (
                voxel_id  # fixel_voxel_ids: 1D, len = # of fixels; each value is the voxel_id of the voxel where this fixel locates
            )
            fixel_id += 1
    sorted_coords = voxel_coords[id_sort]

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': sorted_coords[:, 0],
            'j': sorted_coords[:, 1],
            'k': sorted_coords[:, 2],
        }
    )

    _directions_img, directions_data = mif_to_nifti2(directions_file)
    fixel_table = pd.DataFrame(
        {
            'fixel_id': fixel_ids,
            'voxel_id': fixel_voxel_ids,
            'x': directions_data[:, 0],
            'y': directions_data[:, 1],
            'z': directions_data[:, 2],
        }
    )

    return fixel_table, voxel_table
