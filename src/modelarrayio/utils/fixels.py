"""Utility functions for fixel-wise data."""

import os
import os.path as op
import shutil
import subprocess
import tempfile

import nibabel as nb
import numpy as np


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
