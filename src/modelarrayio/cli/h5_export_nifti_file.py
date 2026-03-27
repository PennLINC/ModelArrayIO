"""Export one scalar matrix row from HDF5 to a NIfTI file."""

from __future__ import annotations

import argparse
import logging
from functools import partial

import nibabel as nb
import numpy as np

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    _is_file,
    add_hdf5_scalar_export_args,
    add_log_level_arg,
)

logger = logging.getLogger(__name__)


def h5_export_nifti_file(
    in_file,
    scalar_name,
    output_file,
    group_mask_file,
    column_index=None,
    source_file=None,
):
    row = cli_utils.load_hdf5_scalar_row(
        in_file,
        scalar_name,
        column_index=column_index,
        source_file=source_file,
    )
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
    num_voxels = int(group_mask_matrix.sum())
    if row.shape[0] != num_voxels:
        raise ValueError(
            f'Scalar row length ({row.shape[0]}) does not match group mask voxels ({num_voxels}).'
        )

    output = np.zeros(group_mask_matrix.shape, dtype=np.float32)
    output[group_mask_matrix] = row.astype(np.float32)
    header = group_mask_img.header.copy()
    header.set_data_dtype(np.float32)
    out_path = cli_utils.prepare_output_parent(output_file)
    nb.Nifti1Image(output, affine=group_mask_img.affine, header=header).to_filename(out_path)


def h5_export_nifti_file_main(**kwargs):
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    h5_export_nifti_file(**kwargs)
    return 0


def _parse_h5_export_nifti_file():
    parser = argparse.ArgumentParser(
        description='Export one row from scalars/<name>/values to a NIfTI file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    is_file = partial(_is_file, parser=parser)
    add_hdf5_scalar_export_args(parser)
    parser.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        required=True,
        type=is_file,
        help='Path to the group mask file used for original voxel flattening.',
    )
    add_log_level_arg(parser)
    return parser
