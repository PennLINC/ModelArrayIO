"""Convert HDF5 file to NIfTI data."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_from_modelarray_args, add_log_level_arg

logger = logging.getLogger(__name__)


def h5_to_nifti(in_file, analysis_name, group_mask_file, output_extension, output_dir):
    """Convert stat results in .h5 file to a list of volume (.nii or .nii.gz) files."""

    data_type_tosave = np.float32

    # group-level mask:
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0

    # modify the header:
    header_tosave = group_mask_img.header
    # modify the data type (mask's data type could be uint8...)
    header_tosave.set_data_dtype(data_type_tosave)

    output_path = Path(output_dir)
    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )

        for result_col, result_name in enumerate(results_names):
            valid_result_name = cli_utils.sanitize_result_name(result_name)
            out_file = output_path / f'{analysis_name}_{valid_result_name}{output_extension}'
            output = np.zeros(group_mask_matrix.shape)
            data_tosave = results_matrix[result_col, :].astype(data_type_tosave)
            output[group_mask_matrix] = data_tosave
            output_img = nb.Nifti1Image(output, affine=group_mask_img.affine, header=header_tosave)
            output_img.to_filename(out_file)

            if 'p.value' not in valid_result_name:
                continue

            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_file_1mpvalue = (
                output_path / f'{analysis_name}_{valid_result_name_1mpvalue}{output_extension}'
            )
            output_1mpvalue = np.zeros(group_mask_matrix.shape)
            output_1mpvalue[group_mask_matrix] = (1 - results_matrix[result_col, :]).astype(
                data_type_tosave
            )
            output_img_1mpvalue = nb.Nifti1Image(
                output_1mpvalue, affine=group_mask_img.affine, header=header_tosave
            )
            output_img_1mpvalue.to_filename(out_file_1mpvalue)


def h5_to_nifti_main(
    group_mask_file,
    analysis_name,
    in_file,
    output_dir,
    output_extension='.nii.gz',
    log_level='INFO',
):
    """Entry point for the ``modelarrayio h5-to-nifti`` command."""
    cli_utils.configure_logging(log_level)
    output_path = cli_utils.prepare_output_directory(output_dir, logger)

    h5_to_nifti(
        in_file=in_file,
        analysis_name=analysis_name,
        group_mask_file=group_mask_file,
        output_extension=output_extension,
        output_dir=output_path,
    )
    return 0


def _parse_h5_to_nifti():
    parser = argparse.ArgumentParser(
        description='Convert statistical results from an hdf5 file to a volume data (NIfTI file)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)
    parser.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        help='Path to a group mask file',
        required=True,
        type=IsFile,
    )

    add_from_modelarray_args(parser)

    parser.add_argument(
        '--output-ext',
        '--output_ext',
        dest='output_extension',
        help=(
            'The extension for output volume data. '
            'Options are .nii.gz (default) and .nii. Please provide the prefix dot.'
        ),
        choices=['.nii.gz', '.nii'],
        default='.nii.gz',
    )
    add_log_level_arg(parser)
    return parser
