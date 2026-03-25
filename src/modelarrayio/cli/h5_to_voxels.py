import argparse
import logging
import os
import os.path as op

import h5py
import nibabel as nb
import numpy as np

from modelarrayio.cli.parser_utils import add_relative_root_arg

logger = logging.getLogger(__name__)


def h5_to_volumes(h5_file, analysis_name, group_mask_file, output_extension, volume_output_dir):
    """Convert stat results in .h5 file to a list of volume (.nii or .nii.gz) files."""

    data_type_tosave = np.float32

    # group-level mask:
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0

    # modify the header:
    header_tosave = group_mask_img.header
    header_tosave.set_data_dtype(
        data_type_tosave
    )  # modify the data type (mask's data type could be uint8...)

    # results in .h5 file:
    h5_data = h5py.File(h5_file, 'r')
    results_matrix = h5_data['results/' + analysis_name + '/results_matrix']

    # NOTE: results_matrix may need to be transposed depending on writer conventions
    # Attempt to read column names: prefer attribute; fallback to dataset-based names
    def _decode_names(arr):
        try:
            if isinstance(arr, (list, tuple)):
                seq = arr
            elif isinstance(arr, np.ndarray):
                seq = arr.tolist()
            else:
                seq = [arr]
            out = []
            for x in seq:
                if isinstance(x, (bytes, bytearray, np.bytes_)):
                    s = x.decode('utf-8', errors='ignore')
                else:
                    s = str(x)
                s = s.rstrip('\x00').strip()
                out.append(s)
            return out
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    results_names = None
    # 1) Try attribute (backward compatibility)
    try:
        names_attr = results_matrix.attrs.get('colnames', None)
        if names_attr is not None:
            results_names = _decode_names(names_attr)
    except (OSError, RuntimeError, TypeError, ValueError):
        results_names = None

    # 2) Fallback to dataset-based column names (new format)
    if not results_names:
        candidate_paths = [
            f'results/{analysis_name}/column_names',
            f'results/{analysis_name}/results_matrix/column_names',
        ]
        for p in candidate_paths:
            if p in h5_data:
                try:
                    names_ds = h5_data[p][()]
                    results_names = _decode_names(names_ds)
                    if results_names:
                        break
                except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug('Could not read column names from %s', p, exc_info=True)
                    continue

    # 3) Final fallback to generated names
    if not results_names:
        print("Unable to read column names, using 'componentNNN' instead")
        results_names = [f'component{n + 1:03d}' for n in range(results_matrix.shape[0])]

    # # Make output directory if it does not exist  # has been done in h5_to_volumes_wrapper()
    # if op.isdir(volume_output_dir) == False:
    #     os.mkdir(volume_output_dir)

    # for loop: save stat metric results one by one:
    for result_col, result_name in enumerate(results_names):
        valid_result_name = result_name.replace(' ', '_').replace('/', '_')

        out_file = op.join(
            volume_output_dir, analysis_name + '_' + valid_result_name + output_extension
        )
        output = np.zeros(group_mask_matrix.shape)
        data_tosave = results_matrix[result_col, :]
        data_tosave = data_tosave.astype(
            data_type_tosave
        )  # make sure each result image's data type is the correct one
        output[group_mask_matrix] = data_tosave
        output_img = nb.Nifti1Image(output, affine=group_mask_img.affine, header=header_tosave)
        output_img.to_filename(out_file)

        # if this result is p.value, also write out 1-p.value (1m.p.value)
        # the result name contains "p.value" (from R package broom)
        if 'p.value' in valid_result_name:
            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_file_1mpvalue = op.join(
                volume_output_dir,
                analysis_name + '_' + valid_result_name_1mpvalue + output_extension,
            )
            output_1mpvalue = np.zeros(group_mask_matrix.shape)
            data_tosave = 1 - results_matrix[result_col, :]  # 1 minus
            data_tosave = data_tosave.astype(
                data_type_tosave
            )  # make sure each result image's data type is the correct one
            output_1mpvalue[group_mask_matrix] = data_tosave
            output_img_1mpvalue = nb.Nifti1Image(
                output_1mpvalue, affine=group_mask_img.affine, header=header_tosave
            )
            output_img_1mpvalue.to_filename(out_file_1mpvalue)


def h5_to_volumes_wrapper():
    parser = get_h5_to_volume_parser()
    args = parser.parse_args()

    volume_output_dir = op.join(
        args.relative_root, args.output_dir
    )  # absolute path for output dir

    if op.exists(volume_output_dir):
        print('WARNING: Output directory exists')
    os.makedirs(volume_output_dir, exist_ok=True)

    # any files to copy?

    # other arguments:
    group_mask_file = op.join(args.relative_root, args.group_mask_file)
    h5_input = op.join(args.relative_root, args.input_hdf5)
    analysis_name = args.analysis_name
    output_extension = args.output_ext

    # call function:
    h5_to_volumes(h5_input, analysis_name, group_mask_file, output_extension, volume_output_dir)


def get_h5_to_volume_parser():
    parser = argparse.ArgumentParser(
        description='Convert statistical results from an hdf5 file to a volume data (NIfTI file)'
    )
    parser.add_argument(
        '--group-mask-file', '--group_mask_file', help='Path to a group mask file', required=True
    )
    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
    )
    add_relative_root_arg(parser)
    parser.add_argument(
        '--analysis-name',
        '--analysis_name',
        help='Name of the statistical analysis results to be saved.',
    )
    parser.add_argument(
        '--input-hdf5',
        '--input_hdf5',
        help='Name of HDF5 (.h5) file where results outputs are saved.',
    )
    parser.add_argument(
        '--output-dir',
        '--output_dir',
        help=(
            'A directory where output volume files will be saved. '
            'If the directory does not exist, it will be automatically created.'
        ),
    )
    parser.add_argument(
        '--output-ext',
        '--output_ext',
        help=(
            'The extension for output volume data. '
            'Options are .nii.gz (default) and .nii. Please provide the prefix dot.'
        ),
        default='.nii.gz',
    )
    return parser
