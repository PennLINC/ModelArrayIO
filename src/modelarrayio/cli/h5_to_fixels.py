import argparse
import os
import shutil
from functools import partial

import h5py
import nibabel as nb
import pandas as pd

from modelarrayio.cli.parser_utils import _is_file
from modelarrayio.utils.fixels import mif_to_nifti2, nifti2_to_mif


def h5_to_mifs(example_mif, in_file, analysis_name, output_dir):
    """Writes the contents of an hdf5 file to a fixels directory.

    The ``in_file`` parameter should point to an HDF5 file that contains at least two
    datasets. There must be one called ``results/results_matrix``, that contains a
    matrix of fixel results. Each column contains a single result and each row is a
    fixel. This matrix should be of type float. The second required dataset must be
    named ``results/has_names``. This data can be of any type and does not need to contain
    more than a single row of data. Instead, its attributes are read to get column names
    for the data represented in ``results/results_matrix``.
    The function takes the example mif file and converts it to Nifti2 to get a header.
    Then each column in ``results/results_matrix`` is extracted to fill the data of a
    new Nifti2 file that gets converted to mif and named according to the corresponding
    item in ``results/has_names``.

    Parameters
    ==========
    example_mif: str
        abspath to a scalar mif file. Its header is used as a template
    in_file: str
        abspath to an h5 file that contains statistical results and their metadata.
    analysis_name: str
        the name for the analysis results to be saved
    output_dir: str
        abspath to where the output fixel data will go. the index and directions mif files
        should already be copied here.

    Outputs
    =======
    None
    """
    # Get a template nifti image.
    nifti2_img, _ = mif_to_nifti2(example_mif)
    h5_data = h5py.File(in_file, 'r')
    results_matrix = h5_data['results/' + analysis_name + '/results_matrix']
    names_data = results_matrix.attrs['colnames']  # NOTE: results_matrix: need to be transposed...
    # print(results_matrix.shape)

    # print(h5_data['results/' + analysis_name + '/results_matrix'].attrs['column_names'])

    try:
        results_names = names_data.tolist()
    except (AttributeError, OSError, TypeError, ValueError):
        print("Unable to read column names, using 'componentNNN' instead")
        results_names = [f'component{n + 1:03d}' for n in range(results_matrix.shape[0])]

    # Make output directory if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for result_col, result_name in enumerate(results_names):
        valid_result_name = result_name.replace(' ', '_').replace('/', '_')
        out_mif = os.path.join(output_dir, f'{analysis_name}_{valid_result_name}.mif')
        temp_nifti2 = nb.Nifti2Image(
            results_matrix[result_col, :].reshape(-1, 1, 1),
            nifti2_img.affine,
            header=nifti2_img.header,
        )
        nifti2_to_mif(temp_nifti2, out_mif)

        # if this result is p.value, also write out 1-p.value (1m.p.value)
        # the result name contains "p.value" (from R package broom)
        if 'p.value' in valid_result_name:
            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_mif_1mpvalue = os.path.join(
                output_dir, f'{analysis_name}_{valid_result_name_1mpvalue}.mif'
            )
            output_mifvalues_1mpvalue = 1 - results_matrix[result_col, :]  # 1 minus
            temp_nifti2_1mpvalue = nb.Nifti2Image(
                output_mifvalues_1mpvalue.reshape(-1, 1, 1),
                nifti2_img.affine,
                header=nifti2_img.header,
            )
            nifti2_to_mif(temp_nifti2_1mpvalue, out_mif_1mpvalue)


def main():
    parser = get_parser()
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        print('WARNING: Output directory exists')
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy in the index and directions
    shutil.copyfile(
        args.directions_file,
        os.path.join(args.output_dir, os.path.basename(args.directions_file)),
    )
    shutil.copyfile(
        args.index_file,
        os.path.join(args.output_dir, os.path.basename(args.index_file)),
    )

    # Get an example mif file
    cohort_df = pd.read_csv(args.cohort_file)
    example_mif = cohort_df['source_file'][0]
    h5_to_mifs(
        example_mif=example_mif,
        in_file=args.in_file,
        analysis_name=args.analysis_name,
        output_dir=args.output_dir,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description='Create a fixel directory from an hdf5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    parser.add_argument(
        '--index-file',
        '--index_file',
        help='Index File',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--directions-file',
        '--directions_file',
        help='Directions File',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--analysis-name',
        '--analysis_name',
        help='Name for the statistical analysis results to be saved.',
    )
    parser.add_argument(
        '--input-hdf5',
        '--input_hdf5',
        help='Name of HDF5 (.h5) file where results outputs are saved.',
        type=IsFile,
        dest='in_file',
    )
    parser.add_argument(
        '--output-dir',
        '--output_dir',
        help=(
            'Fixel directory where outputs will be saved. '
            'If the directory does not exist, it will be automatically created.'
        ),
    )
    return parser
