"""Convert HDF5 file to CIFTI2 dscalar data."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import h5py
import nibabel as nb
import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_from_modelarray_args, add_log_level_arg

logger = logging.getLogger(__name__)


def _cifti_output_ext(cifti_img):
    """Return the output filename extension for a CIFTI image.

    Parameters
    ----------
    cifti_img : :obj:`nibabel.Cifti2Image`
        Loaded CIFTI image whose axes determine the file type.

    Returns
    -------
    ext : :obj:`str`
        One of ``'.dscalar.nii'``, ``'.pscalar.nii'``, or ``'.pconn.nii'``.

    Notes
    -----
    This function currently supports only a limited set of axis-type
    combinations:

    * ``ScalarAxis`` + ``BrainModelAxis``  -> ``.dscalar.nii``
    * ``ScalarAxis`` + ``ParcelsAxis``    -> ``.pscalar.nii``
    * ``ParcelsAxis`` + ``ParcelsAxis``   -> ``.pconn.nii``

    Any other axis configuration (e.g., ``SeriesAxis + BrainModelAxis``
    for ``dtseries``) will raise a :class:`ValueError`.
    """
    axes = [cifti_img.header.get_axis(i) for i in range(cifti_img.ndim)]

    # Expect 2D CIFTI images for these output types.
    if len(axes) != 2:
        raise ValueError(
            f'Unsupported CIFTI dimensionality {len(axes)}; '
            'only 2D CIFTI images are supported for dscalar/pscalar/pconn outputs.'
        )

    scalar_axis_cls = nb.cifti2.cifti2_axes.ScalarAxis
    parcels_axis_cls = nb.cifti2.cifti2_axes.ParcelsAxis
    brainmodel_axis_cls = nb.cifti2.cifti2_axes.BrainModelAxis

    first_axis, second_axis = axes

    # Scalar + BrainModel -> dscalar
    if isinstance(first_axis, scalar_axis_cls) and isinstance(second_axis, brainmodel_axis_cls):
        return '.dscalar.nii'

    # Scalar + Parcels -> pscalar
    if isinstance(first_axis, scalar_axis_cls) and isinstance(second_axis, parcels_axis_cls):
        return '.pscalar.nii'

    # Parcels + Parcels -> pconn
    if isinstance(first_axis, parcels_axis_cls) and isinstance(second_axis, parcels_axis_cls):
        return '.pconn.nii'

    axis_types = (type(first_axis).__name__, type(second_axis).__name__)
    raise ValueError(
        'Unsupported CIFTI axis combination '
        f'{axis_types}; cannot determine appropriate output extension.'
    )


def h5_to_cifti(example_cifti, in_file, analysis_name, output_dir):
    """Write the contents of an hdf5 file to a fixels directory.

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
    example_cifti: pathlike
        abspath to a scalar cifti file. Its header is used as a template
    in_file: str
        abspath to an h5 file that contains statistical results and their metadata.
    analysis_name: str
        the name for the analysis results to be saved
    fixel_output_dir: str
        abspath to where the output cifti files will go.

    Outputs
    =======
    None
    """
    # Get a template nifti image.
    cifti = nb.load(example_cifti)
    output_path = Path(output_dir)

    output_ext = _cifti_output_ext(cifti)
    is_pconn = output_ext == '.pconn.nii'
    if is_pconn:
        n_rows, n_cols = cifti.shape

    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )

        for result_col, result_name in enumerate(results_names):
            valid_result_name = cli_utils.sanitize_result_name(result_name)
            out_cifti = output_path / f'{analysis_name}_{valid_result_name}{output_ext}'
            row_data = results_matrix[result_col, :]
            data_array = row_data.reshape(n_rows, n_cols) if is_pconn else row_data.reshape(1, -1)
            temp_cifti2 = nb.Cifti2Image(
                data_array,
                header=cifti.header,
                nifti_header=cifti.nifti_header,
            )
            temp_cifti2.to_filename(out_cifti)

            if 'p.value' not in valid_result_name:
                continue

            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_cifti_1mpvalue = (
                output_path / f'{analysis_name}_{valid_result_name_1mpvalue}{output_ext}'
            )
            output_mifvalues_1mpvalue = 1 - row_data
            data_array_1mpvalue = (
                output_mifvalues_1mpvalue.reshape(n_rows, n_cols)
                if is_pconn
                else output_mifvalues_1mpvalue.reshape(1, -1)
            )
            temp_nifti2_1mpvalue = nb.Cifti2Image(
                data_array_1mpvalue,
                header=cifti.header,
                nifti_header=cifti.nifti_header,
            )
            temp_nifti2_1mpvalue.to_filename(out_cifti_1mpvalue)


def h5_to_cifti_main(
    analysis_name,
    in_file,
    output_dir,
    cohort_file=None,
    example_cifti=None,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio h5-to-cifti`` command."""
    cli_utils.configure_logging(log_level)
    output_path = cli_utils.prepare_output_directory(output_dir, logger)

    if example_cifti is None:
        logger.warning(
            'No example cifti file provided, using the first cifti file from the cohort file'
        )
        cohort_df = pd.read_csv(cohort_file)
        example_cifti = cohort_df['source_file'].iloc[0]

    h5_to_cifti(
        example_cifti=example_cifti,
        in_file=in_file,
        analysis_name=analysis_name,
        output_dir=output_path,
    )
    return 0


def _parse_h5_to_cifti():
    parser = argparse.ArgumentParser(
        description='Create a directory with cifti results from an hdf5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    add_from_modelarray_args(parser)

    example_cifti_group = parser.add_mutually_exclusive_group(required=True)
    example_cifti_group.add_argument(
        '--cohort-file',
        '--cohort_file',
        help=(
            'Path to a csv with demographic info and paths to data. '
            'Used to select an example CIFTI file if no example CIFTI file is provided.'
        ),
        type=IsFile,
        default=None,
    )
    example_cifti_group.add_argument(
        '--example-cifti',
        '--example_cifti',
        help='Path to an example cifti file.',
        type=IsFile,
        default=None,
    )

    add_log_level_arg(parser)
    return parser
