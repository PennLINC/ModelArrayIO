"""Convert HDF5 file to MIF data."""

from __future__ import annotations

import argparse
import logging
import shutil
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_from_modelarray_args, add_log_level_arg
from modelarrayio.utils.mif import MifImage, mif_to_image

logger = logging.getLogger(__name__)


def h5_to_mif(example_mif, in_file, analysis_name, output_dir):
    """Writes the contents of an hdf5 file to a fixels directory.

    The ``in_file`` parameter should point to an HDF5 file that contains at least two
    datasets. There must be one called ``results/results_matrix``, that contains a
    matrix of fixel results. Each column contains a single result and each row is a
    fixel. This matrix should be of type float. The second required dataset must be
    named ``results/has_names``. This data can be of any type and does not need to contain
    more than a single row of data. Instead, its attributes are read to get column names
    for the data represented in ``results/results_matrix``.
    The function takes the example mif file as a template header. Then each column in
    ``results/results_matrix`` is extracted to fill the data of a new ``MifImage`` and
    named according to the corresponding item in ``results/has_names``.

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
    # Use the example MIF as the template so layout and metadata stay native to MIF.
    template_img, _ = mif_to_image(example_mif)
    template_shape = template_img.shape
    output_path = Path(output_dir)
    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )

        for result_col, result_name in enumerate(results_names):
            valid_result_name = cli_utils.sanitize_result_name(result_name)
            out_mif = output_path / f'{analysis_name}_{valid_result_name}.mif'
            if out_mif.exists():
                logger.warning('Output file already exists. Not overwriting. %s', out_mif)
                continue

            result_data = np.asarray(results_matrix[result_col, :], dtype=np.float32).reshape(
                template_shape
            )
            result_header = template_img.header.copy()
            result_header.set_data_shape(result_data.shape)
            result_header.set_data_dtype(result_data.dtype)
            result_img = MifImage(result_data, template_img.affine, header=result_header)
            result_img.to_filename(out_mif)

            if 'p.value' not in valid_result_name:
                continue

            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_mif_1mpvalue = output_path / f'{analysis_name}_{valid_result_name_1mpvalue}.mif'
            if out_mif_1mpvalue.exists():
                logger.warning('Output file already exists. Not overwriting. %s', out_mif_1mpvalue)
                continue

            output_mifvalues_1mpvalue = np.asarray(
                1 - results_matrix[result_col, :],
                dtype=np.float32,
            ).reshape(template_shape)
            output_header_1mpvalue = template_img.header.copy()
            output_header_1mpvalue.set_data_shape(output_mifvalues_1mpvalue.shape)
            output_header_1mpvalue.set_data_dtype(output_mifvalues_1mpvalue.dtype)
            output_img_1mpvalue = MifImage(
                output_mifvalues_1mpvalue,
                template_img.affine,
                header=output_header_1mpvalue,
            )
            output_img_1mpvalue.to_filename(out_mif_1mpvalue)


def h5_to_mif_main(
    index_file,
    directions_file,
    analysis_name,
    in_file,
    output_dir,
    cohort_file=None,
    example_mif=None,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio h5-to-mif`` command."""
    cli_utils.configure_logging(log_level)
    output_path = cli_utils.prepare_output_directory(output_dir, logger)

    shutil.copyfile(
        directions_file,
        output_path / Path(directions_file).name,
    )
    shutil.copyfile(
        index_file,
        output_path / Path(index_file).name,
    )

    if example_mif is None:
        logger.warning(
            'No example MIF file provided, using the first MIF file from the cohort file'
        )
        cohort_df = pd.read_csv(cohort_file)
        example_mif = cohort_df['source_file'].iloc[0]

    h5_to_mif(
        example_mif=example_mif,
        in_file=in_file,
        analysis_name=analysis_name,
        output_dir=output_path,
    )
    return 0


def _parse_h5_to_mif():
    parser = argparse.ArgumentParser(
        description='Create a fixel directory from an hdf5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    parser.add_argument(
        '--index-file',
        '--index_file',
        help='Index file used to reconstruct MIF files.',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--directions-file',
        '--directions_file',
        help='Directions file used to reconstruct MIF files.',
        required=True,
        type=IsFile,
    )

    add_from_modelarray_args(parser)

    example_mif_group = parser.add_mutually_exclusive_group(required=True)
    example_mif_group.add_argument(
        '--cohort-file',
        '--cohort_file',
        help=(
            'Path to a csv with demographic info and paths to data. '
            'Used to select an example MIF file if no example MIF file is provided.'
        ),
        type=IsFile,
        default=None,
    )
    example_mif_group.add_argument(
        '--example-mif',
        '--example_mif',
        help='Path to an example MIF file.',
        type=IsFile,
        default=None,
    )
    add_log_level_arg(parser)
    return parser
