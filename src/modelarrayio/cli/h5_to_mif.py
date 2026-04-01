"""Convert HDF5 file to MIF data."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import nibabel as nb

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.mif import mif_to_nifti2, nifti2_to_mif

logger = logging.getLogger(__name__)


def h5_to_mif(example_mif, in_file, analysis_name, compress, output_dir):
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
    ----------
    example_mif: str
        abspath to a scalar mif file. Its header is used as a template
    in_file: str
        abspath to an h5 file that contains statistical results and their metadata.
    analysis_name: str
        the name for the analysis results to be saved
    compress: bool
        whether to compress output MIF files
    output_dir: str
        abspath to where the output fixel data will go. the index and directions mif files
        should already be copied here.

    Outputs
    -------
    None
    """
    # Get a template nifti image.
    nifti2_img, _ = mif_to_nifti2(example_mif)
    output_path = Path(output_dir)
    ext = '.mif.gz' if compress else '.mif'
    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )

        for result_col, result_name in enumerate(results_names):
            valid_result_name = cli_utils.sanitize_result_name(result_name)
            out_mif = output_path / f'{analysis_name}_{valid_result_name}{ext}'
            temp_nifti2 = nb.Nifti2Image(
                results_matrix[result_col, :].reshape(-1, 1, 1),
                nifti2_img.affine,
                header=nifti2_img.header,
            )
            nifti2_to_mif(temp_nifti2, out_mif)

            if 'p.value' not in valid_result_name:
                continue

            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_mif_1mpvalue = output_path / f'{analysis_name}_{valid_result_name_1mpvalue}{ext}'
            output_mifvalues_1mpvalue = 1 - results_matrix[result_col, :]
            temp_nifti2_1mpvalue = nb.Nifti2Image(
                output_mifvalues_1mpvalue.reshape(-1, 1, 1),
                nifti2_img.affine,
                header=nifti2_img.header,
            )
            nifti2_to_mif(temp_nifti2_1mpvalue, out_mif_1mpvalue)
