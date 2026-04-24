"""Convert HDF5 file to MIF data."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.mif import MifImage, mif_to_image

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
    The function takes the example mif file as a template header. Then each column in
    ``results/results_matrix`` is extracted to fill the data of a new ``MifImage`` and
    named according to the corresponding item in ``results/has_names``.

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
    # Use the example MIF as the template so layout and metadata stay native to MIF.
    template_img, _ = mif_to_image(example_mif)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ext = '.mif.gz' if compress else '.mif'
    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )

        for result_col, result_name in enumerate(results_names):
            valid_result_name = cli_utils.sanitize_result_name(result_name)
            out_mif = output_path / f'{analysis_name}_{valid_result_name}{ext}'
            write_mif(
                arr=np.asarray(results_matrix[result_col, :], dtype=np.float32),
                template_img=template_img,
                out_file=out_mif,
            )

            if 'p.value' not in valid_result_name:
                continue

            valid_result_name_1mpvalue = valid_result_name.replace('p.value', '1m.p.value')
            out_mif_1mpvalue = output_path / f'{analysis_name}_{valid_result_name_1mpvalue}{ext}'
            output_mifvalues_1mpvalue = np.asarray(
                1 - results_matrix[result_col, :],
                dtype=np.float32,
            )
            write_mif(
                arr=output_mifvalues_1mpvalue,
                template_img=template_img,
                out_file=out_mif_1mpvalue,
            )

    return 0


def write_mif(arr, template_img, out_file):
    """Write array to MIF file.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray`
        Array to write to file.
    template_img : :obj:`MifImage`
        Template MIF image.
    out_file : :obj:`pathlib.Path`
        Output file to write.
        If it already exists, this function will raise a warning and not overwrite.
    """
    if out_file.exists():
        logger.warning('Output file already exists. Not overwriting. %s', out_file)
        return

    result_data = arr.reshape(template_img.shape)
    result_header = template_img.header.copy()
    result_header.set_data_shape(result_data.shape)
    result_header.set_data_dtype(result_data.dtype)
    result_img = MifImage(result_data, template_img.affine, header=result_header)
    result_img.to_filename(out_file)
