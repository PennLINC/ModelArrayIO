"""Convert ModelArray HDF5 results back to an ODX file.

The ODX analogue of :mod:`modelarrayio.cli.h5_to_mif`. Reads the
``results/<analysis_name>/results_matrix`` written by ModelArray and paints each
result metric (e.g. ``<term>.estimate``, ``<term>.p.value.fdr``) onto a template
ODX's group-fixel geometry as a per-fixel (DPF) array, producing a single ODX
that can be visualized (e.g. in trxviz). For each ``p.value`` metric a
``1m.p.value`` (1 - p) companion is also written, matching ``h5_to_mif``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.odx import write_odx_results

logger = logging.getLogger(__name__)


def h5_to_odx(example_odx, in_file, analysis_name, compress, output_dir):
    """Write ModelArray results from an HDF5 file onto a template ODX.

    Parameters
    ----------
    example_odx : path-like
        Template ODX whose group-fixel geometry the results are painted onto
        (any of the per-subject ODX used to build the HDF5, or the group ODX).
    in_file : path-like
        HDF5 file containing ``results/<analysis_name>/results_matrix``.
    analysis_name : str
        Name of the ModelArray results group inside the HDF5 file.
    compress : bool
        Accepted for signature parity with the other ``h5_to_*`` writers; the
        ``.odx`` archive is already compressed.
    output_dir : path-like
        Directory where ``<analysis_name>.odx`` is written.

    Returns
    -------
    status : :obj:`int`
        0 if successful.
    """
    del compress  # ODX archives are self-compressed; kept for API parity.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_file, 'r') as h5_data:
        results_matrix = h5_data[f'results/{analysis_name}/results_matrix']
        results_names = cli_utils.read_result_names(
            h5_data, analysis_name, results_matrix, logger=logger
        )
        results: dict[str, np.ndarray] = {}
        for result_col, result_name in enumerate(results_names):
            safe = cli_utils.sanitize_result_name(result_name)
            column = np.asarray(results_matrix[result_col, :], dtype=np.float32)
            results[f'{analysis_name}_{safe}'] = column
            if 'p.value' in safe:
                results[f'{analysis_name}_{safe.replace("p.value", "1m.p.value")}'] = (
                    1.0 - column
                ).astype(np.float32)

    out_odx = output_path / f'{analysis_name}.odx'
    write_odx_results(example_odx, results, out_odx)
    logger.info('Wrote %d result DPF arrays to %s', len(results), out_odx)
    return 0
