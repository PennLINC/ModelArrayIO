"""Export one scalar matrix row from HDF5 to a MIF file."""

from __future__ import annotations

import argparse
import logging
from functools import partial

import nibabel as nb

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    _is_file,
    add_hdf5_scalar_export_args,
    add_log_level_arg,
)
from modelarrayio.utils.fixels import mif_to_nifti2, nifti2_to_mif

logger = logging.getLogger(__name__)


def h5_export_mif_file(
    in_file,
    scalar_name,
    output_file,
    example_mif,
    column_index=None,
    source_file=None,
):
    row = cli_utils.load_hdf5_scalar_row(
        in_file,
        scalar_name,
        column_index=column_index,
        source_file=source_file,
    )
    template_nifti2, _ = mif_to_nifti2(example_mif)
    out_nifti2 = nb.Nifti2Image(
        row.reshape(-1, 1, 1),
        affine=template_nifti2.affine,
        header=template_nifti2.header,
    )
    out_path = cli_utils.prepare_output_parent(output_file)
    nifti2_to_mif(out_nifti2, out_path)


def h5_export_mif_file_main(**kwargs):
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    h5_export_mif_file(**kwargs)
    return 0


def _parse_h5_export_mif_file():
    parser = argparse.ArgumentParser(
        description='Export one row from scalars/<name>/values to a MIF file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    is_file = partial(_is_file, parser=parser)
    add_hdf5_scalar_export_args(parser)
    parser.add_argument(
        '--example-mif',
        '--example_mif',
        required=True,
        type=is_file,
        help='Path to an example MIF file used as output template.',
    )
    add_log_level_arg(parser)
    return parser
