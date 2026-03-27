"""Export one scalar matrix row from HDF5 to a CIFTI dscalar file."""

from __future__ import annotations

import argparse
import logging
from functools import partial

import nibabel as nb
import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    _is_file,
    add_hdf5_scalar_export_args,
    add_log_level_arg,
)

logger = logging.getLogger(__name__)


def h5_export_cifti_file(
    in_file,
    scalar_name,
    output_file,
    column_index=None,
    source_file=None,
    cohort_file=None,
    example_cifti=None,
):
    row = cli_utils.load_hdf5_scalar_row(
        in_file,
        scalar_name,
        column_index=column_index,
        source_file=source_file,
    )

    if example_cifti is None:
        cohort_df = pd.read_csv(cohort_file)
        example_cifti = cohort_df['source_file'].iloc[0]
    cifti = nb.load(example_cifti)
    if row.shape[0] != cifti.shape[-1]:
        raise ValueError(
            f'Scalar row length ({row.shape[0]}) does not match CIFTI greyordinates '
            f'({cifti.shape[-1]}).'
        )

    out_path = cli_utils.prepare_output_parent(output_file)
    nb.Cifti2Image(
        row.reshape(1, -1),
        header=cifti.header,
        nifti_header=cifti.nifti_header,
    ).to_filename(out_path)


def h5_export_cifti_file_main(**kwargs):
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    h5_export_cifti_file(**kwargs)
    return 0


def _parse_h5_export_cifti_file():
    parser = argparse.ArgumentParser(
        description='Export one row from scalars/<name>/values to a CIFTI file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    is_file = partial(_is_file, parser=parser)
    add_hdf5_scalar_export_args(parser)

    example_group = parser.add_mutually_exclusive_group(required=True)
    example_group.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to cohort CSV used to choose an example CIFTI file.',
        type=is_file,
        default=None,
    )
    example_group.add_argument(
        '--example-cifti',
        '--example_cifti',
        help='Path to an example CIFTI file used as output template.',
        type=is_file,
        default=None,
    )
    add_log_level_arg(parser)
    return parser
