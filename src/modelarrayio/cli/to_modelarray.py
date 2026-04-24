"""Convert neuroimaging data to an HDF5 modelarray file."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.cifti_to_h5 import cifti_to_h5
from modelarrayio.cli.mif_to_h5 import mif_to_h5
from modelarrayio.cli.nifti_to_h5 import nifti_to_h5
from modelarrayio.cli.parser_utils import _is_file, add_log_level_arg
from modelarrayio.utils.misc import load_and_normalize_cohort

logger = logging.getLogger(__name__)


def to_modelarray(
    cohort_file,
    backend='hdf5',
    output=Path('modelarray.h5'),
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=1,
    s3_workers=1,
    scalar_columns=None,
    group_mask_file=None,
    index_file=None,
    directions_file=None,
):
    """Load neuroimaging data and write to an HDF5 or TileDB modelarray file.

    The modality (NIfTI, CIFTI, or MIF/fixel) is autodetected from the source
    file extensions listed in the cohort file.

    Parameters
    ----------
    cohort_file : path-like
        Path to a CSV with demographic info and paths to data.
    group_mask_file : path-like, optional
        Path to a NIfTI binary group mask file. Required for NIfTI data.
    index_file : path-like, optional
        Nifti2 index file. Required for MIF/fixel data.
    directions_file : path-like, optional
        Nifti2 directions file. Required for MIF/fixel data.
    """
    cohort_long, modality = load_and_normalize_cohort(cohort_file, scalar_columns=scalar_columns)
    logger.info('Detected modality: %s', modality)

    common_kwargs = {
        'cohort_long': cohort_long,
        'backend': backend,
        'output': output,
        'storage_dtype': storage_dtype,
        'compression': compression,
        'compression_level': compression_level,
        'shuffle': shuffle,
        'chunk_voxels': chunk_voxels,
        'target_chunk_mb': target_chunk_mb,
        'workers': workers,
        's3_workers': s3_workers,
        'split_outputs': bool(scalar_columns),
    }

    if modality == 'nifti':
        if group_mask_file is None:
            raise ValueError(
                'Detected NIfTI data but --mask was not provided. '
                'Please supply the path to a binary group mask NIfTI file.'
            )
        return nifti_to_h5(group_mask_file=group_mask_file, **common_kwargs)

    if modality == 'mif':
        if index_file is None or directions_file is None:
            raise ValueError(
                'Detected MIF/fixel data but --index-file and/or --directions-file '
                'were not provided. Both are required for MIF data.'
            )
        return mif_to_h5(index_file=index_file, directions_file=directions_file, **common_kwargs)

    # cifti
    return cifti_to_h5(**common_kwargs)


def to_modelarray_main(**kwargs):
    """Entry point for the ``modelarrayio to-modelarray`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return to_modelarray(**kwargs)


def _parse_to_modelarray():
    parser = argparse.ArgumentParser(
        description=(
            'Convert neuroimaging data to a modelarray HDF5 file. '
            'The modality (NIfTI, CIFTI, or MIF/fixel) is autodetected from '
            'the source file extensions in the cohort file.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
        type=partial(_is_file, parser=parser),
    )
    parser.add_argument(
        '--output',
        help=(
            'Output path. For the hdf5 backend, path to an .h5 file; '
            'for the tiledb backend, path to a .tdb directory.'
        ),
        default=Path('modelarray.h5'),
        type=Path,
    )
    parser.add_argument(
        '--scalar-columns',
        '--scalar_columns',
        nargs='+',
        help=(
            'Column names containing scalar file paths when the cohort table is in wide format. '
            'If omitted, the cohort file must include "scalar_name" and "source_file" columns.'
        ),
    )
    parser.add_argument(
        '--backend',
        help='Storage backend for subject-by-element matrix',
        choices=['hdf5', 'tiledb'],
        default='hdf5',
    )
    parser.add_argument(
        '--dtype',
        help='Floating dtype for storing values: float32 (default) or float64',
        choices=['float32', 'float64'],
        default='float32',
        dest='storage_dtype',
    )
    parser.add_argument(
        '--compression',
        help=(
            'Compression filter (default gzip). '
            'gzip works for both backends; '
            'lzf is HDF5-only; '
            'zstd is TileDB-only.'
        ),
        choices=['gzip', 'zstd', 'lzf', 'none'],
        default='gzip',
    )
    parser.add_argument(
        '--compression-level',
        '--compression_level',
        type=int,
        help='Compression level (codec-dependent). Default 4.',
        default=4,
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Disable shuffle filter (enabled by default when compression is used).',
        default=True,
    )

    chunk_allocation_group = parser.add_mutually_exclusive_group()
    chunk_allocation_group.add_argument(
        '--chunk-voxels',
        '--chunk_voxels',
        type=int,
        help=(
            'Chunk/tile size along voxel/greyordinate/fixel axis. '
            'If 0, auto-compute based on --target-chunk-mb and number of subjects.'
        ),
        default=0,
    )
    chunk_allocation_group.add_argument(
        '--target-chunk-mb',
        '--target_chunk_mb',
        type=float,
        help='Target chunk/tile size in MiB when auto-computing the spatial axis length. Default 2.0.',
        default=2.0,
    )

    tiledb_group = parser.add_argument_group('TileDB arguments')
    tiledb_group.add_argument(
        '--workers',
        type=int,
        help=(
            'Maximum number of parallel TileDB write workers. '
            'Default 1. '
            'Has no effect when --backend=hdf5.'
        ),
        default=1,
    )

    s3_group = parser.add_argument_group('S3 arguments')
    s3_group.add_argument(
        '--s3-workers',
        '--s3_workers',
        type=int,
        default=1,
        help=(
            'Number of parallel worker processes for loading image files. '
            'Set > 1 to enable parallel downloads when cohort paths begin with s3://. '
            'Default 1 (serial).'
        ),
    )

    nifti_group = parser.add_argument_group('NIfTI arguments (required for NIfTI data)')
    nifti_group.add_argument(
        '--mask',
        help='Path to a NIfTI binary group mask file.',
        type=IsFile,
        default=None,
        dest='group_mask_file',
    )

    mif_group = parser.add_argument_group('MIF/fixel arguments (required for MIF/fixel data)')
    mif_group.add_argument(
        '--index-file',
        '--index_file',
        help='Nifti2 index file.',
        type=IsFile,
        default=None,
    )
    mif_group.add_argument(
        '--directions-file',
        '--directions_file',
        help='Nifti2 directions file.',
        type=IsFile,
        default=None,
    )

    add_log_level_arg(parser)

    return parser
