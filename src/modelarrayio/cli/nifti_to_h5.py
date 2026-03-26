"""Convert NIfTI data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    _is_file,
    add_backend_arg,
    add_cohort_arg,
    add_output_hdf5_arg,
    add_output_tiledb_arg,
    add_s3_workers_arg,
    add_storage_args,
    add_tiledb_storage_args,
)
from modelarrayio.utils.voxels import _load_cohort_voxels

logger = logging.getLogger(__name__)


def nifti_to_h5(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output_hdf5='voxeldb.h5',
    output_tiledb='arraydb.tdb',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    tdb_compression='zstd',
    tdb_compression_level=5,
    tdb_shuffle=True,
    tdb_tile_voxels=0,
    tdb_target_tile_mb=2.0,
    s3_workers=1,
):
    """Load all volume data and write to an HDF5 file with configurable storage.

    Parameters
    ----------
    group_mask_file: str
        Path to a NIfTI-1 binary group mask file.
    cohort_file: str
        Path to a CSV with demographic info and paths to data.
    output_hdf5: str
        Path to a new .h5 file to be written.
    storage_dtype: str
        Floating type to store values. Options: 'float32' (default), 'float64'.
    compression: str
        HDF5 compression filter. Options: 'gzip' (default), 'lzf', 'none'.
    compression_level: int
        Gzip compression level (0-9). Only used when compression == 'gzip'. Default 4.
    shuffle: bool
        Enable HDF5 shuffle filter to improve compression.
        Default True (effective when compression != 'none').
    chunk_voxels: int
        Chunk size along the voxel axis. If 0, auto-compute using target_chunk_mb. Default 0.
    target_chunk_mb: float
        Target chunk size in MiB when auto-computing chunk_voxels. Default 2.0.
    """
    # gather cohort data
    cohort_df = pd.read_csv(cohort_file)
    output_path = Path(output)

    # Load the group mask image to define the rows of the matrix
    group_mask_img = nb.load(group_mask_file)
    # get_fdata(): get matrix data in float format
    group_mask_matrix = group_mask_img.get_fdata() > 0
    voxel_coords = np.column_stack(np.nonzero(group_mask_matrix))

    # voxel_table: records the coordinations of the nonzero voxels;
    # coord starts from 0 (because using python)
    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': voxel_coords[:, 0],
            'j': voxel_coords[:, 1],
            'k': voxel_coords[:, 2],
        }
    )

    logger.info('Extracting NIfTI data...')
    scalars, sources_lists = _load_cohort_voxels(cohort_df, group_mask_matrix, s3_workers)

    # Write the output:
    if backend == 'hdf5':
        output_path = cli_utils.prepare_output_parent(output_path)
        with h5py.File(output_path, 'w') as h5_file:
            cli_utils.write_table_dataset(h5_file, 'voxels', voxel_table)
            cli_utils.write_hdf5_scalar_matrices(
                h5_file,
                scalars,
                sources_lists,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
            )
        return int(not output_path.exists())

    cli_utils.write_tiledb_scalar_matrices(
        output_path,
        scalars,
        sources_lists,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
    )
    return 0


def nifti_to_h5_main(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output_hdf5='voxeldb.h5',
    output_tiledb='arraydb.tdb',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    tdb_compression='zstd',
    tdb_compression_level=5,
    tdb_shuffle=True,
    tdb_tile_voxels=0,
    tdb_target_tile_mb=2.0,
    s3_workers=1,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio nifti-to-h5`` command."""
    cli_utils.configure_logging(log_level)
    return nifti_to_h5(
        group_mask_file=group_mask_file,
        cohort_file=cohort_file,
        backend=backend,
        output_hdf5=output_hdf5,
        output_tiledb=output_tiledb,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
        tdb_compression=tdb_compression,
        tdb_compression_level=tdb_compression_level,
        tdb_shuffle=tdb_shuffle,
        tdb_tile_voxels=tdb_tile_voxels,
        tdb_target_tile_mb=tdb_target_tile_mb,
        s3_workers=s3_workers,
    )


def _parse_nifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of volume data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)
    parser.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        help='Path to a group mask file',
        required=True,
        type=IsFile,
    )
    add_cohort_arg(parser)
    add_output_hdf5_arg(parser, default_name='fixelarray.h5')
    add_output_tiledb_arg(parser, default_name='arraydb.tdb')
    add_backend_arg(parser)
    add_storage_args(parser)
    add_tiledb_storage_args(parser)
    add_s3_workers_arg(parser)
    return parser
