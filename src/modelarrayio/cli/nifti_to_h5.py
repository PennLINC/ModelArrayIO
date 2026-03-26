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
    add_output_arg,
    add_s3_workers_arg,
    add_storage_args,
)
from modelarrayio.utils.voxels import _load_cohort_voxels

logger = logging.getLogger(__name__)


def nifti_to_h5(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output='voxeldb.h5',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    s3_workers=1,
):
    """Load all volume data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    group_mask_file : :obj:`str`
        Path to a NIfTI-1 binary group mask file.
    cohort_file : :obj:`str`
        Path to a CSV with demographic info and paths to data.
    backend : :obj:`str`
        Storage backend (``'hdf5'`` or ``'tiledb'``).
    output : :obj:`str`
        Output path. For the hdf5 backend, path to an .h5 file;
        for the tiledb backend, path to a .tdb directory.
    storage_dtype : :obj:`str`
        Floating type to store values. Options: ``'float32'`` (default), ``'float64'``.
    compression : :obj:`str`
        Compression filter. ``gzip`` works for both backends;
        ``lzf`` is HDF5-only; ``zstd`` is TileDB-only.
    compression_level : :obj:`int`
        Compression level (codec-dependent). Default 4.
    shuffle : :obj:`bool`
        Enable shuffle filter. Default True.
    chunk_voxels : :obj:`int`
        Chunk/tile size along the voxel axis. If 0, auto-compute. Default 0.
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing. Default 2.0.
    s3_workers : :obj:`int`
        Number of parallel workers for S3 downloads. Default 1.
    """
    cohort_df = pd.read_csv(cohort_file)
    output_path = Path(output)

    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
    voxel_coords = np.column_stack(np.nonzero(group_mask_matrix))

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
    output='voxeldb.h5',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    s3_workers=1,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio nifti-to-h5`` command."""
    cli_utils.configure_logging(log_level)
    return nifti_to_h5(
        group_mask_file=group_mask_file,
        cohort_file=cohort_file,
        backend=backend,
        output=output,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
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
    add_output_arg(parser, default_name='voxeldb.h5')
    add_backend_arg(parser)
    add_storage_args(parser)
    add_s3_workers_arg(parser)
    return parser
