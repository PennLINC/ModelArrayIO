"""Convert NIfTI data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_scalar_columns_arg, add_to_modelarray_args
from modelarrayio.utils.misc import cohort_to_long_dataframe
from modelarrayio.utils.voxels import load_cohort_voxels

logger = logging.getLogger(__name__)


def nifti_to_h5(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output=Path('voxelarray.h5'),
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    s3_workers=1,
    scalar_columns=None,
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
    output : :obj:`pathlib.Path`
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
    workers : :obj:`int`
        Maximum number of parallel TileDB write workers. Default 0 (auto).
        Has no effect when ``backend='hdf5'``.
    s3_workers : :obj:`int`
        Number of parallel workers for S3 downloads. Default 1.
    """
    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
    voxel_coords = np.column_stack(np.nonzero(group_mask_matrix))

    cohort_df = pd.read_csv(cohort_file)
    cohort_long = cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': voxel_coords[:, 0],
            'j': voxel_coords[:, 1],
            'k': voxel_coords[:, 2],
        }
    )

    logger.info('Extracting NIfTI data...')
    scalars, sources_lists = load_cohort_voxels(cohort_long, group_mask_matrix, s3_workers)
    if not sources_lists:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    if backend == 'hdf5':
        output = cli_utils.prepare_output_parent(output)
        with h5py.File(output, 'w') as h5_file:
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
        return int(not output.exists())

    output.mkdir(parents=True, exist_ok=True)

    scalar_names = list(sources_lists.keys())
    worker_count = workers if isinstance(workers, int) and workers > 0 else None
    if worker_count is None:
        cpu_count = os.cpu_count() or 1
        worker_count = min(len(scalar_names), max(1, cpu_count))
    else:
        worker_count = min(len(scalar_names), worker_count)

    def _write_scalar_job(scalar_name):
        cli_utils.write_tiledb_scalar_matrices(
            output,
            {scalar_name: scalars[scalar_name]},
            {scalar_name: sources_lists[scalar_name]},
            storage_dtype=storage_dtype,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_voxels=chunk_voxels,
            target_chunk_mb=target_chunk_mb,
        )

    if worker_count <= 1:
        for scalar_name in scalar_names:
            _write_scalar_job(scalar_name)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_write_scalar_job, scalar_name): scalar_name
                for scalar_name in scalar_names
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc='TileDB scalars'):
                future.result()
    return 0


def nifti_to_h5_main(**kwargs):
    """Entry point for the ``modelarrayio nifti-to-h5`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return nifti_to_h5(**kwargs)


def _parse_nifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of volume data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    # NIfTI-specific arguments
    parser.add_argument(
        '--group-mask-file',
        '--group_mask_file',
        help='Path to a group mask file',
        required=True,
        type=IsFile,
    )

    # Common arguments
    add_to_modelarray_args(parser, default_output='voxelarray.h5')
    return parser
