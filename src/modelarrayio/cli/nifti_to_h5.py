"""Convert NIfTI data to an HDF5 file."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.utils.misc import cohort_to_long_dataframe
from modelarrayio.utils.nifti import load_cohort_voxels

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
    workers=1,
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
        Maximum number of parallel TileDB write workers. Default 1.
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

    scalar_names = list(sources_lists.keys())
    split_scalar_outputs = bool(scalar_columns)

    if backend == 'hdf5':
        if split_scalar_outputs:
            outputs: list[Path] = []
            for scalar_name in scalar_names:
                scalar_output = cli_utils.prepare_output_parent(
                    cli_utils.prefixed_output_path(output, scalar_name)
                )
                with h5py.File(scalar_output, 'w') as h5_file:
                    cli_utils.write_table_dataset(h5_file, 'voxels', voxel_table)
                    cli_utils.write_hdf5_scalar_matrices(
                        h5_file,
                        {scalar_name: scalars[scalar_name]},
                        {scalar_name: sources_lists[scalar_name]},
                        storage_dtype=storage_dtype,
                        compression=compression,
                        compression_level=compression_level,
                        shuffle=shuffle,
                        chunk_voxels=chunk_voxels,
                        target_chunk_mb=target_chunk_mb,
                    )
                outputs.append(scalar_output)
            return int(not all(path.exists() for path in outputs))

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

    worker_count = min(len(scalar_names), workers)

    def _write_scalar_job(scalar_name):
        scalar_output = (
            cli_utils.prefixed_output_path(output, scalar_name) if split_scalar_outputs else output
        )
        cli_utils.write_tiledb_scalar_matrices(
            scalar_output,
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
