"""Convert CIFTI2 dscalar data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    add_backend_arg,
    add_cohort_arg,
    add_output_hdf5_arg,
    add_output_tiledb_arg,
    add_s3_workers_arg,
    add_scalar_columns_arg,
    add_storage_args,
    add_tiledb_storage_args,
)
from modelarrayio.utils.cifti import (
    _build_scalar_sources,
    _cohort_to_long_dataframe,
    _load_cohort_cifti,
    brain_names_to_dataframe,
    extract_cifti_scalar_data,
)

logger = logging.getLogger(__name__)


def cifti_to_h5(
    cohort_file,
    backend='hdf5',
    output_hdf5='fixeldb.h5',
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
    tdb_workers=None,
    scalar_columns=None,
    s3_workers=1,
):
    """Load all CIFTI data and write to an HDF5 file with configurable storage.

    Parameters
    ----------
    cohort_file : :obj:`str`
        Path to a csv with demographic info and paths to data
    backend : :obj:`str`
        Backend to use for storage
    output_hdf5 : :obj:`str`
        Path to a new .h5 file to be written
    output_tiledb : :obj:`str`
        Path to a new .tdb file to be written
    storage_dtype : :obj:`str`
        Floating type to store values
    compression : :obj:`str`
        HDF5 compression filter
    compression_level : :obj:`int`
        Gzip compression level (0-9)
    shuffle : :obj:`bool`
        Enable HDF5 shuffle filter
    chunk_voxels : :obj:`int`
        Chunk size along the voxel axis
    target_chunk_mb : :obj:`float`
        Target chunk size in MiB when auto-computing chunk_voxels
    tdb_compression : :obj:`str`
        TileDB compression filter
    tdb_compression_level : :obj:`int`
        TileDB compression level
    tdb_shuffle : :obj:`bool`
        Enable TileDB shuffle filter
    tdb_tile_voxels : :obj:`int`
        Tile size along the voxel axis
    tdb_target_tile_mb : :obj:`float`
        Target tile size in MiB when auto-computing tdb_tile_voxels
    tdb_workers : :obj:`int`
        Number of workers to use for parallel loading
    scalar_columns : :obj:`list`
        List of scalar columns to use
    s3_workers : :obj:`int`
        Number of workers to use for parallel loading

    Returns
    -------
    status : :obj:`int`
        Status of the operation. 0 if successful, 1 if failed.
    """
    cohort_df = pd.read_csv(cohort_file)
    cohort_long = _cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    output_path = Path(output)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    scalar_sources = _build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    if backend == 'hdf5':
        scalars, last_brain_names = _load_cohort_cifti(cohort_long, s3_workers)
        greyordinate_table, structure_names = brain_names_to_dataframe(last_brain_names)
        output_path = cli_utils.prepare_output_parent(output_path)
        with h5py.File(output_path, 'w') as h5_file:
            cli_utils.write_table_dataset(
                h5_file,
                'greyordinates',
                greyordinate_table,
                extra_attrs={'structure_names': structure_names},
            )
            cli_utils.write_hdf5_scalar_matrices(
                h5_file,
                scalars,
                scalar_sources,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
            )
        return int(not output_path.exists())

    output_path.mkdir(parents=True, exist_ok=True)
    if not scalar_sources:
        return 0

    _first_scalar, first_sources = next(iter(scalar_sources.items()))
    first_path = first_sources[0]
    _, reference_brain_names = extract_cifti_scalar_data(first_path)

    def _process_scalar_job(scalar_name, source_files):
        rows = []
        for source_file in source_files:
            cifti_data, _ = extract_cifti_scalar_data(
                source_file, reference_brain_names=reference_brain_names
            )
            rows.append(cifti_data)

        if rows:
            cli_utils.write_tiledb_scalar_matrices(
                output_path,
                {scalar_name: rows},
                {scalar_name: source_files},
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
                write_column_name_arrays=True,
            )
            return scalar_name

    scalar_names = list(scalar_sources.keys())
    worker_count = workers if isinstance(workers, int) and workers > 0 else None
    if worker_count is None:
        cpu_count = os.cpu_count() or 1
        worker_count = min(len(scalar_names), max(1, cpu_count))
    else:
        worker_count = min(len(scalar_names), worker_count)

    if worker_count <= 1:
        for scalar_name in scalar_names:
            _process_scalar_job(scalar_name, scalar_sources[scalar_name])
    else:
        desc = 'TileDB scalars'
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_process_scalar_job, scalar_name, scalar_sources[scalar_name]): (
                    scalar_name
                )
                for scalar_name in scalar_names
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                future.result()
    return 0


def cifti_to_h5_main(
    cohort_file,
    backend='hdf5',
    output_hdf5='fixelarray.h5',
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
    tdb_workers=None,
    scalar_columns=None,
    s3_workers=1,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio cifti-to-h5`` command."""
    cli_utils.configure_logging(log_level)
    return cifti_to_h5(
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
        tdb_workers=tdb_workers,
        scalar_columns=scalar_columns,
        s3_workers=s3_workers,
    )


def _parse_cifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of CIDTI2 dscalar data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cohort_arg(parser)
    add_scalar_columns_arg(parser)
    add_output_hdf5_arg(parser, default_name='fixelarray.h5')
    add_output_tiledb_arg(parser, default_name='arraydb.tdb')
    add_backend_arg(parser)
    add_storage_args(parser)
    add_tiledb_storage_args(parser)
    parser.add_argument(
        '--tdb-workers',
        '--tdb_workers',
        type=int,
        help=(
            'Maximum number of TileDB write workers. Default 0 (auto, uses CPU count). '
            'Set to 1 to disable parallel writes.'
        ),
        default=0,
    )
    add_s3_workers_arg(parser)
    return parser
