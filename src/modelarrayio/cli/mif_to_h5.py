"""Convert MIF data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import _is_file, add_to_modelarray_args
from modelarrayio.utils.mif import gather_fixels, load_cohort_mif
from modelarrayio.utils.misc import cohort_to_long_dataframe

logger = logging.getLogger(__name__)


def mif_to_h5(
    index_file,
    directions_file,
    cohort_file,
    backend='hdf5',
    output=Path('fixelarray.h5'),
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
    """Load all fixeldb data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    index_file : :obj:`pathlib.Path`
        Path to a Nifti2 index file
    directions_file : :obj:`pathlib.Path`
        Path to a Nifti2 directions file
    cohort_file : :obj:`pathlib.Path`
        Path to a csv with demographic info and paths to data
    backend : :obj:`str`
        Backend to use for storage (``'hdf5'`` or ``'tiledb'``)
    output : :obj:`pathlib.Path`
        Output path. For the hdf5 backend, path to an .h5 file;
        for the tiledb backend, path to a .tdb directory.
    storage_dtype : :obj:`str`
        Floating type to store values
    compression : :obj:`str`
        Compression filter. ``gzip`` works for both backends;
        ``lzf`` is HDF5-only; ``zstd`` is TileDB-only.
    compression_level : :obj:`int`
        Compression level (codec-dependent)
    shuffle : :obj:`bool`
        Enable shuffle filter
    chunk_voxels : :obj:`int`
        Chunk/tile size along the fixel axis (0 = auto)
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing the spatial axis length
    workers : :obj:`int`
        Maximum number of parallel TileDB write workers. Default 1.
        Has no effect when ``backend='hdf5'``.
    s3_workers : :obj:`int`
        Number of parallel workers for S3 downloads. Default 1.

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
    """
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(index_file, directions_file)

    cohort_df = pd.read_csv(cohort_file)
    cohort_long = cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')

    logger.info('Extracting .mif data...')
    scalars, sources_lists = load_cohort_mif(cohort_long, s3_workers)
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
                    cli_utils.write_table_dataset(h5_file, 'fixels', fixel_table)
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
            cli_utils.write_table_dataset(h5_file, 'fixels', fixel_table)
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


def mif_to_h5_main(**kwargs):
    """Entry point for the ``modelarrayio mif-to-h5`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return mif_to_h5(**kwargs)


def _parse_mif_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of fixel data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

    # MIF-specific arguments
    parser.add_argument(
        '--index-file',
        '--index_file',
        help='Index File',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--directions-file',
        '--directions_file',
        help='Directions File',
        required=True,
        type=IsFile,
    )

    # Common arguments
    add_to_modelarray_args(parser, default_output='fixelarray.h5')
    return parser
