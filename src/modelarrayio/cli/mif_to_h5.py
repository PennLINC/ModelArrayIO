"""Convert MIF data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import (
    _is_file,
    add_backend_arg,
    add_cohort_arg,
    add_output_arg,
    add_storage_args,
)
from modelarrayio.utils.fixels import gather_fixels, mif_to_nifti2

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

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
    """
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(index_file, directions_file)
    output_path = Path(output)

    # gather cohort data
    cohort_df = pd.read_csv(cohort_file)

    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    logger.info('Extracting .mif data...')
    for row in tqdm(cohort_df.itertuples(index=False), total=cohort_df.shape[0]):
        scalar_file = row.source_file
        _scalar_img, scalar_data = mif_to_nifti2(scalar_file)
        scalars[row.scalar_name].append(scalar_data)
        sources_lists[row.scalar_name].append(row.source_file)

    # Write the output
    if backend == 'hdf5':
        output_path = cli_utils.prepare_output_parent(output_path)
        with h5py.File(output_path, 'w') as h5_file:
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


def mif_to_h5_main(
    index_file,
    directions_file,
    cohort_file,
    backend='hdf5',
    output='fixelarray.h5',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio mif-to-h5`` command."""
    cli_utils.configure_logging(log_level)
    return mif_to_h5(
        index_file=index_file,
        directions_file=directions_file,
        cohort_file=cohort_file,
        backend=backend,
        output=output,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
    )


def _parse_mif_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of fixel data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    IsFile = partial(_is_file, parser=parser)

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
    add_cohort_arg(parser)
    add_output_arg(parser, default_name='fixelarray.h5')
    add_backend_arg(parser)
    add_storage_args(parser)
    return parser
