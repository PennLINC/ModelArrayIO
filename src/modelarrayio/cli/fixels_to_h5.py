import argparse
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli.parser_utils import _is_file
from modelarrayio.storage import h5_storage, tiledb_storage
from modelarrayio.utils.fixels import gather_fixels, mif_to_nifti2


def write_storage(
    index_file,
    directions_file,
    cohort_file,
    backend='hdf5',
    output_hdf5=Path('fixeldb.h5'),
    output_tiledb=Path('arraydb.tdb'),
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
):
    """Load all fixeldb data and write to an HDF5 file with configurable storage.

    Parameters
    ----------
    index_file : :obj:`pathlib.Path`
        Path to a Nifti2 index file
    directions_file : :obj:`pathlib.Path`
        Path to a Nifti2 directions file
    cohort_file : :obj:`pathlib.Path`
        Path to a csv with demographic info and paths to data
    backend : :obj:`str`
        Backend to use for storage
    output_hdf5 : :obj:`pathlib.Path`
        Path to a new .h5 file to be written
    output_tiledb : :obj:`pathlib.Path`
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

    Returns
    -------
    status : :obj:`int`
        Status of the operation. 0 if successful, 1 if failed.
    """
    # gather fixel data
    fixel_table, voxel_table = gather_fixels(index_file, directions_file)

    # gather cohort data
    cohort_df = pd.read_csv(cohort_file)

    # upload each cohort's data
    scalars = defaultdict(list)
    sources_lists = defaultdict(list)
    print('Extracting .mif data...')
    # ix: index of row (start from 0); row: one row of data
    for _ix, row in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0]):
        scalar_file = row['source_file']
        _scalar_img, scalar_data = mif_to_nifti2(scalar_file)
        scalars[row['scalar_name']].append(scalar_data)  # append to specific scalar_name
        # append source mif filename to specific scalar_name
        sources_lists[row['scalar_name']].append(row['source_file'])

    # Write the output
    if backend == 'hdf5':
        f = h5py.File(output_hdf5, 'w')

        fixelsh5 = f.create_dataset(name='fixels', data=fixel_table.to_numpy().T)
        fixelsh5.attrs['column_names'] = list(fixel_table.columns)

        voxelsh5 = f.create_dataset(name='voxels', data=voxel_table.to_numpy().T)
        voxelsh5.attrs['column_names'] = list(voxel_table.columns)

        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_items = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dset = h5_storage.create_empty_scalar_matrix_dataset(
                f,
                f'scalars/{scalar_name}/values',
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_voxels=chunk_voxels,
                target_chunk_mb=target_chunk_mb,
                sources_list=sources_lists[scalar_name],
            )

            h5_storage.write_rows_in_column_stripes(dset, scalars[scalar_name])
        f.close()
        return int(not output_hdf5.exists())

    else:
        output_tiledb.mkdir(parents=True, exist_ok=True)
        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_items = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dataset_path = f'scalars/{scalar_name}/values'
            tiledb_storage.create_empty_scalar_matrix_array(
                output_tiledb,
                dataset_path,
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=tdb_compression,
                compression_level=tdb_compression_level,
                shuffle=tdb_shuffle,
                tile_voxels=tdb_tile_voxels,
                target_tile_mb=tdb_target_tile_mb,
                sources_list=sources_lists[scalar_name],
            )
            uri = output_tiledb / dataset_path
            tiledb_storage.write_rows_in_column_stripes(uri, scalars[scalar_name])

        return 0


def get_parser():
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
    parser.add_argument(
        '--cohort-file',
        '--cohort_file',
        help='Path to a csv with demographic info and paths to data.',
        required=True,
        type=IsFile,
    )
    parser.add_argument(
        '--output-hdf5',
        '--output_hdf5',
        help='Name of HDF5 (.h5) file where outputs will be saved.',
        default='fixelarray.h5',
    )
    parser.add_argument(
        '--output-tiledb',
        '--output_tiledb',
        help='Base URI (directory) where TileDB arrays will be created.',
        default='arraydb.tdb',
    )
    parser.add_argument(
        '--backend',
        help='Storage backend for subject-by-element matrix',
        choices=['hdf5', 'tiledb'],
        default='hdf5',
    )
    # Storage configuration (match voxels.py)
    parser.add_argument(
        '--dtype',
        help='Floating dtype for storing values: float32 (default) or float64',
        choices=['float32', 'float64'],
        default='float32',
        dest='storage_dtype',
    )
    parser.add_argument(
        '--compression',
        help='HDF5 compression filter: gzip (default), lzf, none',
        choices=['gzip', 'lzf', 'none'],
        default='gzip',
    )
    parser.add_argument(
        '--tdb-compression',
        '--tdb_compression',
        help='TileDB compression: zstd (default), gzip, none',
        choices=['zstd', 'gzip', 'none'],
        default='zstd',
    )
    parser.add_argument(
        '--compression-level',
        '--compression_level',
        type=int,
        help='Gzip compression level 0-9 (only if --compression=gzip). Default 4',
        default=4,
    )
    parser.add_argument(
        '--tdb-compression-level',
        '--tdb_compression_level',
        type=int,
        help='Compression level for TileDB (codec-dependent).',
        default=5,
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Disable HDF5 shuffle filter (enabled by default if compression is used).',
        default=True,
    )
    parser.add_argument(
        '--tdb-no-shuffle',
        dest='tdb_shuffle',
        action='store_false',
        help='Disable TileDB shuffle filter (enabled by default).',
        default=True,
    )
    parser.add_argument(
        '--chunk-voxels',
        '--chunk_voxels',
        type=int,
        help=(
            'Chunk size along fixel/voxel axis. '
            'If 0, auto-compute based on --target-chunk-mb and number of subjects'
        ),
        default=0,
    )
    parser.add_argument(
        '--tdb-tile-voxels',
        '--tdb_tile_voxels',
        type=int,
        help=(
            'Tile length along item axis for TileDB. '
            'If 0, auto-compute based on --tdb-target-tile-mb'
        ),
        default=0,
    )
    parser.add_argument(
        '--target-chunk-mb',
        '--target_chunk_mb',
        type=float,
        help='Target chunk size in MiB when auto-computing item chunk length. Default 2.0',
        default=2.0,
    )
    parser.add_argument(
        '--tdb-target-tile-mb',
        '--tdb_target_tile_mb',
        type=float,
        help='Target tile size in MiB when auto-computing item tile length. Default 2.0',
        default=2.0,
    )
    parser.add_argument(
        '--log-level',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default INFO; set to WARNING to reduce verbosity)',
        default='INFO',
    )
    return parser


def main():
    """Main function to write fixel data to an HDF5 or TileDB file."""
    parser = get_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    log_level = kwargs.pop('log_level')
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    return write_storage(**kwargs)
