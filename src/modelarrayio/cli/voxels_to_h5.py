import argparse
import logging
import os
from functools import partial

import h5py
import nibabel as nb
import numpy as np
import pandas as pd

from modelarrayio.cli.parser_utils import (
    _is_file,
    add_backend_arg,
    add_cohort_arg,
    add_output_hdf5_arg,
    add_output_tiledb_arg,
    add_relative_root_arg,
    add_s3_workers_arg,
    add_storage_args,
    add_tiledb_storage_args,
)
from modelarrayio.storage import h5_storage, tiledb_storage
from modelarrayio.utils.voxels import _load_cohort_voxels

logger = logging.getLogger(__name__)


def write_storage(
    group_mask_file,
    cohort_file,
    backend='hdf5',
    output_hdf5='voxeldb.h5',
    output_tiledb='arraydb.tdb',
    relative_root='/',
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
    relative_root: str
        Path to which group_mask_file and cohort_file (and its contents) are relative.
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
    cohort_df = pd.read_csv(os.path.join(relative_root, cohort_file))

    # Load the group mask image to define the rows of the matrix
    group_mask_img = nb.load(os.path.join(relative_root, group_mask_file))
    # get_fdata(): get matrix data in float format
    group_mask_matrix = group_mask_img.get_fdata() > 0
    # np.nonzero() returns the coords of nonzero elements;
    # then np.column_stack() stack them together as an (#voxels, 3) array
    voxel_coords = np.column_stack(np.nonzero(group_mask_img.get_fdata()))

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

    # upload each cohort's data
    print('Extracting NIfTI data...')
    scalars, sources_lists = _load_cohort_voxels(
        cohort_df, group_mask_matrix, relative_root, s3_workers
    )

    # Write the output:
    if backend == 'hdf5':
        output_file = os.path.join(relative_root, output_hdf5)
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        f = h5py.File(output_file, 'w')

        voxelsh5 = f.create_dataset(name='voxels', data=voxel_table.to_numpy().T)
        voxelsh5.attrs['column_names'] = list(voxel_table.columns)

        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_voxels = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dset = h5_storage.create_empty_scalar_matrix_dataset(
                f,
                f'scalars/{scalar_name}/values',
                num_subjects,
                num_voxels,
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
        return int(not os.path.exists(output_file))
    else:
        # TileDB backend
        base_uri = os.path.join(relative_root, output_tiledb)
        os.makedirs(base_uri, exist_ok=True)

        # Store voxel coordinates as a small TileDB array (optional):
        # we store as metadata on base group
        # Here we serialize as a dense 2D array for parity with HDF5 tables if desired,
        # but keep it simple: metadata JSON
        # Create arrays for each scalar
        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_voxels = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dataset_path = f'scalars/{scalar_name}/values'
            tiledb_storage.create_empty_scalar_matrix_array(
                base_uri,
                dataset_path,
                num_subjects,
                num_voxels,
                storage_dtype=storage_dtype,
                compression=tdb_compression,
                compression_level=tdb_compression_level,
                shuffle=tdb_shuffle,
                tile_voxels=tdb_tile_voxels,
                target_tile_mb=tdb_target_tile_mb,
                sources_list=sources_lists[scalar_name],
            )
            # Stripe-write
            uri = os.path.join(base_uri, dataset_path)
            tiledb_storage.write_rows_in_column_stripes(uri, scalars[scalar_name])
        return 0


def get_parser():
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
    add_relative_root_arg(parser)
    add_output_hdf5_arg(parser, default_name='fixelarray.h5')
    add_output_tiledb_arg(parser, default_name='arraydb.tdb')
    add_backend_arg(parser)
    add_storage_args(parser)
    add_tiledb_storage_args(parser)
    add_s3_workers_arg(parser)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    log_level = kwargs.pop('log_level')
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    return write_storage(**kwargs)
