"""Convert NIfTI data to an HDF5 file."""

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
    add_output_arg,
    add_s3_workers_arg,
    add_storage_args,
)
from modelarrayio.storage import h5_storage, tiledb_storage
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

    group_mask_img = nb.load(group_mask_file)
    group_mask_matrix = group_mask_img.get_fdata() > 0
    voxel_coords = np.column_stack(np.nonzero(group_mask_img.get_fdata()))

    voxel_table = pd.DataFrame(
        {
            'voxel_id': np.arange(voxel_coords.shape[0]),
            'i': voxel_coords[:, 0],
            'j': voxel_coords[:, 1],
            'k': voxel_coords[:, 2],
        }
    )

    print('Extracting NIfTI data...')
    scalars, sources_lists = _load_cohort_voxels(cohort_df, group_mask_matrix, s3_workers)

    if backend == 'hdf5':
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        f = h5py.File(output, 'w')

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
        return int(not os.path.exists(output))
    else:
        os.makedirs(output, exist_ok=True)

        for scalar_name in scalars.keys():
            num_subjects = len(scalars[scalar_name])
            num_voxels = scalars[scalar_name][0].shape[0] if num_subjects > 0 else 0
            dataset_path = f'scalars/{scalar_name}/values'
            tiledb_storage.create_empty_scalar_matrix_array(
                output,
                dataset_path,
                num_subjects,
                num_voxels,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                tile_voxels=chunk_voxels,
                target_tile_mb=target_chunk_mb,
                sources_list=sources_lists[scalar_name],
            )
            uri = os.path.join(output, dataset_path)
            tiledb_storage.write_rows_in_column_stripes(uri, scalars[scalar_name])
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
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )
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
