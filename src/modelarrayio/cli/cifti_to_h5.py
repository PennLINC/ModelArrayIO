import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli.parser_utils import (
    add_backend_arg,
    add_cohort_arg,
    add_output_hdf5_arg,
    add_output_tiledb_arg,
    add_relative_root_arg,
    add_s3_workers_arg,
    add_scalar_columns_arg,
    add_storage_args,
    add_tiledb_storage_args,
)
from modelarrayio.storage import h5_storage, tiledb_storage
from modelarrayio.utils.cifti import (
    _build_scalar_sources,
    _cohort_to_long_dataframe,
    _load_cohort_cifti,
    brain_names_to_dataframe,
    extract_cifti_scalar_data,
)

logger = logging.getLogger(__name__)


def write_storage(
    cohort_file,
    backend='hdf5',
    output_hdf5='fixeldb.h5',
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
    relative_root : :obj:`str`
        Root to which all paths are relative
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
    cohort_path = os.path.join(relative_root, cohort_file)
    cohort_df = pd.read_csv(cohort_path)
    cohort_long = _cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    scalar_sources = _build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    if backend == 'hdf5':
        scalars, last_brain_names = _load_cohort_cifti(cohort_long, relative_root, s3_workers)

        output_file = os.path.join(relative_root, output_hdf5)
        f = h5py.File(output_file, 'w')

        greyordinate_table, structure_names = brain_names_to_dataframe(last_brain_names)
        greyordinatesh5 = f.create_dataset(
            name='greyordinates', data=greyordinate_table.to_numpy().T
        )
        greyordinatesh5.attrs['column_names'] = list(greyordinate_table.columns)
        greyordinatesh5.attrs['structure_names'] = structure_names

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
                sources_list=scalar_sources[scalar_name],
            )

            h5_storage.write_rows_in_column_stripes(dset, scalars[scalar_name])
        f.close()
        return int(not os.path.exists(output_file))
    else:
        base_uri = os.path.join(relative_root, output_tiledb)
        os.makedirs(base_uri, exist_ok=True)
        if not scalar_sources:
            return 0

        # Establish a reference brain axis once to ensure consistent ordering across workers.
        _first_scalar, first_sources = next(iter(scalar_sources.items()))
        first_path = os.path.join(relative_root, first_sources[0])
        _, reference_brain_names = extract_cifti_scalar_data(first_path)

        def _process_scalar_job(scalar_name, source_files):
            dataset_path = f'scalars/{scalar_name}/values'
            rows = []
            for source_file in source_files:
                scalar_file = os.path.join(relative_root, source_file)
                cifti_data, _ = extract_cifti_scalar_data(
                    scalar_file, reference_brain_names=reference_brain_names
                )
                rows.append(cifti_data)

            num_subjects = len(rows)
            if num_subjects == 0:
                return scalar_name
            num_items = rows[0].shape[0]
            tiledb_storage.create_empty_scalar_matrix_array(
                base_uri,
                dataset_path,
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=tdb_compression,
                compression_level=tdb_compression_level,
                shuffle=tdb_shuffle,
                tile_voxels=tdb_tile_voxels,
                target_tile_mb=tdb_target_tile_mb,
                sources_list=source_files,
            )
            # write column names array for ModelArray compatibility
            tiledb_storage.write_column_names(base_uri, scalar_name, source_files)
            uri = os.path.join(base_uri, dataset_path)
            tiledb_storage.write_rows_in_column_stripes(uri, rows)
            return scalar_name

        scalar_names = list(scalar_sources.keys())
        worker_count = tdb_workers if isinstance(tdb_workers, int) and tdb_workers > 0 else None
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
                    executor.submit(
                        _process_scalar_job, scalar_name, scalar_sources[scalar_name]
                    ): scalar_name
                    for scalar_name in scalar_names
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    future.result()
        return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Create a hdf5 file of CIDTI2 dscalar data')
    add_cohort_arg(parser)
    add_scalar_columns_arg(parser)
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
