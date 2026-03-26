"""Convert CIFTI2 dscalar data to an HDF5 file."""

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
    add_output_arg,
    add_s3_workers_arg,
    add_scalar_columns_arg,
    add_storage_args,
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


def cifti_to_h5(
    cohort_file,
    backend='hdf5',
    output='fixelarray.h5',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    scalar_columns=None,
    s3_workers=1,
):
    """Load all CIFTI data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    cohort_file : :obj:`str`
        Path to a csv with demographic info and paths to data
    backend : :obj:`str`
        Backend to use for storage (``'hdf5'`` or ``'tiledb'``)
    output : :obj:`str`
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
        Chunk/tile size along the greyordinate axis (0 = auto)
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing the spatial axis length
    workers : :obj:`int`
        Maximum number of parallel TileDB write workers (``None`` = auto).
        Has no effect when ``backend='hdf5'``.
    scalar_columns : :obj:`list`
        List of scalar columns to use
    s3_workers : :obj:`int`
        Number of workers for parallel S3 downloads

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
    """
    cohort_df = pd.read_csv(cohort_file)
    cohort_long = _cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    scalar_sources = _build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    if backend == 'hdf5':
        scalars, last_brain_names = _load_cohort_cifti(cohort_long, s3_workers)

        f = h5py.File(output, 'w')

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
        return int(not os.path.exists(output))
    else:
        os.makedirs(output, exist_ok=True)
        if not scalar_sources:
            return 0

        # Establish a reference brain axis once to ensure consistent ordering across workers.
        _first_scalar, first_sources = next(iter(scalar_sources.items()))
        first_path = first_sources[0]
        _, reference_brain_names = extract_cifti_scalar_data(first_path)

        def _process_scalar_job(scalar_name, source_files):
            dataset_path = f'scalars/{scalar_name}/values'
            rows = []
            for source_file in source_files:
                cifti_data, _ = extract_cifti_scalar_data(
                    source_file, reference_brain_names=reference_brain_names
                )
                rows.append(cifti_data)

            num_subjects = len(rows)
            if num_subjects == 0:
                return scalar_name
            num_items = rows[0].shape[0]
            tiledb_storage.create_empty_scalar_matrix_array(
                output,
                dataset_path,
                num_subjects,
                num_items,
                storage_dtype=storage_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                tile_voxels=chunk_voxels,
                target_tile_mb=target_chunk_mb,
                sources_list=source_files,
            )
            # write column names array for ModelArray compatibility
            tiledb_storage.write_column_names(output, scalar_name, source_files)
            uri = os.path.join(output, dataset_path)
            tiledb_storage.write_rows_in_column_stripes(uri, rows)
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
                    executor.submit(
                        _process_scalar_job, scalar_name, scalar_sources[scalar_name]
                    ): scalar_name
                    for scalar_name in scalar_names
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    future.result()
        return 0


def cifti_to_h5_main(
    cohort_file,
    backend='hdf5',
    output='fixelarray.h5',
    storage_dtype='float32',
    compression='gzip',
    compression_level=4,
    shuffle=True,
    chunk_voxels=0,
    target_chunk_mb=2.0,
    workers=None,
    scalar_columns=None,
    s3_workers=1,
    log_level='INFO',
):
    """Entry point for the ``modelarrayio cifti-to-h5`` command."""
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    return cifti_to_h5(
        cohort_file=cohort_file,
        backend=backend,
        output=output,
        storage_dtype=storage_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        chunk_voxels=chunk_voxels,
        target_chunk_mb=target_chunk_mb,
        workers=workers,
        scalar_columns=scalar_columns,
        s3_workers=s3_workers,
    )


def _parse_cifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of CIFTI2 dscalar data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cohort_arg(parser)
    add_scalar_columns_arg(parser)
    add_output_arg(parser, default_name='fixelarray.h5')
    add_backend_arg(parser)
    add_storage_args(parser)
    parser.add_argument(
        '--workers',
        type=int,
        help=(
            'Maximum number of parallel TileDB write workers. '
            'Default 0 (auto, uses CPU count). '
            'Set to 1 to disable parallel writes. '
            'Has no effect when --backend=hdf5.'
        ),
        default=0,
    )
    add_s3_workers_arg(parser)
    return parser
