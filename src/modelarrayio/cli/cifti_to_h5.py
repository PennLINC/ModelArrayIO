"""Convert CIFTI2 dscalar data to an HDF5 file."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm

from modelarrayio.cli import utils as cli_utils
from modelarrayio.cli.parser_utils import add_to_modelarray_args
from modelarrayio.utils.cifti import (
    brain_names_to_dataframe,
    extract_cifti_scalar_data,
    load_cohort_cifti,
)
from modelarrayio.utils.misc import build_scalar_sources, cohort_to_long_dataframe

logger = logging.getLogger(__name__)


def cifti_to_h5(
    cohort_file,
    backend='hdf5',
    output=Path('greyordinatearray.h5'),
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
    """Load all CIFTI data and write to an HDF5 or TileDB file.

    Parameters
    ----------
    cohort_file : :obj:`str`
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
        Chunk/tile size along the greyordinate axis (0 = auto)
    target_chunk_mb : :obj:`float`
        Target chunk/tile size in MiB when auto-computing the spatial axis length
    workers : :obj:`int`
        Maximum number of parallel TileDB write workers. Default 1.
        Has no effect when ``backend='hdf5'``.
    s3_workers : :obj:`int`
        Number of workers for parallel S3 downloads
    scalar_columns : :obj:`list`
        List of scalar columns to use

    Returns
    -------
    status : :obj:`int`
        0 if successful, 1 if failed.
    """
    cohort_df = pd.read_csv(cohort_file)
    cohort_long = cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    scalar_sources = build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')
    scalar_names = list(scalar_sources.keys())
    split_scalar_outputs = bool(scalar_columns)

    if backend == 'hdf5':
        if split_scalar_outputs:
            scalars, last_brain_names = load_cohort_cifti(cohort_long, s3_workers)
            greyordinate_table, structure_names = brain_names_to_dataframe(last_brain_names)
            outputs: list[Path] = []
            for scalar_name in scalar_names:
                scalar_output = cli_utils.prepare_output_parent(
                    cli_utils.prefixed_output_path(output, scalar_name)
                )
                with h5py.File(scalar_output, 'w') as h5_file:
                    cli_utils.write_table_dataset(
                        h5_file,
                        'greyordinates',
                        greyordinate_table,
                        extra_attrs={'structure_names': structure_names},
                    )
                    cli_utils.write_hdf5_scalar_matrices(
                        h5_file,
                        {scalar_name: scalars[scalar_name]},
                        {scalar_name: scalar_sources[scalar_name]},
                        storage_dtype=storage_dtype,
                        compression=compression,
                        compression_level=compression_level,
                        shuffle=shuffle,
                        chunk_voxels=chunk_voxels,
                        target_chunk_mb=target_chunk_mb,
                    )
                outputs.append(scalar_output)
            return int(not all(path.exists() for path in outputs))

        scalars, last_brain_names = load_cohort_cifti(cohort_long, s3_workers)
        greyordinate_table, structure_names = brain_names_to_dataframe(last_brain_names)
        output = cli_utils.prepare_output_parent(output)
        with h5py.File(output, 'w') as h5_file:
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
        return int(not output.exists())

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
            scalar_output = (
                cli_utils.prefixed_output_path(output, scalar_name)
                if split_scalar_outputs
                else output
            )
            cli_utils.write_tiledb_scalar_matrices(
                scalar_output,
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

    worker_count = min(len(scalar_names), workers)

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


def cifti_to_h5_main(**kwargs):
    """Entry point for the ``modelarrayio cifti-to-h5`` command."""
    log_level = kwargs.pop('log_level', 'INFO')
    cli_utils.configure_logging(log_level)
    return cifti_to_h5(**kwargs)


def _parse_cifti_to_h5():
    parser = argparse.ArgumentParser(
        description='Create a hdf5 file of CIFTI2 dscalar data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_to_modelarray_args(parser, default_output='greyordinatearray.h5')
    return parser
