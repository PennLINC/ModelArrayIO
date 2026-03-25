import argparse
import logging
import os
import os.path as op
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import nibabel as nb
import numpy as np
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
from modelarrayio.h5_storage import create_empty_scalar_matrix_dataset as h5_create_empty
from modelarrayio.h5_storage import write_rows_in_column_stripes as h5_write_stripes
from modelarrayio.s3_utils import is_s3_path, load_nibabel
from modelarrayio.tiledb_storage import create_empty_scalar_matrix_array as tdb_create_empty
from modelarrayio.tiledb_storage import write_column_names as tdb_write_column_names
from modelarrayio.tiledb_storage import write_rows_in_column_stripes as tdb_write_stripes

logger = logging.getLogger(__name__)


def _cohort_to_long_dataframe(cohort_df, scalar_columns=None):
    scalar_columns = [col for col in (scalar_columns or []) if col]
    if scalar_columns:
        missing = [col for col in scalar_columns if col not in cohort_df.columns]
        if missing:
            raise ValueError(f'Wide-format cohort is missing scalar columns: {missing}')
        records = []
        for _, row in cohort_df.iterrows():
            for scalar_col in scalar_columns:
                source_val = row[scalar_col]
                if pd.isna(source_val) or source_val is None:
                    continue
                source_str = str(source_val).strip()
                if not source_str:
                    continue
                records.append({'scalar_name': scalar_col, 'source_file': source_str})
        return pd.DataFrame.from_records(records, columns=['scalar_name', 'source_file'])

    required = {'scalar_name', 'source_file'}
    missing = required - set(cohort_df.columns)
    if missing:
        raise ValueError(
            f'Cohort file must contain columns {sorted(required)} when --scalar-columns is not used.'
        )

    long_df = cohort_df[list(required)].copy()
    long_df = long_df.dropna(subset=['scalar_name', 'source_file'])
    long_df['scalar_name'] = long_df['scalar_name'].astype(str).str.strip()
    long_df['source_file'] = long_df['source_file'].astype(str).str.strip()
    long_df = long_df[(long_df['scalar_name'] != '') & (long_df['source_file'] != '')]
    return long_df.reset_index(drop=True)


def _build_scalar_sources(long_df):
    scalar_sources = OrderedDict()
    for row in long_df.itertuples(index=False):
        scalar = str(row.scalar_name)
        source = str(row.source_file)
        if not scalar or not source:
            continue
        scalar_sources.setdefault(scalar, []).append(source)
    return scalar_sources


def extract_cifti_scalar_data(cifti_file, reference_brain_names=None):
    """Load a scalar cifti file and get its data and mapping

    Parameters
    ----------
    cifti_file : :obj:`str`
        CIFTI2 file on disk
    reference_brain_names : :obj:`numpy.ndarray`
        Array of vertex names

    Returns
    -------
    cifti_scalar_data: :obj:`numpy.ndarray`
        The scalar data from the cifti file
    brain_structures: :obj:`numpy.ndarray`
        The per-greyordinate brain structures as strings
    """
    cifti = cifti_file if hasattr(cifti_file, 'get_fdata') else nb.load(cifti_file)
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if len(axes) > 2:
        raise Exception('Only 2 axes should be present in a scalar cifti file')
    if len(axes) < 2:
        raise Exception()

    scalar_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ScalarAxis)]
    brain_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.BrainModelAxis)]

    if not len(scalar_axes) == 1:
        raise Exception(f'Only one scalar axis should be present. Found {scalar_axes}')
    if not len(brain_axes) == 1:
        raise Exception(f'Only one brain axis should be present. Found {brain_axes}')
    brain_axis = brain_axes.pop()

    cifti_data = cifti.get_fdata().squeeze().astype(np.float32)
    if not cifti_data.ndim == 1:
        raise Exception('Too many dimensions in the cifti data')
    brain_names = brain_axis.name
    if not cifti_data.shape[0] == brain_names.shape[0]:
        raise Exception('Mismatch between the brain names and data array')

    if reference_brain_names is not None:
        if not (brain_names == reference_brain_names).all():
            raise Exception(f'Incosistent vertex names in cifti file {cifti_file}')

    return cifti_data, brain_names


def brain_names_to_dataframe(brain_names):
    """Convert brain names to a dataframe.

    Parameters
    ----------
    brain_names : :obj:`numpy.ndarray`
        Array of brain names

    Returns
    -------
    greyordinate_df : :obj:`pandas.DataFrame`
        DataFrame with vertex_id and structure_id
    structure_name_strings : :obj:`list`
        List of structure names
    """
    # Make a lookup table for greyordinates
    structure_ids, structure_names = pd.factorize(brain_names)
    # Make them a list of strings
    structure_name_strings = list(map(str, structure_names))

    greyordinate_df = pd.DataFrame(
        {'vertex_id': np.arange(structure_ids.shape[0]), 'structure_id': structure_ids}
    )

    return greyordinate_df, structure_name_strings


def _load_cohort_cifti(cohort_long, relative_root, s3_workers):
    """Load all CIFTI scalar rows from the cohort, optionally in parallel.

    The first file is always loaded serially to obtain the reference brain
    structure axis used for validation. When s3_workers > 1, remaining rows
    are submitted to a ThreadPoolExecutor and collected via as_completed.
    Threads share memory so reference_brain_names is accessed directly with
    no copying overhead.

    Parameters
    ----------
    cohort_long : :obj:`pandas.DataFrame`
        Long-format cohort dataframe
    relative_root : :obj:`str`
        Root to which all paths are relative
    s3_workers : :obj:`int`
        Number of workers to use for parallel loading

    Returns
    -------
    scalars : :obj:`dict`
        Per-scalar ordered list of 1-D subject arrays, ready for stripe-write.
    reference_brain_names : :obj:`numpy.ndarray`
        Brain structure names from the first file, for building greyordinate table.
    """
    # Assign stable per-scalar subject indices in cohort order
    scalar_subj_counter = defaultdict(int)
    rows_with_idx = []
    for row in cohort_long.itertuples(index=False):
        subj_idx = scalar_subj_counter[row.scalar_name]
        scalar_subj_counter[row.scalar_name] += 1
        rows_with_idx.append((row.scalar_name, subj_idx, row.source_file))

    # Load the first file serially to get the reference brain axis
    first_sn, _, first_src = rows_with_idx[0]
    first_path = first_src if is_s3_path(first_src) else op.join(relative_root, first_src)
    first_data, reference_brain_names = extract_cifti_scalar_data(
        load_nibabel(first_path, cifti=True)
    )

    def _worker(job):
        sn, subj_idx, src = job
        arr, _ = extract_cifti_scalar_data(
            load_nibabel(src, cifti=True), reference_brain_names=reference_brain_names
        )
        return sn, subj_idx, arr

    if s3_workers > 1 and len(rows_with_idx) > 1:
        results = {first_sn: {0: first_data}}
        jobs = [
            (sn, subj_idx, src if is_s3_path(src) else op.join(relative_root, src))
            for sn, subj_idx, src in rows_with_idx[1:]
        ]
        with ThreadPoolExecutor(max_workers=s3_workers) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Loading CIFTI data',
            ):
                sn, subj_idx, arr = future.result()
                results.setdefault(sn, {})[subj_idx] = arr
        scalars = {
            sn: [results[sn][i] for i in range(cnt)] for sn, cnt in scalar_subj_counter.items()
        }
    else:
        scalars = defaultdict(list)
        scalars[first_sn].append(first_data)
        remaining = [
            (sn, subj_idx, src if is_s3_path(src) else op.join(relative_root, src))
            for sn, subj_idx, src in rows_with_idx[1:]
        ]
        for job in tqdm(remaining, desc='Loading CIFTI data'):
            sn, subj_idx, arr = _worker(job)
            scalars[sn].append(arr)

    return scalars, reference_brain_names


def write_storage(
    cohort_file,
    backend='hdf5',
    output_h5='fixeldb.h5',
    output_tdb='arraydb.tdb',
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
    output_h5 : :obj:`str`
        Path to a new .h5 file to be written
    output_tdb : :obj:`str`
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
    cohort_path = op.join(relative_root, cohort_file)
    cohort_df = pd.read_csv(cohort_path)
    cohort_long = _cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')
    scalar_sources = _build_scalar_sources(cohort_long)
    if not scalar_sources:
        raise ValueError('Unable to derive scalar sources from cohort file.')

    if backend == 'hdf5':
        scalars, last_brain_names = _load_cohort_cifti(cohort_long, relative_root, s3_workers)

        output_file = op.join(relative_root, output_h5)
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
            dset = h5_create_empty(
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

            h5_write_stripes(dset, scalars[scalar_name])
        f.close()
        return int(not op.exists(output_file))
    else:
        base_uri = op.join(relative_root, output_tdb)
        os.makedirs(base_uri, exist_ok=True)
        if not scalar_sources:
            return 0

        # Establish a reference brain axis once to ensure consistent ordering across workers.
        _first_scalar, first_sources = next(iter(scalar_sources.items()))
        first_path = op.join(relative_root, first_sources[0])
        _, reference_brain_names = extract_cifti_scalar_data(first_path)

        def _process_scalar_job(scalar_name, source_files):
            dataset_path = f'scalars/{scalar_name}/values'
            rows = []
            for source_file in source_files:
                scalar_file = op.join(relative_root, source_file)
                cifti_data, _ = extract_cifti_scalar_data(
                    scalar_file, reference_brain_names=reference_brain_names
                )
                rows.append(cifti_data)

            num_subjects = len(rows)
            if num_subjects == 0:
                return scalar_name
            num_items = rows[0].shape[0]
            tdb_create_empty(
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
            tdb_write_column_names(base_uri, scalar_name, source_files)
            uri = op.join(base_uri, dataset_path)
            tdb_write_stripes(uri, rows)
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
    import logging

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(name)s: %(message)s',
    )
    status = write_storage(
        cohort_file=args.cohort_file,
        backend=args.backend,
        output_h5=args.output_hdf5,
        output_tdb=args.output_tiledb,
        relative_root=args.relative_root,
        storage_dtype=args.dtype,
        compression=args.compression,
        compression_level=args.compression_level,
        shuffle=args.shuffle,
        chunk_voxels=args.chunk_voxels,
        target_chunk_mb=args.target_chunk_mb,
        tdb_compression=args.tdb_compression,
        tdb_compression_level=args.tdb_compression_level,
        tdb_shuffle=args.tdb_shuffle,
        tdb_tile_voxels=args.tdb_tile_voxels,
        tdb_target_tile_mb=args.tdb_target_tile_mb,
        tdb_workers=args.tdb_workers,
        scalar_columns=args.scalar_columns,
        s3_workers=args.s3_workers,
    )
    return status
