"""Utility functions for CIFTI data."""

import os.path as op
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nb
import numpy as np
import pandas as pd
from tqdm import tqdm

from modelarrayio.utils.s3_utils import is_s3_path, load_nibabel


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
