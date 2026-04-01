"""Utility functions for CIFTI data."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from tqdm import tqdm

from modelarrayio.utils.s3_utils import load_nibabel


def _cohort_to_long_dataframe(cohort_df, scalar_columns=None):
    scalar_columns = [col for col in (scalar_columns or []) if col]
    if scalar_columns:
        missing = [col for col in scalar_columns if col not in cohort_df.columns]
        if missing:
            raise ValueError(f'Wide-format cohort is missing scalar columns: {missing}')
        records = []
        selected_columns = cohort_df[scalar_columns]
        for row_values in selected_columns.itertuples(index=False, name=None):
            for scalar_col, source_val in zip(scalar_columns, row_values, strict=True):
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
            f'Cohort file must contain columns {sorted(required)} when '
            '--scalar-columns is not used.'
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
    """Load a scalar cifti file and get its data and mapping.

    Supports dscalar (.dscalar.nii), parcellated scalar (.pscalar.nii), and
    parcellated connectivity (.pconn.nii) files. For pconn files the 2-D
    connectivity matrix is flattened to a 1-D array (row-major order) and
    ``brain_structures`` contains the row parcel name repeated once per
    column, so callers receive a consistent 1-D array of element names.

    Parameters
    ----------
    cifti_file : :obj:`str` or :obj:`pathlib.Path` or :obj:`nibabel.Cifti2Image`
        CIFTI2 file on disk or already loaded CIFTI image.
    reference_brain_names : :obj:`numpy.ndarray`
        Array of vertex/parcel names used for cross-file consistency checks.
        For pconn files this must be the ``brain_structures`` value returned
        by a previous call to this function.

    Returns
    -------
    cifti_scalar_data: :obj:`numpy.ndarray`
        The scalar data from the cifti file, always 1-D.  For pconn files
        this is the row-major flattened connectivity matrix.
    brain_structures: :obj:`numpy.ndarray`
        Per-element spatial labels as strings.  For dscalar files these are
        the per-greyordinate brain structure names.  For pscalar files these
        are the parcel names.  For pconn files these are the row-parcel names
        repeated ``n_col_parcels`` times (one entry per flattened element).
    """
    cifti = cifti_file if hasattr(cifti_file, 'get_fdata') else nb.load(Path(cifti_file))
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    if len(axes) != 2:
        raise ValueError(
            f'Expected exactly 2 axes in scalar CIFTI data, found {len(axes)} for {cifti_file!r}.'
        )

    scalar_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ScalarAxis)]
    brain_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.BrainModelAxis)]
    parcel_axes = [ax for ax in axes if isinstance(ax, nb.cifti2.cifti2_axes.ParcelsAxis)]

    if len(scalar_axes) == 1 and len(brain_axes) == 1:
        # dscalar.nii: ScalarAxis (axis 0) + BrainModelAxis (axis 1)
        brain_axis = brain_axes[0]
        cifti_data = cifti.get_fdata().squeeze().astype(np.float32)
        if cifti_data.ndim != 1:
            raise ValueError(f'Expected 1-D scalar CIFTI data in {cifti_file!r}.')
        brain_names = brain_axis.name
        if cifti_data.shape[0] != brain_names.shape[0]:
            raise ValueError(
                f'Mismatch between brain names and data array in {cifti_file!r}: '
                f'{brain_names.shape[0]} names vs {cifti_data.shape[0]} values.'
            )
        if reference_brain_names is not None:
            if not np.array_equal(brain_names, reference_brain_names):
                raise ValueError(f'Inconsistent greyordinate names in CIFTI file {cifti_file!r}.')
        return cifti_data, brain_names

    elif len(scalar_axes) == 1 and len(parcel_axes) == 1:
        # pscalar.nii: ScalarAxis (axis 0) + ParcelsAxis (axis 1)
        parcel_axis = parcel_axes[0]
        cifti_data = cifti.get_fdata().squeeze().astype(np.float32)
        if cifti_data.ndim != 1:
            raise ValueError(f'Expected 1-D parcellated scalar CIFTI data in {cifti_file!r}.')
        parcel_names = parcel_axis.name
        if cifti_data.shape[0] != parcel_names.shape[0]:
            raise ValueError(
                f'Mismatch between parcel names and data array in {cifti_file!r}: '
                f'{parcel_names.shape[0]} names vs {cifti_data.shape[0]} values.'
            )
        if reference_brain_names is not None:
            if not np.array_equal(parcel_names, reference_brain_names):
                raise ValueError(f'Inconsistent parcel names in CIFTI file {cifti_file!r}.')
        return cifti_data, parcel_names

    elif len(parcel_axes) == 2:
        # pconn.nii: ParcelsAxis (axis 0, rows) + ParcelsAxis (axis 1, cols)
        row_axis = parcel_axes[0]
        cifti_data = cifti.get_fdata().astype(np.float32)
        if cifti_data.ndim != 2:
            raise ValueError(
                f'Expected 2-D parcellated connectivity CIFTI data in {cifti_file!r}.'
            )
        n_col = cifti_data.shape[1]
        row_parcel_names = row_axis.name
        # One element name per flattened entry: repeat each row parcel name n_col times
        # XXX: For the pconn (ParcelsAxis+ParcelsAxis) case, element_names are built only from the
        # row parcel names (repeated by n_col). This ignores the column-axis parcel names entirely,
        # so two pconn files with different column axes could incorrectly pass
        # reference_brain_names validation. Also, there's no explicit check that
        # len(row_axis.name)==n_rows and len(col_axis.name)==n_cols. Consider validating both axes
        # against the data shape and incorporating both row+col parcel names into the per-element
        # labels (or at least validating the column axis matches across files).
        # TODO: Add elements_from and elements_to fields to converted H5/TileDB files.
        element_names = np.repeat(row_parcel_names, n_col)
        cifti_data_flat = cifti_data.flatten()
        if reference_brain_names is not None:
            if not np.array_equal(element_names, reference_brain_names):
                raise ValueError(f'Inconsistent parcel names in CIFTI file {cifti_file!r}.')
        return cifti_data_flat, element_names

    else:
        raise ValueError(
            f'Unsupported CIFTI axis combination in {cifti_file!r}. '
            'Supported types: dscalar (ScalarAxis+BrainModelAxis), '
            'pscalar (ScalarAxis+ParcelsAxis), pconn (ParcelsAxis+ParcelsAxis).'
        )


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


def _load_cohort_cifti(cohort_long, s3_workers):
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
    first_data, reference_brain_names = extract_cifti_scalar_data(
        load_nibabel(first_src, cifti=True)
    )

    def _worker(job):
        sn, subj_idx, src = job
        arr, _ = extract_cifti_scalar_data(
            load_nibabel(src, cifti=True), reference_brain_names=reference_brain_names
        )
        return sn, subj_idx, arr

    if s3_workers > 1 and len(rows_with_idx) > 1:
        results = {first_sn: {0: first_data}}
        jobs = [(sn, subj_idx, src) for sn, subj_idx, src in rows_with_idx[1:]]
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
        remaining = [(sn, subj_idx, src) for sn, subj_idx, src in rows_with_idx[1:]]
        for job in tqdm(remaining, desc='Loading CIFTI data'):
            sn, subj_idx, arr = _worker(job)
            scalars[sn].append(arr)

    return scalars, reference_brain_names
