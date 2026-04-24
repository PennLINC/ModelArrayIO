"""Miscellaneous utility functions."""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd

_CIFTI_EXTENSIONS = (
    '.dscalar.nii',
    '.pscalar.nii',
    '.pconn.nii',
)


def detect_modality_from_path(path: str) -> str:
    """Return ``'cifti'``, ``'mif'``, or ``'nifti'`` based on file extension.

    Parameters
    ----------
    path : str
        File path whose extension is used to identify the neuroimaging modality.

    Returns
    -------
    str
        One of ``'cifti'``, ``'mif'``, or ``'nifti'``.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    path = str(path)
    if any(path.endswith(ext) for ext in _CIFTI_EXTENSIONS):
        return 'cifti'
    if path.endswith(('.mif.gz', '.mif')):
        return 'mif'
    if path.endswith(('.nii.gz', '.nii')):
        return 'nifti'
    raise ValueError(
        f'Cannot detect modality from file extension: {path!r}. '
        'Expected .mif, .nii, .nii.gz, or a CIFTI compound extension '
        '(e.g. .dscalar.nii, .pscalar.nii).'
    )


def cohort_to_long_dataframe(cohort_df, scalar_columns=None):
    """Convert a wide-format cohort dataframe to a long-format dataframe.

    Parameters
    ----------
    cohort_df : :obj:`pandas.DataFrame`
        Wide-format cohort dataframe
    scalar_columns : :obj:`list`
        List of scalar columns to use. If provided, these columns are treated as
        file-path columns and melted into 'scalar_name'/'source_file' rows. All
        remaining columns (e.g. 'source_mask_file') are broadcast to every output
        row. If not provided, the dataframe is treated as already long-format.

    Returns
    -------
    long_df : :obj:`pandas.DataFrame`
        Long-format cohort dataframe with columns 'scalar_name', 'source_file',
        and any non-scalar columns from the input.
    """
    scalar_columns = [col for col in (scalar_columns or []) if col]
    if scalar_columns:
        missing = [col for col in scalar_columns if col not in cohort_df.columns]
        if missing:
            raise ValueError(f'Wide-format cohort is missing scalar columns: {missing}')
        extra_columns = [col for col in cohort_df.columns if col not in scalar_columns]
        records = []
        for _, row in cohort_df.iterrows():
            extra = {col: row[col] for col in extra_columns}
            for scalar_col in scalar_columns:
                source_val = row[scalar_col]
                if pd.isna(source_val) or source_val is None:
                    continue
                source_str = str(source_val).strip()
                if not source_str:
                    continue
                records.append({'scalar_name': scalar_col, 'source_file': source_str, **extra})
        output_columns = ['scalar_name', 'source_file'] + extra_columns
        return pd.DataFrame.from_records(records, columns=output_columns)

    required = {'scalar_name', 'source_file'}
    missing = required - set(cohort_df.columns)
    if missing:
        raise ValueError(
            f'Cohort file must contain columns {sorted(required)} when '
            '--scalar-columns is not used.'
        )

    long_df = cohort_df.copy()
    long_df = long_df.dropna(subset=['scalar_name', 'source_file'])
    long_df['scalar_name'] = long_df['scalar_name'].astype(str).str.strip()
    long_df['source_file'] = long_df['source_file'].astype(str).str.strip()
    long_df = long_df[(long_df['scalar_name'] != '') & (long_df['source_file'] != '')]
    return long_df.reset_index(drop=True)


def load_and_normalize_cohort(
    cohort_file,
    scalar_columns=None,
) -> tuple[pd.DataFrame, str]:
    """Load a cohort CSV, normalise it, and detect the neuroimaging modality.

    This is the single entry-point for cohort ingestion shared by all
    ``*_to_h5`` converters. It performs, in order:

    1. ``pd.read_csv`` the file.
    2. ``cohort_to_long_dataframe`` to normalise wide/long format.
    3. Empty-cohort validation.
    4. Modality detection from every unique ``source_file`` extension.
    5. Mixed-modality validation (all rows must be the same modality).

    Parameters
    ----------
    cohort_file : path-like
        Path to the cohort CSV file.
    scalar_columns : list of str, optional
        Column names for wide-format cohort files.  If omitted the CSV must
        already contain ``scalar_name`` and ``source_file`` columns.

    Returns
    -------
    cohort_long : pandas.DataFrame
        Normalised long-format cohort dataframe.
    modality : str
        Detected modality: ``'nifti'``, ``'mif'``, or ``'cifti'``.

    Raises
    ------
    ValueError
        If the cohort is empty after normalisation, if a source file has an
        unrecognised extension, or if the cohort contains mixed modalities.
    """
    cohort_df = pd.read_csv(cohort_file)
    cohort_long = cohort_to_long_dataframe(cohort_df, scalar_columns=scalar_columns)
    if cohort_long.empty:
        raise ValueError('Cohort file does not contain any scalar entries after normalization.')

    modalities = {
        detect_modality_from_path(str(path)) for path in cohort_long['source_file'].unique()
    }
    if len(modalities) > 1:
        raise ValueError(
            f'Cohort file contains mixed modalities ({", ".join(sorted(modalities))}). '
            'All source_file entries must be the same type (NIfTI, CIFTI, or MIF).'
        )
    modality = modalities.pop()
    return cohort_long, modality


def build_scalar_sources(long_df):
    """Build a dictionary of scalar sources from a long dataframe.

    Parameters
    ----------
    long_df : :obj:`pandas.DataFrame`
        Long-format cohort dataframe with columns 'scalar_name' and 'source_file'.

    Returns
    -------
    scalar_sources : :obj:`OrderedDict`
        Dictionary of scalar sources.
        Keys are scalar names, values are lists of source files.
    """
    scalar_sources = OrderedDict()
    for row in long_df.itertuples(index=False):
        scalar = str(row.scalar_name)
        source = str(row.source_file)
        if not scalar or not source:
            continue
        scalar_sources.setdefault(scalar, []).append(source)
    return scalar_sources
