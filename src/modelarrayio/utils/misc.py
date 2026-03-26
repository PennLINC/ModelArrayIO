"""Miscellaneous utility functions."""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd


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
