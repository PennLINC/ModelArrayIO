"""Tests for CIFTI cohort normalization and greyordinate helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelarrayio.utils.cifti import brain_names_to_dataframe
from modelarrayio.utils.misc import build_scalar_sources, cohort_to_long_dataframe


def test_cohort_long_format_preserves_rows() -> None:
    df = pd.DataFrame(
        {
            'scalar_name': ['THICK', 'THICK'],
            'source_file': ['sub-1.dscalar.nii', 'sub-2.dscalar.nii'],
            'extra_col': [1, 2],
        }
    )
    long_df = cohort_to_long_dataframe(df)
    assert len(long_df) == 2
    assert set(long_df.columns) == {'extra_col', 'scalar_name', 'source_file'}
    assert long_df.iloc[0]['scalar_name'] == 'THICK'


def test_cohort_long_format_strips_and_drops_empty() -> None:
    df = pd.DataFrame(
        {
            'scalar_name': ['  THICK  ', ''],
            'source_file': ['  a.nii  ', '  b.nii  '],
        }
    )
    long_df = cohort_to_long_dataframe(df)
    assert len(long_df) == 1
    assert long_df.iloc[0]['scalar_name'] == 'THICK'


def test_cohort_wide_format_expands_columns() -> None:
    df = pd.DataFrame(
        {
            'id': [1, 2],
            'THICK': ['s1.nii', 's2.nii'],
            'FA': ['f1.nii', ''],
        }
    )
    long_df = cohort_to_long_dataframe(df, scalar_columns=['THICK', 'FA'])
    # Row 2 has empty FA — skipped
    assert len(long_df) == 3
    scalars = set(long_df['scalar_name'])
    assert scalars == {'THICK', 'FA'}


def test_cohort_wide_format_missing_scalar_column_raises() -> None:
    df = pd.DataFrame({'THICK': ['a.nii']})
    with pytest.raises(ValueError, match='missing scalar columns'):
        cohort_to_long_dataframe(df, scalar_columns=['THICK', 'MISSING'])


def test_cohort_long_missing_required_raises() -> None:
    df = pd.DataFrame({'only_this': [1]})
    with pytest.raises(ValueError, match='scalar_name'):
        cohort_to_long_dataframe(df)


def test_build_scalar_sources_ordering() -> None:
    long_df = pd.DataFrame(
        {
            'scalar_name': ['A', 'A', 'B'],
            'source_file': ['x1', 'x2', 'y1'],
        }
    )
    src = build_scalar_sources(long_df)
    assert list(src.keys()) == ['A', 'B']
    assert src['A'] == ['x1', 'x2']
    assert src['B'] == ['y1']


def test_brain_names_to_dataframe() -> None:
    names = np.array(['CORTEX_LEFT', 'CORTEX_LEFT', 'CORTEX_RIGHT'])
    gdf, struct_strings = brain_names_to_dataframe(names)
    assert len(gdf) == 3
    assert 'vertex_id' in gdf.columns
    assert 'structure_id' in gdf.columns
    assert gdf['vertex_id'].tolist() == [0, 1, 2]
    assert len(struct_strings) == 2  # factorize unique structures
