"""Unit tests for modelarrayio.utils.misc cohort and scalar-source helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from modelarrayio.utils.misc import (
    build_scalar_sources,
    cohort_to_long_dataframe,
    detect_modality_from_path,
    load_and_normalize_cohort,
)


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


# ===========================================================================
# detect_modality_from_path
# ===========================================================================


class TestDetectModalityFromPath:
    @pytest.mark.parametrize('ext', ['.dscalar.nii', '.pscalar.nii', '.pconn.nii'])
    def test_cifti_extensions(self, ext) -> None:
        assert detect_modality_from_path(f'sub-01{ext}') == 'cifti'

    def test_nii_gz_is_nifti(self) -> None:
        assert detect_modality_from_path('sub-01.nii.gz') == 'nifti'

    def test_nii_is_nifti(self) -> None:
        assert detect_modality_from_path('sub-01.nii') == 'nifti'

    def test_mif_is_mif(self) -> None:
        assert detect_modality_from_path('sub-01.mif') == 'mif'

    def test_mif_gz_is_mif(self) -> None:
        assert detect_modality_from_path('sub-01.mif.gz') == 'mif'

    def test_dscalar_nii_is_cifti_not_nifti(self) -> None:
        # .dscalar.nii ends with .nii — CIFTI must take priority
        assert detect_modality_from_path('sub-01.dscalar.nii') == 'cifti'

    def test_path_with_directory_prefix(self) -> None:
        assert detect_modality_from_path('/data/study/sub-01.dscalar.nii') == 'cifti'

    def test_s3_cifti_path(self) -> None:
        assert detect_modality_from_path('s3://bucket/prefix/sub-01.dscalar.nii') == 'cifti'

    def test_s3_nifti_path(self) -> None:
        assert detect_modality_from_path('s3://bucket/sub-01.nii.gz') == 'nifti'

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match='Cannot detect modality'):
            detect_modality_from_path('sub-01.mat')

    def test_no_extension_raises(self) -> None:
        with pytest.raises(ValueError, match='Cannot detect modality'):
            detect_modality_from_path('sub-01')

    def test_partial_cifti_ext_is_not_cifti(self) -> None:
        # .nii alone is NIfTI; only the compound extension qualifies as CIFTI
        assert detect_modality_from_path('sub-01.nii') == 'nifti'


# ===========================================================================
# load_and_normalize_cohort
# ===========================================================================


class TestLoadAndNormalizeCohort:
    def test_detects_cifti(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['THICK', 'THICK'], 'source_file': ['a.dscalar.nii', 'b.dscalar.nii']}
        ).to_csv(cohort, index=False)
        df, modality = load_and_normalize_cohort(cohort)
        assert modality == 'cifti'
        assert len(df) == 2

    def test_detects_nifti(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FA', 'FA'], 'source_file': ['a.nii.gz', 'b.nii.gz']}
        ).to_csv(cohort, index=False)
        _, modality = load_and_normalize_cohort(cohort)
        assert modality == 'nifti'

    def test_detects_mif(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FD', 'FD'], 'source_file': ['a.mif', 'b.mif']}
        ).to_csv(cohort, index=False)
        _, modality = load_and_normalize_cohort(cohort)
        assert modality == 'mif'

    def test_detects_mif_gz(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FD', 'FD'], 'source_file': ['a.mif.gz', 'b.mif.gz']}
        ).to_csv(cohort, index=False)
        _, modality = load_and_normalize_cohort(cohort)
        assert modality == 'mif'

    def test_wide_format_detects_modality(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort_wide.csv'
        pd.DataFrame(
            {'id': ['sub-01', 'sub-02'], 'FA': ['a.nii.gz', 'b.nii.gz']}
        ).to_csv(cohort, index=False)
        df, modality = load_and_normalize_cohort(cohort, scalar_columns=['FA'])
        assert modality == 'nifti'
        assert len(df) == 2

    def test_mixed_nifti_and_cifti_raises(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FA', 'THICK'], 'source_file': ['a.nii.gz', 'b.dscalar.nii']}
        ).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='mixed modalities'):
            load_and_normalize_cohort(cohort)

    def test_mixed_modality_error_names_both_types(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FA', 'THICK'], 'source_file': ['a.nii.gz', 'b.dscalar.nii']}
        ).to_csv(cohort, index=False)
        with pytest.raises(ValueError) as exc_info:
            load_and_normalize_cohort(cohort)
        message = str(exc_info.value)
        assert 'nifti' in message
        assert 'cifti' in message

    def test_mixed_mif_and_nifti_raises(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['FD', 'FA'], 'source_file': ['a.mif', 'b.nii.gz']}
        ).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='mixed modalities'):
            load_and_normalize_cohort(cohort)

    def test_empty_cohort_raises(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(columns=['scalar_name', 'source_file']).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='does not contain any scalar entries'):
            load_and_normalize_cohort(cohort)

    def test_missing_required_columns_raises(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame({'subject': ['sub-01']}).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='must contain columns'):
            load_and_normalize_cohort(cohort)

    def test_unknown_extension_raises(self, tmp_path) -> None:
        cohort = tmp_path / 'cohort.csv'
        pd.DataFrame(
            {'scalar_name': ['WEIRD'], 'source_file': ['sub-01.unknown_ext']}
        ).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='Cannot detect modality'):
            load_and_normalize_cohort(cohort)


# ===========================================================================
# build_scalar_sources
# ===========================================================================


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
