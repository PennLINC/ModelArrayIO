"""Tests for modality detection and routing in to_modelarray and export_results.

Covers:
- Correct dispatch to the right downstream function for each modality
- All user-error paths: missing required flags, wrong file type for the flags
  supplied, mixed modalities, unknown extensions
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import modelarrayio.cli.export_results as _export_results_mod
import modelarrayio.cli.to_modelarray as _to_modelarray_mod
from modelarrayio.cli.export_results import export_results
from modelarrayio.cli.to_modelarray import to_modelarray


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _write_cohort(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a long-format cohort CSV and return its path."""
    fieldnames = list(rows[0].keys())
    cohort = tmp_path / 'cohort.csv'
    with cohort.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return cohort


def _minimal_h5(tmp_path: Path, analysis: str = 'lm') -> Path:
    """Write a minimal results HDF5 file and return its path."""
    path = tmp_path / 'results.h5'
    with h5py.File(path, 'w') as h5:
        grp = h5.require_group(f'results/{analysis}')
        ds = grp.create_dataset('results_matrix', data=np.zeros((1, 3), dtype=np.float32))
        ds.attrs['colnames'] = [b'effect']
    return path


# ===========================================================================
# to_modelarray: standard routing
# ===========================================================================


class TestToModelarrayRouting:
    """to_modelarray detects the modality from cohort file extensions and dispatches
    to the correct downstream converter."""

    def test_dscalar_cohort_calls_cifti_to_h5(self, tmp_path, monkeypatch):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'THICK', 'source_file': 'sub-01.dscalar.nii'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'cifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5')
        mock.assert_called_once()

    def test_pscalar_cohort_calls_cifti_to_h5(self, tmp_path, monkeypatch):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'MYELIN', 'source_file': 'sub-01.pscalar.nii'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'cifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5')
        mock.assert_called_once()

    def test_pconn_cohort_calls_cifti_to_h5(self, tmp_path, monkeypatch):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FC', 'source_file': 'sub-01.pconn.nii'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'cifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5')
        mock.assert_called_once()

    def test_nifti_gz_cohort_with_mask_calls_nifti_to_h5(self, tmp_path, monkeypatch):
        mask = tmp_path / 'mask.nii.gz'
        mask.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': 'sub-01.nii.gz'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'nifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5', group_mask_file=mask)
        mock.assert_called_once()

    def test_uncompressed_nifti_cohort_with_mask_calls_nifti_to_h5(self, tmp_path, monkeypatch):
        mask = tmp_path / 'mask.nii.gz'
        mask.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': 'sub-01.nii'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'nifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5', group_mask_file=mask)
        mock.assert_called_once()

    def test_mif_cohort_with_both_mif_args_calls_mif_to_h5(self, tmp_path, monkeypatch):
        index = tmp_path / 'index.mif'
        directions = tmp_path / 'directions.mif'
        index.touch()
        directions.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': 'sub-01.mif'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'mif_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5', index_file=index, directions_file=directions)
        mock.assert_called_once()

    def test_mif_gz_cohort_with_both_mif_args_calls_mif_to_h5(self, tmp_path, monkeypatch):
        index = tmp_path / 'index.mif'
        directions = tmp_path / 'directions.mif'
        index.touch()
        directions.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': 'sub-01.mif.gz'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'mif_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5', index_file=index, directions_file=directions)
        mock.assert_called_once()

    def test_cohort_long_dataframe_is_passed_not_file_path(self, tmp_path, monkeypatch):
        """The downstream converter receives a pre-normalised DataFrame, not the CSV path."""
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'THICK', 'source_file': 'sub-01.dscalar.nii'}]
        )
        mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'cifti_to_h5', mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5')
        _, kwargs = mock.call_args
        assert 'cohort_long' in kwargs
        assert hasattr(kwargs['cohort_long'], 'itertuples'), 'cohort_long must be a DataFrame'


# ===========================================================================
# to_modelarray: user errors
# ===========================================================================


class TestToModelarrayUserErrors:
    """Errors raised when required flags are missing for the detected modality."""

    def test_nifti_cohort_without_mask_raises(self, tmp_path):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': 'sub-01.nii.gz'}]
        )
        with pytest.raises(ValueError, match='--mask'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_nifti_cohort_without_mask_error_names_nifti(self, tmp_path):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': 'sub-01.nii.gz'}]
        )
        with pytest.raises(ValueError, match='Detected NIfTI'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_mif_cohort_without_index_file_raises(self, tmp_path):
        directions = tmp_path / 'directions.mif'
        directions.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': 'sub-01.mif'}]
        )
        with pytest.raises(ValueError, match='--index-file'):
            to_modelarray(cohort, output=tmp_path / 'out.h5', directions_file=directions)

    def test_mif_cohort_without_directions_file_raises(self, tmp_path):
        index = tmp_path / 'index.mif'
        index.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': 'sub-01.mif'}]
        )
        with pytest.raises(ValueError, match='--directions-file'):
            to_modelarray(cohort, output=tmp_path / 'out.h5', index_file=index)

    def test_mif_cohort_without_any_mif_args_raises(self, tmp_path):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': 'sub-01.mif'}]
        )
        with pytest.raises(ValueError, match='MIF'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_mixed_nifti_and_cifti_cohort_raises(self, tmp_path):
        cohort = _write_cohort(
            tmp_path,
            [
                {'scalar_name': 'FA', 'source_file': 'sub-01.nii.gz'},
                {'scalar_name': 'THICK', 'source_file': 'sub-01.dscalar.nii'},
            ],
        )
        with pytest.raises(ValueError, match='mixed modalities'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_mixed_mif_and_nifti_cohort_raises(self, tmp_path):
        cohort = _write_cohort(
            tmp_path,
            [
                {'scalar_name': 'FD', 'source_file': 'sub-01.mif'},
                {'scalar_name': 'FA', 'source_file': 'sub-02.nii.gz'},
            ],
        )
        with pytest.raises(ValueError, match='mixed modalities'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_unknown_extension_in_cohort_raises(self, tmp_path):
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'WEIRD', 'source_file': 'sub-01.unknown'}]
        )
        with pytest.raises(ValueError, match='Cannot detect modality'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_empty_cohort_raises(self, tmp_path):
        cohort = tmp_path / 'empty.csv'
        pd.DataFrame(columns=['scalar_name', 'source_file']).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='does not contain any scalar entries'):
            to_modelarray(cohort, output=tmp_path / 'out.h5')

    def test_cifti_cohort_with_mask_ignores_nifti_flag_and_routes_to_cifti(
        self, tmp_path, monkeypatch
    ):
        """Providing --mask with a CIFTI cohort is a user mistake: the detected modality
        (from file extensions) is CIFTI, so the mask is unused and cifti_to_h5 is called."""
        mask = tmp_path / 'mask.nii.gz'
        mask.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'THICK', 'source_file': 'sub-01.dscalar.nii'}]
        )
        cifti_mock = MagicMock(return_value=0)
        nifti_mock = MagicMock(return_value=0)
        monkeypatch.setattr(_to_modelarray_mod, 'cifti_to_h5', cifti_mock)
        monkeypatch.setattr(_to_modelarray_mod, 'nifti_to_h5', nifti_mock)
        to_modelarray(cohort, output=tmp_path / 'out.h5', group_mask_file=mask)
        cifti_mock.assert_called_once()
        nifti_mock.assert_not_called()


# ===========================================================================
# export_results: standard routing
# ===========================================================================


class TestExportResultsRouting:
    """export_results infers modality from flags and dispatches to the correct exporter."""

    def test_mask_routes_to_h5_to_nifti(self, tmp_path, monkeypatch):
        h5 = _minimal_h5(tmp_path)
        mask = tmp_path / 'mask.nii.gz'
        mask.touch()
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_nifti', mock)
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            group_mask_file=mask,
        )
        assert result == 0
        mock.assert_called_once()

    def test_index_and_directions_with_example_routes_to_h5_to_mif(self, tmp_path, monkeypatch):
        h5 = _minimal_h5(tmp_path)
        index = tmp_path / 'index.mif'
        directions = tmp_path / 'directions.mif'
        example = tmp_path / 'sub-01.mif'
        index.touch()
        directions.touch()
        example.touch()
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_mif', mock)
        monkeypatch.setattr(_export_results_mod.shutil, 'copyfile', MagicMock())
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            index_file=index, directions_file=directions, example_file=example,
        )
        assert result == 0
        mock.assert_called_once()

    def test_index_and_directions_with_cohort_routes_to_h5_to_mif(self, tmp_path, monkeypatch):
        """Using --cohort-file instead of --example-file for MIF template."""
        h5 = _minimal_h5(tmp_path)
        index = tmp_path / 'index.mif'
        directions = tmp_path / 'directions.mif'
        index.touch()
        directions.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': str(tmp_path / 'sub-01.mif')}]
        )
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_mif', mock)
        monkeypatch.setattr(_export_results_mod.shutil, 'copyfile', MagicMock())
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            index_file=index, directions_file=directions, cohort_file=cohort,
        )
        assert result == 0
        mock.assert_called_once()

    def test_dscalar_example_file_routes_to_h5_to_cifti(self, tmp_path, monkeypatch):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.dscalar.nii'
        example.touch()
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_cifti', mock)
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            example_file=example,
        )
        assert result == 0
        mock.assert_called_once()

    def test_pscalar_example_file_routes_to_h5_to_cifti(self, tmp_path, monkeypatch):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.pscalar.nii'
        example.touch()
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_cifti', mock)
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            example_file=example,
        )
        assert result == 0
        mock.assert_called_once()

    def test_pconn_example_file_routes_to_h5_to_cifti(self, tmp_path, monkeypatch):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.pconn.nii'
        example.touch()
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_cifti', mock)
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            example_file=example,
        )
        assert result == 0
        mock.assert_called_once()

    def test_cifti_cohort_file_routes_to_h5_to_cifti(self, tmp_path, monkeypatch):
        """--cohort-file with CIFTI source paths routes to CIFTI export."""
        h5 = _minimal_h5(tmp_path)
        cohort = _write_cohort(
            tmp_path,
            [{'scalar_name': 'THICK', 'source_file': str(tmp_path / 'sub-01.dscalar.nii')}],
        )
        mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_cifti', mock)
        result = export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            cohort_file=cohort,
        )
        assert result == 0
        mock.assert_called_once()

    def test_mask_takes_priority_over_cohort_file(self, tmp_path, monkeypatch):
        """When --mask is given alongside --cohort-file, NIfTI wins (mask is the
        unambiguous modality indicator for export_results)."""
        h5 = _minimal_h5(tmp_path)
        mask = tmp_path / 'mask.nii.gz'
        mask.touch()
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': str(tmp_path / 'sub-01.nii.gz')}]
        )
        nifti_mock = MagicMock(return_value=None)
        cifti_mock = MagicMock(return_value=None)
        monkeypatch.setattr(_export_results_mod, 'h5_to_nifti', nifti_mock)
        monkeypatch.setattr(_export_results_mod, 'h5_to_cifti', cifti_mock)
        export_results(
            in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
            group_mask_file=mask, cohort_file=cohort,
        )
        nifti_mock.assert_called_once()
        cifti_mock.assert_not_called()


# ===========================================================================
# export_results: user errors
# ===========================================================================


class TestExportResultsUserErrors:
    """Errors raised for bad or missing argument combinations."""

    def test_no_modality_args_raises(self, tmp_path):
        h5 = _minimal_h5(tmp_path)
        with pytest.raises(ValueError, match='Cannot determine modality'):
            export_results(in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out')

    # -- NIfTI example/cohort without --mask --

    def test_nii_gz_example_without_mask_raises(self, tmp_path):
        """Classic mistake: user passes the subject NIfTI as --example-file but forgets --mask."""
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.nii.gz'
        example.touch()
        with pytest.raises(ValueError, match='appears to be NIfTI'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_nii_gz_example_error_mentions_mask_flag(self, tmp_path):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.nii.gz'
        example.touch()
        with pytest.raises(ValueError, match='--mask'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_uncompressed_nii_example_without_mask_raises(self, tmp_path):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.nii'
        example.touch()
        with pytest.raises(ValueError, match='appears to be NIfTI'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_nifti_cohort_without_mask_raises(self, tmp_path):
        """User passes --cohort-file whose source_files are .nii.gz but omits --mask."""
        h5 = _minimal_h5(tmp_path)
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FA', 'source_file': str(tmp_path / 'sub-01.nii.gz')}]
        )
        with pytest.raises(ValueError, match='appears to be NIfTI'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                cohort_file=cohort,
            )

    # -- MIF example/cohort without --index-file / --directions-file --

    def test_mif_example_without_index_dirs_raises(self, tmp_path):
        """User passes a MIF file as --example-file but forgets the MIF flags."""
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.mif'
        example.touch()
        with pytest.raises(ValueError, match='appears to be MIF'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_mif_example_error_mentions_index_and_directions_flags(self, tmp_path):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.mif'
        example.touch()
        with pytest.raises(ValueError, match='--index-file'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_mif_gz_example_without_index_dirs_raises(self, tmp_path):
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.mif.gz'
        example.touch()
        with pytest.raises(ValueError, match='appears to be MIF'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )

    def test_mif_cohort_without_index_dirs_raises(self, tmp_path):
        """User passes --cohort-file whose source_files are .mif but omits the MIF flags."""
        h5 = _minimal_h5(tmp_path)
        cohort = _write_cohort(
            tmp_path, [{'scalar_name': 'FD', 'source_file': str(tmp_path / 'sub-01.mif')}]
        )
        with pytest.raises(ValueError, match='appears to be MIF'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                cohort_file=cohort,
            )

    # -- Incomplete MIF flag pairs --

    def test_only_index_file_raises_missing_directions(self, tmp_path):
        """User provides --index-file but forgets --directions-file."""
        h5 = _minimal_h5(tmp_path)
        index = tmp_path / 'index.mif'
        index.touch()
        with pytest.raises(ValueError, match='Both --index-file and --directions-file'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                index_file=index,
            )

    def test_only_directions_file_raises_missing_index(self, tmp_path):
        """User provides --directions-file but forgets --index-file."""
        h5 = _minimal_h5(tmp_path)
        directions = tmp_path / 'directions.mif'
        directions.touch()
        with pytest.raises(ValueError, match='Both --index-file and --directions-file'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                directions_file=directions,
            )

    def test_both_mif_files_but_no_template_raises(self, tmp_path):
        """User provides --index-file and --directions-file but no cohort or example."""
        h5 = _minimal_h5(tmp_path)
        index = tmp_path / 'index.mif'
        directions = tmp_path / 'directions.mif'
        index.touch()
        directions.touch()
        with pytest.raises(ValueError, match='--cohort-file or --example-file'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                index_file=index, directions_file=directions,
            )

    # -- Completely unrecognised file type --

    def test_unknown_extension_in_example_file_raises(self, tmp_path):
        """An unrecognised file extension raises a clear error rather than silently
        routing to CIFTI or producing a cryptic downstream failure."""
        h5 = _minimal_h5(tmp_path)
        example = tmp_path / 'sub-01.mat'
        example.touch()
        with pytest.raises(ValueError, match='Cannot detect modality'):
            export_results(
                in_file=h5, analysis_name='lm', output_dir=tmp_path / 'out',
                example_file=example,
            )
