"""Tests for CIFTI conversion functions and the to-modelarray CLI command.

Covers dscalar, pscalar, and pconn CIFTI types for both conversion directions,
and includes end-to-end tests via the top-level modelarrayio CLI entry point.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from utils import make_dscalar, make_parcels_axis, make_pconn, make_pscalar  # noqa: F401

from modelarrayio.cli.cifti_to_h5 import cifti_to_h5
from modelarrayio.cli.h5_to_cifti import _cifti_output_ext, h5_to_cifti
from modelarrayio.cli.main import main as modelarrayio_main
from modelarrayio.utils.cifti import _get_cifti_parcel_info

DATA_DIR = Path(__file__).parent / 'data_cifti_toy'
EXAMPLE_DSCALAR = DATA_DIR / 'example.dscalar.nii'
EXAMPLE_PCONN = DATA_DIR / 'example.pconn.nii'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_cohort_csv(path: Path, rows: list[dict]) -> None:
    """Write a simple long-format cohort CSV with scalar_name and source_file columns."""
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scalar_name', 'source_file'])
        writer.writeheader()
        writer.writerows(rows)


def _make_h5_results(
    h5_path: Path,
    analysis_name: str,
    results_matrix: np.ndarray,
    result_names: list[str],
) -> None:
    """Write a minimal HDF5 results file consumable by h5_to_cifti."""
    with h5py.File(h5_path, 'w') as h5:
        grp = h5.require_group(f'results/{analysis_name}')
        ds = grp.create_dataset('results_matrix', data=results_matrix)
        ds.attrs['colnames'] = [n.encode('utf-8') for n in result_names]


# ---------------------------------------------------------------------------
# Subject-batch writers
# ---------------------------------------------------------------------------

_VOL_SHAPE = (3, 3, 3)
_MASK_COORDS = [(0, 0, 0), (0, 1, 2), (1, 1, 1), (2, 2, 0), (2, 1, 2)]


def _dscalar_mask() -> np.ndarray:
    mask = np.zeros(_VOL_SHAPE, dtype=bool)
    for ijk in _MASK_COORDS:
        mask[ijk] = True
    return mask


def _write_dscalar_subjects(
    tmp_path: Path,
    mask: np.ndarray,
    n_subjects: int = 2,
    scalar_name: str = 'THICK',
) -> list[Path]:
    n_go = int(mask.sum())
    paths = []
    for i in range(n_subjects):
        vals = np.arange(n_go, dtype=np.float32) * (i + 1)
        img = make_dscalar(mask, vals)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.dscalar.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


_PSCALAR_PARCELS = ['parcel_A', 'parcel_B', 'parcel_C', 'parcel_D']


def _write_pscalar_subjects(
    tmp_path: Path,
    parcel_names: list[str],
    n_subjects: int = 2,
    scalar_name: str = 'MYELIN',
) -> list[Path]:
    n = len(parcel_names)
    paths = []
    for i in range(n_subjects):
        vals = np.arange(n, dtype=np.float32) + i * 10
        img = make_pscalar(parcel_names, vals)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.pscalar.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


_PCONN_PARCELS = ['net_A', 'net_B', 'net_C']


def _write_pconn_subjects(
    tmp_path: Path,
    parcel_names: list[str],
    n_subjects: int = 2,
    scalar_name: str = 'FC',
) -> list[Path]:
    n = len(parcel_names)
    paths = []
    for i in range(n_subjects):
        matrix = (np.eye(n, dtype=np.float32) * (i + 1)).astype(np.float32)
        img = make_pconn(parcel_names, matrix)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.pconn.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


# ===========================================================================
# _get_cifti_parcel_info
# ===========================================================================


class TestGetCiftiElementInfo:
    def test_dscalar_returns_empty_arrays(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        img = make_dscalar(mask, np.array([1.0], dtype=np.float32))
        cifti_type, element_arrays = _get_cifti_parcel_info(img)
        assert cifti_type == 'dscalar'
        assert element_arrays == {}

    def test_pscalar_returns_parcel_id(self):
        parcel_names = ['A', 'B', 'C']
        img = make_pscalar(parcel_names, np.zeros(3, dtype=np.float32))
        cifti_type, element_arrays = _get_cifti_parcel_info(img)
        assert cifti_type == 'pscalar'
        assert list(element_arrays.keys()) == ['parcel_id']
        assert list(element_arrays['parcel_id']) == parcel_names

    def test_pconn_returns_parcel_id_from_and_to(self):
        parcel_names = ['X', 'Y']
        img = make_pconn(parcel_names, np.eye(2, dtype=np.float32))
        cifti_type, element_arrays = _get_cifti_parcel_info(img)
        assert cifti_type == 'pconn'
        assert set(element_arrays.keys()) == {'parcel_id_from', 'parcel_id_to'}
        assert list(element_arrays['parcel_id_from']) == parcel_names
        assert list(element_arrays['parcel_id_to']) == parcel_names

    def test_dscalar_from_file(self, tmp_path):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        p = tmp_path / 'img.dscalar.nii'
        make_dscalar(mask, np.array([1.0], dtype=np.float32)).to_filename(p)
        cifti_type, element_arrays = _get_cifti_parcel_info(str(p))
        assert cifti_type == 'dscalar'
        assert element_arrays == {}

    def test_pscalar_from_file(self, tmp_path):
        parcel_names = ['parcel_A', 'parcel_B']
        p = tmp_path / 'img.pscalar.nii'
        make_pscalar(parcel_names, np.zeros(2, dtype=np.float32)).to_filename(p)
        cifti_type, element_arrays = _get_cifti_parcel_info(str(p))
        assert cifti_type == 'pscalar'
        assert list(element_arrays['parcel_id']) == parcel_names

    def test_pconn_from_toy_data(self):
        cifti_type, element_arrays = _get_cifti_parcel_info(str(EXAMPLE_PCONN))
        assert cifti_type == 'pconn'
        assert 'parcel_id_from' in element_arrays
        assert 'parcel_id_to' in element_arrays
        # Row and column axes should have the same length for this symmetric file
        assert len(element_arrays['parcel_id_from']) == len(element_arrays['parcel_id_to'])


# ===========================================================================
# _cifti_output_ext
# ===========================================================================


class TestCiftiOutputExt:
    def test_dscalar_extension(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        img = make_dscalar(mask, np.array([1.0], dtype=np.float32))
        assert _cifti_output_ext(img) == '.dscalar.nii'

    def test_pscalar_extension(self):
        img = make_pscalar(['A', 'B'], np.array([1.0, 2.0], dtype=np.float32))
        assert _cifti_output_ext(img) == '.pscalar.nii'

    def test_pconn_extension(self):
        img = make_pconn(['A', 'B'], np.eye(2, dtype=np.float32))
        assert _cifti_output_ext(img) == '.pconn.nii'

    def test_dscalar_from_toy_data(self):
        img = nb.load(EXAMPLE_DSCALAR)
        assert _cifti_output_ext(img) == '.dscalar.nii'

    def test_pconn_from_toy_data(self):
        img = nb.load(EXAMPLE_PCONN)
        assert _cifti_output_ext(img) == '.pconn.nii'


# ===========================================================================
# cifti_to_h5: dscalar → HDF5
# ===========================================================================


class TestCiftiToH5Dscalar:
    def test_output_file_created(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0
        assert out_h5.exists()

    def test_greyordinates_shape(self, tmp_path):
        mask = _dscalar_mask()
        n_go = int(mask.sum())
        paths = _write_dscalar_subjects(tmp_path, mask)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            grey = h5['greyordinates'][...]
            # Stored transposed: (n_columns=2, n_rows=n_go)
            assert grey.shape == (2, n_go)

    def test_greyordinates_structure_names_attr(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert 'structure_names' in h5['greyordinates'].attrs

    def test_scalars_matrix_shape(self, tmp_path):
        mask = _dscalar_mask()
        n_go = int(mask.sum())
        n_subjects = 3
        paths = _write_dscalar_subjects(tmp_path, mask, n_subjects=n_subjects)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert h5['scalars/THICK/values'].shape == (n_subjects, n_go)

    def test_scalars_values_correct(self, tmp_path):
        mask = _dscalar_mask()
        n_go = int(mask.sum())
        paths = _write_dscalar_subjects(tmp_path, mask, n_subjects=2)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            vals = h5['scalars/THICK/values'][...]
        np.testing.assert_allclose(vals[0], np.arange(n_go, dtype=np.float32) * 1, rtol=1e-5)
        np.testing.assert_allclose(vals[1], np.arange(n_go, dtype=np.float32) * 2, rtol=1e-5)

    def test_multiple_scalar_names(self, tmp_path):
        mask = _dscalar_mask()
        paths_thick = _write_dscalar_subjects(tmp_path, mask, scalar_name='THICK')
        paths_area = _write_dscalar_subjects(tmp_path, mask, scalar_name='AREA')
        cohort = tmp_path / 'cohort.csv'
        rows = [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths_thick] + [
            {'scalar_name': 'AREA', 'source_file': str(p)} for p in paths_area
        ]
        _write_cohort_csv(cohort, rows)
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert 'scalars/THICK/values' in h5
            assert 'scalars/AREA/values' in h5

    def test_column_names_written(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask, n_subjects=2)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert 'column_names' in h5['scalars/THICK']
            assert len(h5['scalars/THICK']['column_names'][...]) == 2


# ===========================================================================
# cifti_to_h5: pscalar → HDF5
# ===========================================================================


class TestCiftiToH5Pscalar:
    def test_output_file_created(self, tmp_path):
        paths = _write_pscalar_subjects(tmp_path, _PSCALAR_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(
            cohort, [{'scalar_name': 'MYELIN', 'source_file': str(p)} for p in paths]
        )
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0
        assert out_h5.exists()

    def test_scalars_matrix_shape(self, tmp_path):
        n = len(_PSCALAR_PARCELS)
        paths = _write_pscalar_subjects(tmp_path, _PSCALAR_PARCELS, n_subjects=3)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(
            cohort, [{'scalar_name': 'MYELIN', 'source_file': str(p)} for p in paths]
        )
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert h5['scalars/MYELIN/values'].shape == (3, n)

    def test_parcels_parcel_id_written(self, tmp_path):
        n = len(_PSCALAR_PARCELS)
        paths = _write_pscalar_subjects(tmp_path, _PSCALAR_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(
            cohort, [{'scalar_name': 'MYELIN', 'source_file': str(p)} for p in paths]
        )
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert 'parcels/parcel_id' in h5
            names = h5['parcels/parcel_id'][...].astype(str)
            assert len(names) == n
            assert list(names) == _PSCALAR_PARCELS

    def test_scalars_values_correct(self, tmp_path):
        n = len(_PSCALAR_PARCELS)
        paths = _write_pscalar_subjects(tmp_path, _PSCALAR_PARCELS, n_subjects=2)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(
            cohort, [{'scalar_name': 'MYELIN', 'source_file': str(p)} for p in paths]
        )
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            vals = h5['scalars/MYELIN/values'][...]
        np.testing.assert_allclose(vals[0], np.arange(n, dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(vals[1], np.arange(n, dtype=np.float32) + 10, rtol=1e-5)

    def test_pscalar_from_toy_data_axes(self, tmp_path):
        """Construct pscalar files from axes of the real toy dscalar and pconn."""
        dscalar_img = nb.load(EXAMPLE_DSCALAR)
        pconn_img = nb.load(EXAMPLE_PCONN)
        scalar_axis = dscalar_img.header.get_axis(0)  # ScalarAxis
        parcel_axis = pconn_img.header.get_axis(0)  # ParcelsAxis
        n_parcels = len(parcel_axis)

        header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, parcel_axis))
        paths = []
        for i in range(2):
            data = (np.arange(n_parcels, dtype=np.float32) + i).reshape(1, -1)
            img = nb.Cifti2Image(data, header=header)
            p = tmp_path / f'sub-{i:02d}.pscalar.nii'
            img.to_filename(p)
            paths.append(p)

        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0
        with h5py.File(out_h5, 'r') as h5:
            assert h5['scalars/FC/values'].shape == (2, n_parcels)


# ===========================================================================
# cifti_to_h5: pconn → HDF5
# ===========================================================================


class TestCiftiToH5Pconn:
    def test_output_file_created(self, tmp_path):
        paths = _write_pconn_subjects(tmp_path, _PCONN_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0
        assert out_h5.exists()

    def test_scalars_matrix_shape_flattened(self, tmp_path):
        n = len(_PCONN_PARCELS)
        paths = _write_pconn_subjects(tmp_path, _PCONN_PARCELS, n_subjects=2)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            # pconn matrix is row-major flattened: n_subjects x (n_parcels * n_parcels)
            assert h5['scalars/FC/values'].shape == (2, n * n)

    def test_parcels_parcel_id_from_to_written(self, tmp_path):
        paths = _write_pconn_subjects(tmp_path, _PCONN_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            assert 'parcels/parcel_id_from' in h5
            assert 'parcels/parcel_id_to' in h5
            from_names = h5['parcels/parcel_id_from'][...].astype(str)
            to_names = h5['parcels/parcel_id_to'][...].astype(str)
            assert list(from_names) == _PCONN_PARCELS
            assert list(to_names) == _PCONN_PARCELS

    def test_scalars_values_flattened_row_major(self, tmp_path):
        n = len(_PCONN_PARCELS)
        paths = _write_pconn_subjects(tmp_path, _PCONN_PARCELS, n_subjects=1)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            vals = h5['scalars/FC/values'][...]
        expected = np.eye(n, dtype=np.float32).flatten()
        np.testing.assert_allclose(vals[0], expected, rtol=1e-5)

    def test_pconn_from_toy_data(self, tmp_path):
        pconn_img = nb.load(EXAMPLE_PCONN)
        n_rows, n_cols = pconn_img.shape

        # Write two synthetic pconn files with the same axes as the toy data
        parcel_axis_0 = pconn_img.header.get_axis(0)
        parcel_axis_1 = pconn_img.header.get_axis(1)
        header = nb.cifti2.Cifti2Header.from_axes((parcel_axis_0, parcel_axis_1))
        paths = []
        for i in range(2):
            rng = np.random.default_rng(i)
            data = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
            img = nb.Cifti2Image(data, header=header)
            p = tmp_path / f'sub-{i:02d}.pconn.nii'
            img.to_filename(p)
            paths.append(p)

        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0
        with h5py.File(out_h5, 'r') as h5:
            assert h5['scalars/FC/values'].shape == (2, n_rows * n_cols)


# ===========================================================================
# cifti_to_h5: error handling
# ===========================================================================


class TestCiftiToH5Errors:
    def test_empty_cohort_raises(self, tmp_path):
        cohort = tmp_path / 'empty.csv'
        pd.DataFrame(columns=['scalar_name', 'source_file']).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='does not contain any scalar entries'):
            cifti_to_h5(cohort, output=tmp_path / 'out.h5')

    def test_missing_required_columns_raises(self, tmp_path):
        cohort = tmp_path / 'bad.csv'
        pd.DataFrame({'subject': ['sub-01']}).to_csv(cohort, index=False)
        with pytest.raises(ValueError, match='must contain columns'):
            cifti_to_h5(cohort, output=tmp_path / 'out.h5')


# ===========================================================================
# cifti_to_h5 additional entry-point smoke tests
# ===========================================================================


class TestCiftiToH5EntryPoint:
    def test_returns_zero_on_success(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask, n_subjects=1)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        assert cifti_to_h5(cohort, output=out_h5) == 0

    def test_output_file_exists_after_call(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        assert out_h5.exists()


# ===========================================================================
# h5_to_cifti: dscalar output
# ===========================================================================


def _make_dscalar_example(tmp_path: Path, n_go: int = 5) -> Path:
    mask = np.zeros((3, 3, 3), dtype=bool)
    for ijk in _MASK_COORDS[:n_go]:
        mask[ijk] = True
    img = make_dscalar(mask, np.zeros(n_go, dtype=np.float32))
    p = tmp_path / 'example.dscalar.nii'
    img.to_filename(p)
    return p


class TestH5ToCiftiDscalar:
    def test_output_files_created(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((2, 5), np.float32), ['beta', 'tstat'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.dscalar.nii').exists()
        assert (out_dir / 'analysis_tstat.dscalar.nii').exists()

    def test_output_data_values(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((1, 5)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', matrix, ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.dscalar.nii')
        out_data = out_img.get_fdata().squeeze().astype(np.float32)
        np.testing.assert_allclose(out_data, matrix[0], rtol=1e-5)

    def test_output_uses_example_header(self, tmp_path):
        """Output CIFTI inherits the header from the example file."""
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.dscalar.nii')
        # Shape should match: (1 scalar map, n_greyordinates)
        assert out_img.shape == (1, 5)

    def test_pvalue_creates_1m_file(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['p.value'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_p.value.dscalar.nii').exists()
        assert (out_dir / 'analysis_1m.p.value.dscalar.nii').exists()

    def test_pvalue_1m_values_are_complement(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        rng = np.random.default_rng(1)
        pvals = rng.uniform(0, 1, (1, 5)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', pvals, ['p.value'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        p_data = (
            nb.load(out_dir / 'analysis_p.value.dscalar.nii')
            .get_fdata()
            .squeeze()
            .astype(np.float32)
        )
        oneminus = (
            nb.load(out_dir / 'analysis_1m.p.value.dscalar.nii')
            .get_fdata()
            .squeeze()
            .astype(np.float32)
        )
        np.testing.assert_allclose(oneminus, 1 - p_data, rtol=1e-5)

    def test_non_pvalue_does_not_create_1m_file(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert not list(out_dir.glob('*1m*'))

    def test_result_name_with_space_sanitized(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['my result'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_my_result.dscalar.nii').exists()

    def test_multiple_results(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        names = ['beta', 'se', 'tstat', 'p.value', 'fdr']
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((5, 5), np.float32), names)
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        for name in ['beta', 'se', 'tstat', 'p.value', 'fdr']:
            assert (out_dir / f'analysis_{name}.dscalar.nii').exists()
        # p.value also gets a 1m file
        assert (out_dir / 'analysis_1m.p.value.dscalar.nii').exists()


# ===========================================================================
# h5_to_cifti: pscalar output
# ===========================================================================


def _make_pscalar_example(tmp_path: Path, parcel_names: list[str]) -> Path:
    n = len(parcel_names)
    img = make_pscalar(parcel_names, np.zeros(n, dtype=np.float32))
    p = tmp_path / 'example.pscalar.nii'
    img.to_filename(p)
    return p


class TestH5ToCiftiPscalar:
    PARCELS = ['net_DMN', 'net_FPN', 'net_SOM', 'net_VIS']

    def test_pscalar_output_files_created(self, tmp_path):
        example = _make_pscalar_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        n = len(self.PARCELS)
        _make_h5_results(h5_path, 'analysis', np.ones((2, n), np.float32), ['beta', 'tstat'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.pscalar.nii').exists()
        assert (out_dir / 'analysis_tstat.pscalar.nii').exists()

    def test_pscalar_not_dscalar_extension(self, tmp_path):
        example = _make_pscalar_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        n = len(self.PARCELS)
        _make_h5_results(h5_path, 'analysis', np.ones((1, n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert not list(out_dir.glob('*.dscalar.nii'))
        assert len(list(out_dir.glob('*.pscalar.nii'))) == 1

    def test_pscalar_output_data_values(self, tmp_path):
        example = _make_pscalar_example(tmp_path, self.PARCELS)
        rng = np.random.default_rng(42)
        n = len(self.PARCELS)
        matrix = rng.standard_normal((1, n)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', matrix, ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.pscalar.nii')
        out_data = out_img.get_fdata().squeeze().astype(np.float32)
        np.testing.assert_allclose(out_data, matrix[0], rtol=1e-5)

    def test_pscalar_output_shape(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pscalar_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.pscalar.nii')
        assert out_img.shape == (1, n)

    def test_pscalar_from_toy_data_axes(self, tmp_path):
        """Build a pscalar template from real toy dscalar ScalarAxis + pconn ParcelsAxis."""
        dscalar_img = nb.load(EXAMPLE_DSCALAR)
        pconn_img = nb.load(EXAMPLE_PCONN)
        scalar_axis = dscalar_img.header.get_axis(0)
        parcel_axis = pconn_img.header.get_axis(0)
        n_parcels = len(parcel_axis)

        header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, parcel_axis))
        template = nb.Cifti2Image(np.zeros((1, n_parcels), dtype=np.float32), header=header)
        example_path = tmp_path / 'example_real.pscalar.nii'
        template.to_filename(example_path)

        rng = np.random.default_rng(7)
        matrix = rng.standard_normal((2, n_parcels)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', matrix, ['comp1', 'comp2'])

        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example_path), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_comp1.pscalar.nii').exists()
        assert (out_dir / 'analysis_comp2.pscalar.nii').exists()
        out_img = nb.load(out_dir / 'analysis_comp1.pscalar.nii')
        assert out_img.shape == (1, n_parcels)


# ===========================================================================
# h5_to_cifti: pconn output
# ===========================================================================


def _make_pconn_example(tmp_path: Path, parcel_names: list[str]) -> Path:
    n = len(parcel_names)
    img = make_pconn(parcel_names, np.eye(n, dtype=np.float32))
    p = tmp_path / 'example.pconn.nii'
    img.to_filename(p)
    return p


class TestH5ToCiftiPconn:
    PARCELS = ['net_A', 'net_B', 'net_C']

    def test_pconn_output_file_created(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pconn_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n * n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.pconn.nii').exists()

    def test_pconn_not_dscalar_extension(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pconn_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n * n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert not list(out_dir.glob('*.dscalar.nii'))
        assert not list(out_dir.glob('*.pscalar.nii'))
        assert len(list(out_dir.glob('*.pconn.nii'))) == 1

    def test_pconn_output_shape(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pconn_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n * n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.pconn.nii')
        assert out_img.shape == (n, n)

    def test_pconn_output_data_values(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pconn_example(tmp_path, self.PARCELS)
        rng = np.random.default_rng(3)
        matrix = rng.standard_normal((1, n * n)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', matrix, ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        out_img = nb.load(out_dir / 'analysis_beta.pconn.nii')
        out_data = out_img.get_fdata().astype(np.float32)
        np.testing.assert_allclose(out_data, matrix[0].reshape(n, n), rtol=1e-5)

    def test_pconn_multiple_results(self, tmp_path):
        n = len(self.PARCELS)
        example = _make_pconn_example(tmp_path, self.PARCELS)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(
            h5_path, 'analysis', np.ones((3, n * n), np.float32), ['IC1', 'IC2', 'IC3']
        )
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        for name in ['IC1', 'IC2', 'IC3']:
            assert (out_dir / f'analysis_{name}.pconn.nii').exists()

    def test_pconn_from_toy_data(self, tmp_path):
        """Use the real toy pconn file as the example template."""
        pconn_img = nb.load(EXAMPLE_PCONN)
        n_rows, n_cols = pconn_img.shape

        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((2, n_rows * n_cols)).astype(np.float32)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'my_analysis', matrix, ['comp1', 'comp2'])

        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(EXAMPLE_PCONN), str(h5_path), 'my_analysis', str(out_dir))
        assert (out_dir / 'my_analysis_comp1.pconn.nii').exists()
        assert (out_dir / 'my_analysis_comp2.pconn.nii').exists()

        out_img = nb.load(out_dir / 'my_analysis_comp1.pconn.nii')
        assert out_img.shape == (n_rows, n_cols)
        np.testing.assert_allclose(
            out_img.get_fdata().astype(np.float32),
            matrix[0].reshape(n_rows, n_cols),
            rtol=1e-5,
        )


# ===========================================================================
# h5_to_cifti additional smoke tests
# ===========================================================================


class TestH5ToCiftiDirect:
    def test_dscalar_returns_zero(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((2, 5), np.float32), ['beta', 'tstat'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.dscalar.nii').exists()
        assert (out_dir / 'analysis_tstat.dscalar.nii').exists()

    def test_pscalar_with_example(self, tmp_path):
        parcels = ['A', 'B', 'C']
        example = _make_pscalar_example(tmp_path, parcels)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, len(parcels)), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.pscalar.nii').exists()

    def test_pconn_with_example(self, tmp_path):
        parcels = ['X', 'Y']
        n = len(parcels)
        example = _make_pconn_example(tmp_path, parcels)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n * n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        h5_to_cifti(str(example), str(h5_path), 'analysis', str(out_dir))
        assert (out_dir / 'analysis_beta.pconn.nii').exists()


# ===========================================================================
# End-to-end tests via the top-level modelarrayio CLI entry point
# ===========================================================================


class TestCiftiToModelarrayViaCLI:
    """End-to-end tests that exercise CIFTI conversion through the to-modelarray subcommand."""

    def test_cifti_to_h5_creates_expected_hdf5(self, tmp_path, monkeypatch):
        vol_shape = (3, 3, 3)
        mask = np.zeros(vol_shape, dtype=bool)
        true_vox = [(0, 0, 0), (0, 1, 2), (1, 1, 1), (2, 2, 0), (2, 1, 2)]
        for ijk in true_vox:
            mask[ijk] = True
        n_grayordinates = int(mask.sum())

        subjects = []
        for sidx in range(2):
            vals = np.arange(n_grayordinates, dtype=np.float32) + sidx
            img = make_dscalar(mask, vals)
            path = tmp_path / f'sub-{sidx + 1}.dscalar.nii'
            img.to_filename(path)
            subjects.append(str(path.name))

        cohort_csv = tmp_path / 'cohort_cifti.csv'
        _write_cohort_csv(
            cohort_csv,
            [{'scalar_name': 'THICK', 'source_file': sname} for sname in subjects],
        )

        out_h5 = tmp_path / 'out_cifti.h5'
        monkeypatch.chdir(tmp_path)
        assert (
            modelarrayio_main(
                [
                    'to-modelarray',
                    '--cohort-file',
                    str(cohort_csv),
                    '--output',
                    str(out_h5),
                    '--backend',
                    'hdf5',
                    '--dtype',
                    'float32',
                    '--compression',
                    'gzip',
                    '--compression-level',
                    '1',
                    '--target-chunk-mb',
                    '1.0',
                ]
            )
            == 0
        )
        assert out_h5.exists()

        with h5py.File(out_h5, 'r') as h5:
            assert 'greyordinates' in h5
            grey = np.array(h5['greyordinates'])
            assert grey.shape[0] == 2  # vertex_id, structure_id columns
            assert grey.shape[1] == n_grayordinates

            g = h5['greyordinates']
            assert 'structure_names' in g.attrs
            assert len(g.attrs['structure_names']) >= 1

            dset = h5['scalars/THICK/values']
            n_files, n_elements = dset.shape
            assert n_files == 2
            assert n_elements == n_grayordinates

            grp = h5['scalars/THICK']
            assert 'column_names' in grp
            colnames = [
                x.decode('utf-8') if isinstance(x, bytes) else str(x)
                for x in grp['column_names'][...]
            ]
            assert len(colnames) == 2

            assert np.isclose(float(dset[0, 0]), 0.0)
            assert np.isclose(float(dset[1, 0]), 1.0)

    def test_cifti_to_h5_scalar_columns_writes_prefixed_outputs(self, tmp_path, monkeypatch):
        vol_shape = (2, 2, 2)
        mask = np.zeros(vol_shape, dtype=bool)
        mask[(0, 0, 0)] = True
        mask[(1, 1, 1)] = True
        n_grayordinates = int(mask.sum())

        rows = []
        for sidx in range(2):
            alpha_vals = np.arange(n_grayordinates, dtype=np.float32) + sidx
            beta_vals = np.arange(n_grayordinates, dtype=np.float32) + 10 + sidx

            alpha_img = make_dscalar(mask, alpha_vals)
            beta_img = make_dscalar(mask, beta_vals)

            alpha_path = tmp_path / f'sub-{sidx + 1}_alpha.dscalar.nii'
            beta_path = tmp_path / f'sub-{sidx + 1}_beta.dscalar.nii'
            alpha_img.to_filename(alpha_path)
            beta_img.to_filename(beta_path)

            rows.append(
                {'subject_id': f'sub-{sidx + 1}', 'alpha': alpha_path.name, 'beta': beta_path.name}
            )

        cohort_csv = tmp_path / 'cohort_cifti_wide.csv'
        with cohort_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'alpha', 'beta'])
            writer.writeheader()
            writer.writerows(rows)

        out_h5 = tmp_path / 'greyordinatearray.h5'
        alpha_out = tmp_path / 'alpha_greyordinatearray.h5'
        beta_out = tmp_path / 'beta_greyordinatearray.h5'

        monkeypatch.chdir(tmp_path)
        assert (
            modelarrayio_main(
                [
                    'to-modelarray',
                    '--cohort-file',
                    str(cohort_csv),
                    '--scalar-columns',
                    'alpha',
                    'beta',
                    '--output',
                    str(out_h5),
                ]
            )
            == 0
        )

        assert alpha_out.exists()
        assert beta_out.exists()
        assert not out_h5.exists()

        with h5py.File(alpha_out, 'r') as h5:
            assert 'greyordinates' in h5
            assert sorted(h5['scalars'].keys()) == ['alpha']

        with h5py.File(beta_out, 'r') as h5:
            assert 'greyordinates' in h5
            assert sorted(h5['scalars'].keys()) == ['beta']
