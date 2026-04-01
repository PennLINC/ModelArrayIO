"""Comprehensive unit tests for cifti_to_h5 and h5_to_cifti CLI functions.

Covers dscalar, pscalar, and pconn CIFTI types for both directions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ParcelsAxis, ScalarAxis

from modelarrayio.cli.cifti_to_h5 import cifti_to_h5, cifti_to_h5_main
from modelarrayio.cli.h5_to_cifti import _cifti_output_ext, h5_to_cifti, h5_to_cifti_main

DATA_DIR = Path(__file__).parent / 'data_cifti_toy'
EXAMPLE_DSCALAR = DATA_DIR / 'example.dscalar.nii'
EXAMPLE_PCONN = DATA_DIR / 'example.pconn.nii'


# ---------------------------------------------------------------------------
# CIFTI image factories (shared with test_cifti_utils.py pattern)
# ---------------------------------------------------------------------------


def _make_parcels_axis(parcel_names: list[str]) -> ParcelsAxis:
    """Create a minimal surface-only ParcelsAxis (one vertex per parcel)."""
    n = len(parcel_names)
    nvertices = {'CIFTI_STRUCTURE_CORTEX_LEFT': n}
    vox_dtype = np.dtype([('ijk', '<i4', (3,))])
    voxels = [np.array([], dtype=vox_dtype) for _ in range(n)]
    vertices = [{'CIFTI_STRUCTURE_CORTEX_LEFT': np.array([i], dtype=np.int32)} for i in range(n)]
    return ParcelsAxis(parcel_names, voxels, vertices, np.eye(4), (10, 10, 10), nvertices)


def _make_dscalar(mask: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    brain_axis = BrainModelAxis.from_mask(mask)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def _make_pscalar(parcel_names: list[str], values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    parcels_axis = _make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, parcels_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def _make_pconn(parcel_names: list[str], matrix: np.ndarray) -> nb.Cifti2Image:
    parcels_axis = _make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))
    n = len(parcel_names)
    return nb.Cifti2Image(matrix.reshape(n, n).astype(np.float32), header=header)


def _write_cohort_csv(path: Path, rows: list[dict]) -> None:
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


# ===========================================================================
# _cifti_output_ext
# ===========================================================================


class TestCiftiOutputExt:
    def test_dscalar_extension(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        img = _make_dscalar(mask, np.array([1.0], dtype=np.float32))
        assert _cifti_output_ext(img) == '.dscalar.nii'

    def test_pscalar_extension(self):
        img = _make_pscalar(['A', 'B'], np.array([1.0, 2.0], dtype=np.float32))
        assert _cifti_output_ext(img) == '.pscalar.nii'

    def test_pconn_extension(self):
        img = _make_pconn(['A', 'B'], np.eye(2, dtype=np.float32))
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
        img = _make_dscalar(mask, vals)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.dscalar.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


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
        img = _make_pscalar(parcel_names, vals)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.pscalar.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


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

    def test_greyordinates_size_equals_n_parcels(self, tmp_path):
        n = len(_PSCALAR_PARCELS)
        paths = _write_pscalar_subjects(tmp_path, _PSCALAR_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(
            cohort, [{'scalar_name': 'MYELIN', 'source_file': str(p)} for p in paths]
        )
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            grey = h5['greyordinates'][...]
            assert grey.shape[1] == n

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
        img = _make_pconn(parcel_names, matrix)
        p = tmp_path / f'sub-{i:02d}_{scalar_name}.pconn.nii'
        img.to_filename(p)
        paths.append(p)
    return paths


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

    def test_greyordinates_size_equals_flattened_matrix(self, tmp_path):
        n = len(_PCONN_PARCELS)
        paths = _write_pconn_subjects(tmp_path, _PCONN_PARCELS)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'FC', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5(cohort, output=out_h5)
        with h5py.File(out_h5, 'r') as h5:
            grey = h5['greyordinates'][...]
            assert grey.shape[1] == n * n

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
# cifti_to_h5_main entry point
# ===========================================================================


class TestCiftiToH5Main:
    def test_returns_zero_on_success(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask, n_subjects=1)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        result = cifti_to_h5_main(cohort_file=str(cohort), output=out_h5)
        assert result == 0

    def test_output_file_exists_after_main(self, tmp_path):
        mask = _dscalar_mask()
        paths = _write_dscalar_subjects(tmp_path, mask)
        cohort = tmp_path / 'cohort.csv'
        _write_cohort_csv(cohort, [{'scalar_name': 'THICK', 'source_file': str(p)} for p in paths])
        out_h5 = tmp_path / 'out.h5'
        cifti_to_h5_main(cohort_file=str(cohort), output=out_h5)
        assert out_h5.exists()


# ===========================================================================
# h5_to_cifti: dscalar output
# ===========================================================================


def _make_dscalar_example(tmp_path: Path, n_go: int = 5) -> Path:
    mask = np.zeros((3, 3, 3), dtype=bool)
    for ijk in _MASK_COORDS[:n_go]:
        mask[ijk] = True
    img = _make_dscalar(mask, np.zeros(n_go, dtype=np.float32))
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
    img = _make_pscalar(parcel_names, np.zeros(n, dtype=np.float32))
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
    img = _make_pconn(parcel_names, np.eye(n, dtype=np.float32))
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
# h5_to_cifti_main entry point
# ===========================================================================


class TestH5ToCiftiMain:
    def test_main_with_example_cifti_returns_zero(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((2, 5), np.float32), ['beta', 'tstat'])
        out_dir = tmp_path / 'out'
        result = h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            example_cifti=str(example),
        )
        assert result == 0

    def test_main_with_example_cifti_creates_files(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((2, 5), np.float32), ['beta', 'tstat'])
        out_dir = tmp_path / 'out'
        h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            example_cifti=str(example),
        )
        assert (out_dir / 'analysis_beta.dscalar.nii').exists()
        assert (out_dir / 'analysis_tstat.dscalar.nii').exists()

    def test_main_with_cohort_file_returns_zero(self, tmp_path):
        example = _make_dscalar_example(tmp_path)
        cohort_csv = tmp_path / 'cohort.csv'
        pd.DataFrame({'source_file': [str(example)]}).to_csv(cohort_csv, index=False)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        result = h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            cohort_file=str(cohort_csv),
        )
        assert result == 0

    def test_main_with_cohort_file_uses_first_source(self, tmp_path):
        """cohort_file mode picks the first source_file row as the example CIFTI."""
        example = _make_dscalar_example(tmp_path)
        cohort_csv = tmp_path / 'cohort.csv'
        pd.DataFrame({'source_file': [str(example)]}).to_csv(cohort_csv, index=False)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, 5), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            cohort_file=str(cohort_csv),
        )
        assert (out_dir / 'analysis_beta.dscalar.nii').exists()

    def test_main_pscalar_with_example_cifti(self, tmp_path):
        parcels = ['A', 'B', 'C']
        example = _make_pscalar_example(tmp_path, parcels)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, len(parcels)), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        result = h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            example_cifti=str(example),
        )
        assert result == 0
        assert (out_dir / 'analysis_beta.pscalar.nii').exists()

    def test_main_pconn_with_example_cifti(self, tmp_path):
        parcels = ['X', 'Y']
        n = len(parcels)
        example = _make_pconn_example(tmp_path, parcels)
        h5_path = tmp_path / 'results.h5'
        _make_h5_results(h5_path, 'analysis', np.ones((1, n * n), np.float32), ['beta'])
        out_dir = tmp_path / 'out'
        result = h5_to_cifti_main(
            analysis_name='analysis',
            in_file=str(h5_path),
            output_dir=str(out_dir),
            example_cifti=str(example),
        )
        assert result == 0
        assert (out_dir / 'analysis_beta.pconn.nii').exists()
