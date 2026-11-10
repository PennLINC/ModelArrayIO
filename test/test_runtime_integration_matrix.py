"""Small end-to-end runtime matrix across modalities, backends, workers, and dtypes."""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import pytest
import tiledb
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

from modelarrayio.cli.main import main as modelarrayio_main
from modelarrayio.utils.mif_image import MifImage

RUNTIME_CASES = [
    pytest.param(backend, dtype, workers, id=f'{backend}-{dtype}-workers{workers}')
    for backend in ('hdf5', 'tiledb')
    for dtype in ('float32', 'float64')
    for workers in (1, 2)
]


def _read_values(output: Path, backend: str, scalar_name: str) -> np.ndarray:
    if backend == 'hdf5':
        with h5py.File(output, 'r') as h5_file:
            return h5_file[f'scalars/{scalar_name}/values'][...]
    with tiledb.open(str(output / 'scalars' / scalar_name / 'values'), 'r') as array:
        return array[:]['values']


def _common_args(
    cohort: Path,
    output: Path,
    backend: str,
    dtype: str,
    workers: int,
) -> list[str]:
    return [
        'to-modelarray',
        '--cohort-file',
        str(cohort),
        '--output',
        str(output),
        '--backend',
        backend,
        '--dtype',
        dtype,
        '--workers',
        str(workers),
        '--compression',
        'none',
    ]


@pytest.mark.parametrize(('backend', 'dtype', 'workers'), RUNTIME_CASES)
def test_nifti_runtime_matrix(
    tmp_path: Path,
    backend: str,
    dtype: str,
    workers: int,
) -> None:
    mask_path = tmp_path / 'mask.nii.gz'
    mask = np.ones((2, 2, 1), dtype=np.uint8)
    nb.Nifti1Image(mask, np.eye(4)).to_filename(mask_path)
    values = [
        np.array([[[1.125], [2.25]], [[3.5], [4.75]]], dtype=np.float64),
        np.array([[[5.125], [6.25]], [[7.5], [8.75]]], dtype=np.float64),
    ]
    sources = []
    for index, data in enumerate(values):
        source = tmp_path / f'sub-{index}.nii.gz'
        nb.Nifti1Image(data, np.eye(4)).to_filename(source)
        sources.append(source)
    cohort = tmp_path / 'cohort.csv'
    pd.DataFrame(
        {
            'scalar_name': ['FA', 'FA'],
            'source_file': [str(source) for source in sources],
            'source_mask_file': [str(mask_path), str(mask_path)],
        }
    ).to_csv(cohort, index=False)
    output = tmp_path / ('output.h5' if backend == 'hdf5' else 'output.tdb')

    status = modelarrayio_main(
        [*_common_args(cohort, output, backend, dtype, workers), '--mask', str(mask_path)]
    )

    assert status == 0
    actual = _read_values(output, backend, 'FA')
    expected = np.stack([value.ravel() for value in values]).astype(dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(('backend', 'dtype', 'workers'), RUNTIME_CASES)
def test_cifti_runtime_matrix(
    tmp_path: Path,
    backend: str,
    dtype: str,
    workers: int,
) -> None:
    mask = np.ones((1, 1, 2), dtype=bool)
    brain_axis = BrainModelAxis.from_mask(mask)
    values = [
        np.array([1.125, 2.25], dtype=np.float64),
        np.array([3.5, 4.75], dtype=np.float64),
    ]
    sources = []
    for index, data in enumerate(values):
        source = tmp_path / f'sub-{index}.dscalar.nii'
        header = nb.cifti2.Cifti2Header.from_axes(
            (ScalarAxis([f'scalar-label-{index}']), brain_axis)
        )
        nb.Cifti2Image(data.reshape(1, -1), header=header).to_filename(source)
        sources.append(source)
    cohort = tmp_path / 'cohort.csv'
    pd.DataFrame(
        {
            'scalar_name': ['FA', 'FA'],
            'source_file': [str(source) for source in sources],
        }
    ).to_csv(cohort, index=False)
    output = tmp_path / ('output.h5' if backend == 'hdf5' else 'output.tdb')

    status = modelarrayio_main(_common_args(cohort, output, backend, dtype, workers))

    assert status == 0
    actual = _read_values(output, backend, 'FA')
    expected = np.stack(values).astype(dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(('backend', 'dtype', 'workers'), RUNTIME_CASES)
def test_mif_runtime_matrix(
    tmp_path: Path,
    backend: str,
    dtype: str,
    workers: int,
) -> None:
    index_path = tmp_path / 'index.mif'
    directions_path = tmp_path / 'directions.mif'
    index_data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    index_data[..., 0] = 1
    index_data[..., 1] = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    MifImage(index_data, np.eye(4)).to_filename(index_path)
    MifImage(
        np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1)),
        np.eye(4),
    ).to_filename(directions_path)

    values = [
        np.arange(8, dtype=np.float64) + 0.125,
        np.arange(8, dtype=np.float64) + 10.25,
    ]
    sources = []
    for index, data in enumerate(values):
        source = tmp_path / f'sub-{index}.mif'
        MifImage(data, np.eye(4)).to_filename(source)
        sources.append(source)
    cohort = tmp_path / 'cohort.csv'
    pd.DataFrame(
        {
            'scalar_name': ['FD', 'FD'],
            'source_file': [str(source) for source in sources],
        }
    ).to_csv(cohort, index=False)
    output = tmp_path / ('output.h5' if backend == 'hdf5' else 'output.tdb')
    args = [
        *_common_args(cohort, output, backend, dtype, workers),
        '--index-file',
        str(index_path),
        '--directions-file',
        str(directions_path),
    ]

    status = modelarrayio_main(args)

    assert status == 0
    actual = _read_values(output, backend, 'FD')
    expected = np.stack(values).astype(dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual, expected)
