"""Unit tests for CIFTI validation helpers."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pytest
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ParcelsAxis, ScalarAxis

from modelarrayio.utils.cifti import extract_cifti_scalar_data


def _make_scalar_cifti(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def _make_parcels_axis(parcel_names: list[str]) -> ParcelsAxis:
    """Create a minimal surface-only ParcelsAxis for testing."""
    # One vertex per parcel on the left cortex
    n = len(parcel_names)
    nvertices = {'CIFTI_STRUCTURE_CORTEX_LEFT': n}
    vox_dtype = np.dtype([('ijk', '<i4', (3,))])
    voxels = [np.array([], dtype=vox_dtype) for _ in range(n)]
    vertices = [{'CIFTI_STRUCTURE_CORTEX_LEFT': np.array([i], dtype=np.int32)} for i in range(n)]
    affine = np.eye(4)
    volume_shape = (10, 10, 10)
    return ParcelsAxis(parcel_names, voxels, vertices, affine, volume_shape, nvertices)


def _make_pscalar_cifti(parcel_names: list[str], values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    parcels_axis = _make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, parcels_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


def _make_pconn_cifti(parcel_names: list[str], values: np.ndarray) -> nb.Cifti2Image:
    parcels_axis = _make_parcels_axis(parcel_names)
    header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))
    n = len(parcel_names)
    return nb.Cifti2Image(values.reshape(n, n).astype(np.float32), header=header)


class _FakeHeader:
    def __init__(self, axes):
        self._axes = axes

    def get_axis(self, index: int):
        return self._axes[index]


class _FakeCiftiImage:
    def __init__(self, axes, data: np.ndarray):
        self.header = _FakeHeader(axes)
        self.ndim = len(axes)
        self._data = data

    def get_fdata(self) -> np.ndarray:
        return self._data


def test_extract_cifti_scalar_data_returns_data_and_names() -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    image = _make_scalar_cifti(mask, np.array([1.0, 2.0], dtype=np.float32))

    data, names = extract_cifti_scalar_data(image)

    np.testing.assert_array_equal(data, np.array([1.0, 2.0], dtype=np.float32))
    assert names.shape == (2,)


def test_extract_cifti_scalar_data_rejects_wrong_axis_count() -> None:
    image = _FakeCiftiImage([ScalarAxis(['a'])], np.array([1.0], dtype=np.float32))

    with pytest.raises(ValueError, match='exactly 2 axes'):
        extract_cifti_scalar_data(image)


def test_extract_cifti_scalar_data_rejects_unsupported_axes() -> None:
    image = _FakeCiftiImage(
        [ScalarAxis(['a']), ScalarAxis(['b'])],
        np.array([[1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='Unsupported CIFTI axis combination'):
        extract_cifti_scalar_data(image)


def test_extract_cifti_scalar_data_rejects_inconsistent_reference_names() -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    image = _make_scalar_cifti(mask, np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match='Inconsistent greyordinate names'):
        extract_cifti_scalar_data(image, reference_brain_names=np.array(['wrong', 'names']))


def test_extract_cifti_scalar_data_pscalar_returns_data_and_names() -> None:
    parcel_names = ['parcel_A', 'parcel_B', 'parcel_C']
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    image = _make_pscalar_cifti(parcel_names, values)

    data, names = extract_cifti_scalar_data(image)

    np.testing.assert_array_equal(data, values)
    assert list(names) == parcel_names


def test_extract_cifti_scalar_data_pscalar_validates_reference_names() -> None:
    parcel_names = ['parcel_A', 'parcel_B', 'parcel_C']
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    image = _make_pscalar_cifti(parcel_names, values)

    with pytest.raises(ValueError, match='Inconsistent parcel names'):
        extract_cifti_scalar_data(image, reference_brain_names=np.array(['X', 'Y', 'Z']))


def test_extract_cifti_scalar_data_pconn_flattens_matrix() -> None:
    parcel_names = ['parcel_A', 'parcel_B']
    n = len(parcel_names)
    matrix = np.arange(n * n, dtype=np.float32).reshape(n, n)
    image = _make_pconn_cifti(parcel_names, matrix)

    data, names = extract_cifti_scalar_data(image)

    np.testing.assert_array_equal(data, matrix.flatten())
    assert len(names) == n * n
    # Each row parcel name should appear n_col times consecutively
    assert list(names[:n]) == [parcel_names[0]] * n
    assert list(names[n:]) == [parcel_names[1]] * n


def test_extract_cifti_scalar_data_pconn_validates_reference_names() -> None:
    parcel_names = ['parcel_A', 'parcel_B']
    n = len(parcel_names)
    matrix = np.zeros((n, n), dtype=np.float32)
    image = _make_pconn_cifti(parcel_names, matrix)

    # Get the correct element_names first
    _, element_names = extract_cifti_scalar_data(image)

    # Modify one name to trigger validation failure
    bad_names = element_names.copy()
    bad_names[0] = 'wrong'
    with pytest.raises(ValueError, match='Inconsistent parcel names'):
        extract_cifti_scalar_data(image, reference_brain_names=bad_names)
