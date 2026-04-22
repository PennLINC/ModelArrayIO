"""Unit tests for CIFTI utility helpers."""

from __future__ import annotations

import numpy as np
import pytest
from nibabel.cifti2.cifti2_axes import ScalarAxis
from utils import make_dscalar, make_parcels_axis, make_pconn, make_pscalar

from modelarrayio.utils.cifti import brain_names_to_dataframe, extract_cifti_scalar_data


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
    image = make_dscalar(mask, np.array([1.0, 2.0], dtype=np.float32))

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
    image = make_dscalar(mask, np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match='Inconsistent greyordinate names'):
        extract_cifti_scalar_data(image, reference_brain_names=np.array(['wrong', 'names']))


def test_extract_cifti_scalar_data_pscalar_returns_data_and_names() -> None:
    parcel_names = ['parcel_A', 'parcel_B', 'parcel_C']
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    image = make_pscalar(parcel_names, values)

    data, names = extract_cifti_scalar_data(image)

    np.testing.assert_array_equal(data, values)
    assert list(names) == parcel_names


def test_extract_cifti_scalar_data_pscalar_validates_reference_names() -> None:
    parcel_names = ['parcel_A', 'parcel_B', 'parcel_C']
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    image = make_pscalar(parcel_names, values)

    with pytest.raises(ValueError, match='Inconsistent parcel names'):
        extract_cifti_scalar_data(image, reference_brain_names=np.array(['X', 'Y', 'Z']))


def test_extract_cifti_scalar_data_pconn_flattens_matrix() -> None:
    parcel_names = ['parcel_A', 'parcel_B']
    n = len(parcel_names)
    matrix = np.arange(n * n, dtype=np.float32).reshape(n, n)
    image = make_pconn(parcel_names, matrix)

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
    image = make_pconn(parcel_names, matrix)

    # Get the correct element_names first
    _, element_names = extract_cifti_scalar_data(image)

    # Modify one name to trigger validation failure
    bad_names = element_names.copy()
    bad_names[0] = 'wrong'
    with pytest.raises(ValueError, match='Inconsistent parcel names'):
        extract_cifti_scalar_data(image, reference_brain_names=bad_names)


def test_make_parcels_axis_produces_valid_axis() -> None:
    """Smoke test: make_parcels_axis should return a ParcelsAxis with correct length."""
    from nibabel.cifti2.cifti2_axes import ParcelsAxis

    names = ['A', 'B', 'C']
    axis = make_parcels_axis(names)
    assert isinstance(axis, ParcelsAxis)
    assert len(axis) == len(names)


def test_brain_names_to_dataframe() -> None:
    names = np.array(['CORTEX_LEFT', 'CORTEX_LEFT', 'CORTEX_RIGHT'])
    gdf, struct_strings = brain_names_to_dataframe(names)
    assert len(gdf) == 3
    assert 'vertex_id' in gdf.columns
    assert 'structure_id' in gdf.columns
    assert gdf['vertex_id'].tolist() == [0, 1, 2]
    assert len(struct_strings) == 2  # factorize unique structures
