"""Unit tests for CIFTI validation helpers."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pytest
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

from modelarrayio.utils.cifti import extract_cifti_scalar_data


def _make_scalar_cifti(mask_bool: np.ndarray, values: np.ndarray) -> nb.Cifti2Image:
    scalar_axis = ScalarAxis(['synthetic'])
    brain_axis = BrainModelAxis.from_mask(mask_bool)
    header = nb.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    return nb.Cifti2Image(values.reshape(1, -1).astype(np.float32), header=header)


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


def test_extract_cifti_scalar_data_rejects_missing_brain_axis() -> None:
    image = _FakeCiftiImage(
        [ScalarAxis(['a']), ScalarAxis(['b'])],
        np.array([[1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='scalar axis'):
        extract_cifti_scalar_data(image)


def test_extract_cifti_scalar_data_rejects_inconsistent_reference_names() -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    image = _make_scalar_cifti(mask, np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match='Inconsistent greyordinate names'):
        extract_cifti_scalar_data(image, reference_brain_names=np.array(['wrong', 'names']))
