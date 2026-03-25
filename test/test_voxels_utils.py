"""Unit tests for voxel array helpers."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pytest

from modelarrayio.utils.voxels import flattened_image


def _eye_affine():
    return np.eye(4)


def test_flattened_image_extracts_group_masked_values() -> None:
    shape = (3, 3, 3)
    group_mask = np.zeros(shape, dtype=bool)
    group_mask[0, 1, 2] = True
    group_mask[2, 0, 1] = True

    scalar = np.zeros(shape, dtype=np.float32)
    scalar[0, 1, 2] = 1.5
    scalar[2, 0, 1] = 2.5

    indiv_mask = group_mask.copy()
    scalar_img = nb.Nifti1Image(scalar, _eye_affine())
    mask_img = nb.Nifti1Image(indiv_mask.astype(np.float32), _eye_affine())

    flat = flattened_image(scalar_img, mask_img, group_mask)
    assert flat.shape == (2,)
    assert np.allclose(flat, np.array([1.5, 2.5], dtype=np.float32))


def test_flattened_image_nan_outside_individual_mask() -> None:
    shape = (2, 2, 2)
    group_mask = np.zeros(shape, dtype=bool)
    group_mask[0, 0, 0] = True
    group_mask[1, 1, 1] = True

    scalar = np.ones(shape, dtype=np.float32) * 3.0
    # Individual mask drops one group voxel
    indiv = group_mask.copy()
    indiv[1, 1, 1] = False

    scalar_img = nb.Nifti1Image(scalar, _eye_affine())
    mask_img = nb.Nifti1Image(indiv.astype(np.float32), _eye_affine())

    flat = flattened_image(scalar_img, mask_img, group_mask)
    assert flat.shape == (2,)
    assert np.isnan(flat[1])
    assert flat[0] == pytest.approx(3.0)
