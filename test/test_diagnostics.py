"""Unit tests for conversion diagnostics helpers."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pytest

from modelarrayio.cli import diagnostics as cli_diagnostics


def test_verify_nifti_element_mapping_raises_on_mismatch(monkeypatch):
    group_mask_matrix = np.zeros((2, 2, 2), dtype=bool)
    group_mask_matrix[0, 0, 0] = True
    group_mask_matrix[1, 1, 1] = True
    group_mask_img = nb.Nifti1Image(group_mask_matrix.astype(np.uint8), affine=np.eye(4))

    monkeypatch.setattr(
        cli_diagnostics,
        'flattened_image',
        lambda *_args, **_kwargs: np.array([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='Element ID mapping check failed'):
        cli_diagnostics.verify_nifti_element_mapping(group_mask_img, group_mask_matrix)
