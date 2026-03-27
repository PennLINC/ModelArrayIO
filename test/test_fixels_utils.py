"""Unit tests for fixel utility error handling."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pytest

from modelarrayio.utils import mif


def _make_nifti2(shape=(2, 1, 1)) -> nb.Nifti2Image:
    data = np.zeros(shape, dtype=np.float32)
    return nb.Nifti2Image(data, affine=np.eye(4))


def test_nifti2_to_mif_raises_when_mrconvert_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mif, 'find_mrconvert', lambda: None)

    with pytest.raises(FileNotFoundError, match='mrconvert'):
        mif.nifti2_to_mif(_make_nifti2(), tmp_path / 'out.mif')


def test_mif_to_nifti2_raises_when_mrconvert_missing(monkeypatch) -> None:
    monkeypatch.setattr(mif, 'find_mrconvert', lambda: None)

    with pytest.raises(FileNotFoundError, match='mrconvert'):
        mif.mif_to_nifti2('missing_input.mif')
