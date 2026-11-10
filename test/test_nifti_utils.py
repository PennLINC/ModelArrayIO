"""Unit tests for voxel array helpers."""

from __future__ import annotations

import threading
import time

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from modelarrayio.utils import nifti
from modelarrayio.utils.loading import SourceLoadError
from modelarrayio.utils.nifti import ReducedMaskCache, flattened_image, load_cohort_voxels


class _TrackingImage:
    def __init__(self, data: np.ndarray, *, return_shared: bool = False) -> None:
        self.dataobj = data
        self._data = data
        self._return_shared = return_shared
        self.get_fdata_calls: list[tuple[np.dtype, str]] = []

    def get_fdata(self, *, dtype, caching):
        resolved = np.dtype(dtype)
        self.get_fdata_calls.append((resolved, caching))
        if self._return_shared and self._data.dtype == resolved:
            return self._data
        return np.asarray(self._data, dtype=resolved)


def test_flattened_image_extracts_group_masked_values() -> None:
    shape = (3, 3, 3)
    group_mask = np.zeros(shape, dtype=bool)
    group_mask[0, 1, 2] = True
    group_mask[2, 0, 1] = True

    scalar = np.zeros(shape, dtype=np.float32)
    scalar[0, 1, 2] = 1.5
    scalar[2, 0, 1] = 2.5

    indiv_mask = group_mask.copy()
    scalar_img = nb.Nifti1Image(scalar, np.eye(4))
    mask_img = nb.Nifti1Image(indiv_mask.astype(np.float32), np.eye(4))

    flat = flattened_image(scalar_img, mask_img, group_mask)
    assert flat.shape == (2,)
    assert flat.dtype == np.float32
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

    scalar_img = nb.Nifti1Image(scalar, np.eye(4))
    mask_img = nb.Nifti1Image(indiv.astype(np.float32), np.eye(4))

    flat = flattened_image(scalar_img, mask_img, group_mask)
    assert flat.shape == (2,)
    assert np.isnan(flat[1])
    assert flat[0] == pytest.approx(3.0)


def test_flattened_image_requests_dtype_without_caching_and_preserves_precision() -> None:
    precise = np.array([[[1.0 + 2.0**-40]]], dtype=np.float64)
    scalar_img = _TrackingImage(precise)
    mask_img = _TrackingImage(np.ones((1, 1, 1), dtype=np.uint8))

    flat = flattened_image(
        scalar_img,
        mask_img,
        np.ones((1, 1, 1), dtype=bool),
        storage_dtype='float64',
    )

    assert flat.dtype == np.float64
    assert flat[0] == precise.item()
    assert scalar_img.get_fdata_calls == [(np.dtype(np.float64), 'unchanged')]
    assert mask_img.get_fdata_calls == []


def test_load_cohort_voxels_is_lazy_ordered_typed_and_skips_group_mask(
    monkeypatch, tmp_path
) -> None:
    group_mask = np.array([[[True, False, True]]])
    group_mask_path = tmp_path / 'group.nii.gz'
    equivalent_group_path = tmp_path / 'unused' / '..' / group_mask_path.name
    images = {
        'late.nii.gz': _TrackingImage(np.array([[[30.0, 0.0, 31.0]]])),
        'early.nii.gz': _TrackingImage(np.array([[[10.0, 0.0, 11.0]]])),
        'other.nii.gz': _TrackingImage(np.array([[[20.0, 0.0, 21.0]]])),
    }
    loaded_paths: list[str] = []
    lock = threading.Lock()

    def fake_load(path):
        path = str(path)
        with lock:
            loaded_paths.append(path)
        if path == 'late.nii.gz':
            time.sleep(0.02)
        return images[path]

    monkeypatch.setattr(nifti, 'load_nibabel', fake_load)
    cohort = pd.DataFrame(
        [
            {
                'scalar_name': 'fa',
                'source_file': 'late.nii.gz',
                'source_mask_file': str(equivalent_group_path),
            },
            {
                'scalar_name': 'fa',
                'source_file': 'early.nii.gz',
                'source_mask_file': str(group_mask_path),
            },
            {
                'scalar_name': 'md',
                'source_file': 'other.nii.gz',
                'source_mask_file': str(group_mask_path),
            },
        ]
    )

    scalars, sources = load_cohort_voxels(
        cohort,
        group_mask,
        workers=2,
        storage_dtype='float64',
        group_mask_path=group_mask_path,
    )

    assert loaded_paths == []
    assert list(scalars) == ['fa', 'md']
    assert sources == {
        'fa': ['late.nii.gz', 'early.nii.gz'],
        'md': ['other.nii.gz'],
    }

    fa = scalars['fa']
    assert sorted(loaded_paths) == ['early.nii.gz', 'late.nii.gz']
    assert np.array_equal(fa, [[30.0, 31.0], [10.0, 11.0]])
    assert fa.dtype == np.float64
    assert fa.flags.c_contiguous
    assert all(
        image.get_fdata_calls == [(np.dtype(np.float64), 'unchanged')]
        for name, image in images.items()
        if name != 'other.nii.gz'
    )


def test_mask_cache_deduplicates_concurrent_loads_and_preserves_order(
    monkeypatch, tmp_path
) -> None:
    group_mask = np.ones((1, 1, 2), dtype=bool)
    mask_path = str(tmp_path / 'individual-mask.nii.gz')
    images = {
        f'sub-{index}.nii.gz': _TrackingImage(
            np.array([[[float(index), float(index) + 0.5]]], dtype=np.float32)
        )
        for index in range(6)
    }
    images[mask_path] = _TrackingImage(np.array([[[1, 0]]], dtype=np.uint8))
    mask_load_count = 0
    lock = threading.Lock()

    def fake_load(path):
        nonlocal mask_load_count
        path = str(path)
        if path == mask_path:
            with lock:
                mask_load_count += 1
            time.sleep(0.02)
        return images[path]

    monkeypatch.setattr(nifti, 'load_nibabel', fake_load)
    cohort = pd.DataFrame(
        [
            {
                'scalar_name': 'fa',
                'source_file': f'sub-{index}.nii.gz',
                'source_mask_file': mask_path,
            }
            for index in range(6)
        ]
    )

    scalars, _ = load_cohort_voxels(cohort, group_mask, workers=4)
    matrix = scalars['fa']

    assert mask_load_count == 1
    assert np.array_equal(matrix[:, 0], np.arange(6, dtype=np.float32))
    assert np.isnan(matrix[:, 1]).all()
    assert len(scalars.mask_cache) == 1


def test_repeated_scalar_decode_is_never_mutated_by_subject_mask(monkeypatch, tmp_path) -> None:
    group_mask = np.ones((1, 1, 2), dtype=bool)
    group_path = str(tmp_path / 'group.nii.gz')
    individual_path = str(tmp_path / 'individual.nii.gz')
    shared_data = np.array([[[1.0, 2.0]]], dtype=np.float32)
    scalar_image = _TrackingImage(shared_data, return_shared=True)
    images = {
        'shared.nii.gz': scalar_image,
        individual_path: _TrackingImage(np.array([[[1, 0]]], dtype=np.uint8)),
    }
    monkeypatch.setattr(nifti, 'load_nibabel', lambda path: images[str(path)])
    cohort = pd.DataFrame(
        [
            {
                'scalar_name': 'fa',
                'source_file': 'shared.nii.gz',
                'source_mask_file': individual_path,
            },
            {
                'scalar_name': 'fa',
                'source_file': 'shared.nii.gz',
                'source_mask_file': group_path,
            },
        ]
    )

    scalars, _ = load_cohort_voxels(
        cohort,
        group_mask,
        workers=1,
        group_mask_path=group_path,
    )
    matrix = scalars['fa']

    assert np.isnan(matrix[0, 1])
    assert np.array_equal(matrix[1], [1.0, 2.0])
    assert np.array_equal(shared_data, [[[1.0, 2.0]]])


def test_reduced_mask_cache_is_bounded_and_protects_cached_arrays(tmp_path) -> None:
    cache = ReducedMaskCache(max_size=2)
    loads = 0

    def load(value):
        nonlocal loads
        loads += 1
        return np.array([value], dtype=bool)

    first = cache.get_or_load(tmp_path / 'a.nii.gz', lambda: load(True))
    cache.get_or_load(tmp_path / 'b.nii.gz', lambda: load(False))
    cache.get_or_load(tmp_path / 'c.nii.gz', lambda: load(True))

    assert len(cache) == 2
    assert not first.flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        first[0] = False

    cache.get_or_load(tmp_path / 'a.nii.gz', lambda: load(True))
    assert loads == 4


def test_load_cohort_voxels_adds_scalar_and_source_context(monkeypatch, tmp_path) -> None:
    bad_source = str(tmp_path / 'bad.nii.gz')
    group_path = str(tmp_path / 'group.nii.gz')
    monkeypatch.setattr(
        nifti,
        'load_nibabel',
        lambda _path: _TrackingImage(np.ones((2, 2, 2), dtype=np.float32)),
    )
    cohort = pd.DataFrame(
        [
            {
                'scalar_name': 'fa',
                'source_file': bad_source,
                'source_mask_file': group_path,
            }
        ]
    )
    scalars, _ = load_cohort_voxels(
        cohort,
        np.ones((1, 1, 1), dtype=bool),
        workers=1,
        group_mask_path=group_path,
    )

    with pytest.raises(SourceLoadError) as exc_info:
        scalars['fa']
    message = str(exc_info.value)
    assert bad_source in message
    assert "scalar 'fa'" in message
    assert 'matching shapes' in message


def test_load_cohort_voxels_legacy_worker_argument_and_conflict() -> None:
    empty_cohort = pd.DataFrame(columns=['scalar_name', 'source_file', 'source_mask_file'])
    scalars, sources = load_cohort_voxels(
        empty_cohort,
        np.ones((1, 1, 1), dtype=bool),
        2,
    )
    assert len(scalars) == 0
    assert sources == {}

    with pytest.raises(ValueError, match='Conflicting loader worker counts'):
        load_cohort_voxels(
            empty_cohort,
            np.ones((1, 1, 1), dtype=bool),
            2,
            workers=3,
        )
