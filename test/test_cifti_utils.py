"""Unit tests for CIFTI utility helpers."""

from __future__ import annotations

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis
from utils import make_dscalar, make_parcels_axis, make_pconn, make_pscalar

from modelarrayio.utils import cifti as cifti_utils
from modelarrayio.utils.cifti import (
    _build_cifti_raw_read_plan,
    _read_cifti_row_raw,
    brain_names_to_dataframe,
    cifti_spatial_signature,
    extract_cifti_scalar_data,
    load_cohort_cifti,
)


def _write_dscalar(path, mask, values, *, map_name='synthetic', metadata=None):
    """Write a real dscalar ``.nii`` file to disk and return its path string."""
    header = nb.cifti2.Cifti2Header.from_axes(
        (ScalarAxis([map_name]), BrainModelAxis.from_mask(mask))
    )
    if metadata:
        # Provenance-style metadata lengthens the CIFTI XML and shifts vox_offset
        # without touching the brain-model axis, as production pipelines do.
        header.matrix.metadata = nb.cifti2.Cifti2MetaData(metadata)
    image = nb.Cifti2Image(np.asarray(values, dtype=np.float32).reshape(1, -1), header=header)
    image.to_filename(str(path))
    return str(path)


def _spy_raw_reads(monkeypatch):
    """Record which sources the raw fast path successfully read from disk."""
    successes: list[str] = []
    real = cifti_utils._read_cifti_row_raw

    def _wrapped(source, plan):
        row = real(source, plan)
        if row is not None:
            successes.append(str(source))
        return row

    monkeypatch.setattr(cifti_utils, '_read_cifti_row_raw', _wrapped)
    return successes


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

    def get_fdata(self, dtype=np.float64, caching='fill') -> np.ndarray:
        return self._data.astype(dtype, copy=False)


def test_extract_cifti_scalar_data_returns_data_and_names() -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    image = make_dscalar(mask, np.array([1.0, 2.0], dtype=np.float32))

    data, names = extract_cifti_scalar_data(image)

    np.testing.assert_array_equal(data, np.array([1.0, 2.0], dtype=np.float32))
    assert names.shape == (2,)


def test_extract_cifti_path_releases_owned_image_buffers(monkeypatch) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    image = make_dscalar(mask, np.array([1.0], dtype=np.float32))
    released = []
    monkeypatch.setattr(cifti_utils, 'load_nibabel', lambda *_args, **_kwargs: image)
    monkeypatch.setattr(cifti_utils, 'release_image_buffers', released.append)

    data, _ = extract_cifti_scalar_data('source.dscalar.nii')

    np.testing.assert_array_equal(data, np.array([1.0], dtype=np.float32))
    assert released == [image]


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


@pytest.mark.parametrize('storage_dtype', ['float32', 'float64'])
def test_extract_cifti_requests_storage_dtype_end_to_end(storage_dtype: str) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    header = nb.cifti2.Cifti2Header.from_axes(
        (ScalarAxis(['precision']), BrainModelAxis.from_mask(mask))
    )
    image = nb.Cifti2Image(
        np.array([[1.123456789, 2.987654321]], dtype=np.float64),
        header=header,
    )

    data, _ = extract_cifti_scalar_data(image, storage_dtype=storage_dtype)

    assert data.dtype == np.dtype(storage_dtype)


def test_cifti_signature_ignores_scalar_labels_but_rejects_spatial_changes() -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    brain_axis = BrainModelAxis.from_mask(mask)
    image_a = nb.Cifti2Image(
        np.array([[1.0, 2.0]], dtype=np.float32),
        header=nb.cifti2.Cifti2Header.from_axes((ScalarAxis(['FA']), brain_axis)),
    )
    image_b = nb.Cifti2Image(
        np.array([[3.0, 4.0]], dtype=np.float32),
        header=nb.cifti2.Cifti2Header.from_axes((ScalarAxis(['MD']), brain_axis)),
    )

    reference_signature = cifti_spatial_signature(image_a)
    assert cifti_spatial_signature(image_b) == reference_signature

    changed_mask = np.zeros_like(mask)
    changed_mask[0, 0, 1] = True
    changed_mask[1, 1, 0] = True
    changed = make_dscalar(changed_mask, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match='Inconsistent spatial axis'):
        extract_cifti_scalar_data(changed, reference_signature=reference_signature)


def test_cifti_signature_covers_both_pconn_axes() -> None:
    row_axis = make_parcels_axis(['row-a', 'row-b'])
    column_axis = make_parcels_axis(['column-a', 'column-b'])
    changed_column_axis = make_parcels_axis(['column-a', 'changed-column'])
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    reference = nb.Cifti2Image(
        values,
        header=nb.cifti2.Cifti2Header.from_axes((row_axis, column_axis)),
    )
    changed = nb.Cifti2Image(
        values,
        header=nb.cifti2.Cifti2Header.from_axes((row_axis, changed_column_axis)),
    )

    assert cifti_spatial_signature(reference) != cifti_spatial_signature(changed)


def test_load_cohort_cifti_reuses_first_decoded_row_and_cheap_validation(monkeypatch) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    brain_axis = BrainModelAxis.from_mask(mask)

    class _CountingImage:
        def __init__(self, scalar_label: str, values: list[float]) -> None:
            self.header = nb.cifti2.Cifti2Header.from_axes(
                (ScalarAxis([scalar_label]), brain_axis)
            )
            self.ndim = 2
            self._values = np.asarray([values], dtype=np.float64)
            self.decode_count = 0

        def get_fdata(self, dtype=np.float64, caching='fill'):
            self.decode_count += 1
            return self._values.astype(dtype, copy=True)

    images = {
        'first.dscalar.nii': _CountingImage('FA', [1.0, 2.0]),
        'second.dscalar.nii': _CountingImage('different-scalar-label', [3.0, 4.0]),
    }
    open_counts = dict.fromkeys(images, 0)

    def _load(source, *, cifti=False):
        assert cifti is True
        open_counts[source] += 1
        return images[source]

    validation_calls = []
    real_extract = cifti_utils.extract_cifti_scalar_data

    def _extract(*args, **kwargs):
        validation_calls.append(kwargs.get('reference_signature'))
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(cifti_utils, 'load_nibabel', _load)
    monkeypatch.setattr(cifti_utils, 'extract_cifti_scalar_data', _extract)
    cohort = pd.DataFrame(
        {
            'scalar_name': ['FA', 'FA'],
            'source_file': ['first.dscalar.nii', 'second.dscalar.nii'],
        }
    )

    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')
    matrix = scalars['FA']

    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert matrix.flags.c_contiguous
    assert open_counts == {'first.dscalar.nii': 1, 'second.dscalar.nii': 1}
    assert images['first.dscalar.nii'].decode_count == 1
    assert images['second.dscalar.nii'].decode_count == 1
    assert validation_calls[0] is None
    assert validation_calls[1] == scalars.reference_signature


def test_signature_validated_extract_skips_full_name_comparison(monkeypatch) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    image = make_dscalar(mask, np.array([1.0, 2.0], dtype=np.float32))
    signature = cifti_spatial_signature(image)
    _, names = extract_cifti_scalar_data(image)

    def _unexpected_comparison(*_args, **_kwargs):
        raise AssertionError('compact signature should replace full label comparison')

    monkeypatch.setattr(cifti_utils.np, 'array_equal', _unexpected_comparison)
    values, returned_names = extract_cifti_scalar_data(
        image,
        reference_brain_names=names,
        reference_signature=signature,
        spatial_signature=signature,
    )

    np.testing.assert_allclose(values, [1.0, 2.0])
    assert returned_names is names


def test_load_cohort_cifti_releases_image_when_signature_validation_fails(monkeypatch) -> None:
    first_mask = np.zeros((2, 2, 2), dtype=bool)
    first_mask[0, 0, 0] = True
    first_mask[0, 1, 1] = True
    changed_mask = np.zeros_like(first_mask)
    changed_mask[0, 0, 0] = True
    changed_mask[1, 1, 1] = True
    images = {
        'first.dscalar.nii': make_dscalar(first_mask, np.array([1.0, 2.0], dtype=np.float32)),
        'changed.dscalar.nii': make_dscalar(changed_mask, np.array([3.0, 4.0], dtype=np.float32)),
    }
    released = []
    monkeypatch.setattr(cifti_utils, 'load_nibabel', lambda source, **_kwargs: images[source])
    monkeypatch.setattr(cifti_utils, 'release_image_buffers', released.append)
    cohort = pd.DataFrame(
        {
            'scalar_name': ['FA', 'FA'],
            'source_file': ['first.dscalar.nii', 'changed.dscalar.nii'],
        }
    )

    scalars, _ = load_cohort_cifti(cohort)
    with pytest.raises(cifti_utils.SourceLoadError, match=r'changed\.dscalar\.nii'):
        _ = scalars['FA']

    assert released == [images['first.dscalar.nii'], images['changed.dscalar.nii']]


def _toy_mask() -> np.ndarray:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = mask[1, 1, 1] = mask[2, 2, 2] = mask[3, 3, 3] = True
    return mask


@pytest.mark.parametrize('workers', [1, 2])
def test_load_cohort_cifti_raw_fast_path_matches_nibabel(tmp_path, monkeypatch, workers) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    truth = []
    rows = []
    for i in range(4):
        values = np.arange(n, dtype=np.float32) + i * 10.0
        if i == 2:
            values[1] = np.nan  # NaN positions must survive the raw read
        path = _write_dscalar(tmp_path / f'sub{i}.dscalar.nii', mask, values)
        rows.append({'scalar_name': 'FA', 'source_file': path})
        truth.append(values)
    cohort = pd.DataFrame(rows)

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=workers, storage_dtype='float32')
    matrix = scalars['FA']

    np.testing.assert_array_equal(matrix, np.stack(truth))
    assert matrix.flags.c_contiguous
    # The three non-reference subjects are read straight from their data block.
    assert {f'sub{i}.dscalar.nii' for i in (1, 2, 3)} <= {
        source.split('/')[-1] for source in raw_reads
    }


def test_load_cohort_cifti_raw_path_handles_per_scalar_offsets(tmp_path, monkeypatch) -> None:
    # A longer map name lengthens the CIFTI XML extension and shifts vox_offset,
    # so the raw plan must be rebuilt per scalar rather than shared globally.
    mask = _toy_mask()
    n = int(mask.sum())
    rows = []
    truth = {}
    for scalar, map_name in (('thickness', 'thickness'), ('area', 'desc-fwhm25_area_long_map')):
        truth[scalar] = []
        for i in range(3):
            values = np.arange(n, dtype=np.float32) + hash(scalar) % 7 + i
            path = _write_dscalar(
                tmp_path / f'{scalar}_sub{i}.dscalar.nii', mask, values, map_name=map_name
            )
            rows.append({'scalar_name': scalar, 'source_file': path})
            truth[scalar].append(values)
    cohort = pd.DataFrame(rows)

    offsets = {nb.load(row['source_file']).dataobj.offset for row in rows}
    assert len(offsets) >= 2  # the two scalars really do have different offsets

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')
    for scalar in ('thickness', 'area'):
        np.testing.assert_array_equal(scalars[scalar], np.stack(truth[scalar]))
    # Both scalars used the fast path despite differing offsets.
    assert any('thickness_sub' in source for source in raw_reads)
    assert any('area_sub' in source for source in raw_reads)


def test_load_cohort_cifti_raw_path_tolerates_metadata_only_offset_shift(
    tmp_path, monkeypatch
) -> None:
    # Production dscalars embed per-subject provenance in the CIFTI metadata, so
    # vox_offset differs between files that share one brain-model axis. The raw
    # path must still serve them (verified by the brain-model map digest) and
    # reproduce the nibabel values exactly.
    mask = _toy_mask()
    n = int(mask.sum())
    rows = []
    truth = []
    for i in range(3):
        values = np.arange(n, dtype=np.float32) + i
        metadata = None if i != 1 else {'Provenance': f'job-{i}-sub-{i:05d}-' + 'x' * 40}
        path = _write_dscalar(tmp_path / f'sub{i}.dscalar.nii', mask, values, metadata=metadata)
        rows.append({'scalar_name': 'FA', 'source_file': path})
        truth.append(values)
    cohort = pd.DataFrame(rows)

    offsets = [nb.load(row['source_file']).dataobj.offset for row in rows]
    assert offsets[1] != offsets[0]  # the offset shift is real

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')

    np.testing.assert_array_equal(scalars['FA'], np.stack(truth))
    # Every non-reference source went through the fast path despite the shift.
    assert any('sub1.dscalar.nii' in source for source in raw_reads)
    assert any('sub2.dscalar.nii' in source for source in raw_reads)
    assert scalars.stats['FA'].fast_path_rows == 2


def test_load_cohort_cifti_raw_path_declines_brain_model_mismatch(tmp_path, monkeypatch) -> None:
    # A source with the same element count but a different brain-model axis has
    # a different MatrixIndicesMap, so the raw path must decline it; the nibabel
    # path then validates the spatial axis and rejects the inconsistent file.
    mask = _toy_mask()
    other = np.zeros_like(mask)
    other[0, 0, 1] = other[1, 1, 2] = other[2, 2, 3] = other[3, 3, 0] = True  # same count
    n = int(mask.sum())
    rows = []
    for i in range(3):
        values = np.arange(n, dtype=np.float32) + i
        path = _write_dscalar(tmp_path / f'sub{i}.dscalar.nii', other if i == 1 else mask, values)
        rows.append({'scalar_name': 'FA', 'source_file': path})
    cohort = pd.DataFrame(rows)

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    with pytest.raises(cifti_utils.SourceLoadError, match='Inconsistent spatial axis'):
        scalars['FA']
    assert not any('sub1.dscalar.nii' in source for source in raw_reads)


def test_load_cohort_cifti_raw_path_disabled_by_env(tmp_path, monkeypatch) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    rows = []
    truth = []
    for i in range(3):
        values = np.arange(n, dtype=np.float32) + i
        path = _write_dscalar(tmp_path / f'sub{i}.dscalar.nii', mask, values)
        rows.append({'scalar_name': 'FA', 'source_file': path})
        truth.append(values)
    cohort = pd.DataFrame(rows)

    monkeypatch.setenv('MODELARRAYIO_CIFTI_RAW', '0')
    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')

    np.testing.assert_array_equal(scalars['FA'], np.stack(truth))
    assert raw_reads == []  # every source decoded through nibabel


def test_load_cohort_cifti_pconn_row_major_values_are_correct(tmp_path) -> None:
    # pconn stores a 2-D matrix flattened row-major. The raw plan self-validates
    # against the decoded reference row, so it only engages when the on-disk
    # order reproduces ``get_fdata().ravel()`` exactly; either way the staged
    # values must match the flattened matrices.
    parcel_names = ['a', 'b', 'c']
    rows = []
    truth = []
    for i in range(2):
        matrix = np.arange(9, dtype=np.float32).reshape(3, 3) + i
        path = str(tmp_path / f'sub{i}.pconn.nii')
        make_pconn(parcel_names, matrix).to_filename(path)
        rows.append({'scalar_name': 'conn', 'source_file': path})
        truth.append(matrix.ravel())
    cohort = pd.DataFrame(rows)

    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')

    np.testing.assert_array_equal(scalars['conn'], np.stack(truth))


def test_read_cifti_row_raw_rejects_non_matching_sources(tmp_path) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'ref.dscalar.nii', mask, np.arange(n, dtype=np.float32))
    ref_image = nb.load(reference)
    ref_row = ref_image.get_fdata(dtype=np.float32, caching='unchanged').reshape(-1)
    plan = _build_cifti_raw_read_plan(ref_image, ref_row, reference, 'float32')
    assert plan is not None

    # An s3:// URI and a .gz path are declined outright.
    assert _read_cifti_row_raw('s3://bucket/key.dscalar.nii', plan) is None
    assert _read_cifti_row_raw(f'{reference}.gz', plan) is None

    # A truncated copy (wrong size) is declined.
    truncated = tmp_path / 'short.dscalar.nii'
    truncated.write_bytes(b'\x00' * (plan.expected_file_size - 8))
    assert _read_cifti_row_raw(str(truncated), plan) is None

    # A missing file is declined.
    assert _read_cifti_row_raw(str(tmp_path / 'missing.dscalar.nii'), plan) is None

    # The reference itself round-trips through the fast path.
    np.testing.assert_array_equal(_read_cifti_row_raw(reference, plan), ref_row)


def test_build_cifti_raw_read_plan_rejects_in_memory_image() -> None:
    # An in-memory image has no on-disk data offset, so no plan can be built.
    image = make_dscalar(_toy_mask(), np.arange(int(_toy_mask().sum()), dtype=np.float32))
    row = image.get_fdata(dtype=np.float32).reshape(-1)
    assert _build_cifti_raw_read_plan(image, row, 'memory.dscalar.nii', 'float32') is None


def test_load_cohort_cifti_raw_fast_path_serves_pscalar(tmp_path, monkeypatch) -> None:
    # Parcellated scalars store a ParcelsAxis map at dimension 1, so the raw
    # plan must engage for them exactly as it does for dense greyordinates.
    parcel_names = [f'p{i}' for i in range(6)]
    truth = []
    rows = []
    for i in range(4):
        values = np.arange(len(parcel_names), dtype=np.float32) + i * 10.0
        path = str(tmp_path / f'sub{i}.pscalar.nii')
        make_pscalar(parcel_names, values).to_filename(path)
        rows.append({'scalar_name': 'thickness', 'source_file': path})
        truth.append(values)
    cohort = pd.DataFrame(rows)

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, names = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')
    matrix = scalars['thickness']

    np.testing.assert_array_equal(matrix, np.stack(truth))
    assert list(names) == parcel_names
    assert {f'sub{i}.pscalar.nii' for i in (1, 2, 3)} <= {
        source.split('/')[-1] for source in raw_reads
    }
    assert scalars.stats['thickness'].fast_path_rows == 3


def test_load_cohort_cifti_rows_match_build_scalar_sources(tmp_path) -> None:
    # The loader and the CLI's column_names must be built from the same rows,
    # so a blank source cell is skipped by both rather than failing late.
    from modelarrayio.utils.misc import build_scalar_sources

    mask = _toy_mask()
    n = int(mask.sum())
    rows = []
    for i in range(2):
        path = _write_dscalar(tmp_path / f'sub{i}.dscalar.nii', mask, np.full(n, float(i)))
        rows.append({'scalar_name': 'FA', 'source_file': path})
    rows.append({'scalar_name': 'FA', 'source_file': ''})
    cohort = pd.DataFrame(rows)

    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    matrix = scalars['FA']

    expected_sources = build_scalar_sources(cohort)['FA']
    assert matrix.shape[0] == len(expected_sources) == 2
    assert list(scalars.sources_by_scalar['FA']) == expected_sources


def test_read_cifti_row_raw_reads_file_once(tmp_path, monkeypatch) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'ref.dscalar.nii', mask, np.arange(n, dtype=np.float32))
    other = _write_dscalar(
        tmp_path / 'other.dscalar.nii', mask, np.arange(n, dtype=np.float32) + 1
    )
    ref_image = nb.load(reference)
    ref_row = ref_image.get_fdata(dtype=np.float32).reshape(-1)
    plan = _build_cifti_raw_read_plan(ref_image, ref_row, reference, 'float32')
    assert plan is not None

    opened: list[str] = []
    real_open = open

    def _counting_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes)) or hasattr(file, '__fspath__'):
            opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr('builtins.open', _counting_open)
    row = _read_cifti_row_raw(other, plan)
    np.testing.assert_array_equal(row, np.arange(n, dtype=np.float32) + 1)
    assert opened.count(other) == 1
