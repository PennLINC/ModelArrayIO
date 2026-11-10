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
    assert scalars.stats['FA'].fast_path_rows == 3


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
    # pconn rows are the matrix flattened row-major. make_pconn shares one axis
    # for rows and columns (one "0,1" map); the fast path must reorder the
    # column-major data block for it too. Asymmetric matrices expose any
    # transpose.
    parcel_names = ['a', 'b', 'c']
    rows = []
    truth = []
    for i in range(2):
        matrix = np.arange(9, dtype=np.float32).reshape(3, 3) * (i + 1) + np.triu(np.ones((3, 3)))
        path = str(tmp_path / f'sub{i}.pconn.nii')
        make_pconn(parcel_names, matrix).to_filename(path)
        rows.append({'scalar_name': 'conn', 'source_file': path})
        truth.append(matrix.ravel())
    cohort = pd.DataFrame(rows)

    scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')

    np.testing.assert_array_equal(scalars['conn'], np.stack(truth))
    assert scalars.stats['conn'].fast_path_rows == 1


def _write_pconn(path, row_names, column_names, matrix):
    """Write a pconn with distinct row and column parcel axes (two separate maps)."""
    header = nb.cifti2.Cifti2Header.from_axes(
        (make_parcels_axis(list(row_names)), make_parcels_axis(list(column_names)))
    )
    nb.Cifti2Image(np.asarray(matrix, dtype=np.float32), header=header).to_filename(str(path))
    return str(path)


def test_load_cohort_cifti_pconn_asymmetric_follower_matches_nibabel(
    tmp_path, monkeypatch
) -> None:
    # Regression: a symmetric reference matrix used to pass the raw plan's
    # reference probe, after which every non-symmetric subject was stored
    # transposed. The fast path must read the column-major block in order.
    rows, cols = list('ABCD'), list('WXYZ')
    asym = np.arange(16, dtype=np.float32).reshape(4, 4)
    sym = asym + asym.T
    paths = [
        _write_pconn(tmp_path / 'sub0.pconn.nii', rows, cols, sym),
        _write_pconn(tmp_path / 'sub1.pconn.nii', rows, cols, asym),
    ]
    cohort = pd.DataFrame({'scalar_name': 'conn', 'source_file': paths})

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    matrix = scalars['conn']

    np.testing.assert_array_equal(matrix[0], sym.ravel())
    np.testing.assert_array_equal(matrix[1], asym.ravel())
    np.testing.assert_array_equal(matrix[1], nb.load(paths[1]).get_fdata(dtype=np.float32).ravel())
    assert paths[1] in raw_reads
    assert scalars.stats['conn'].fast_path_rows == 1


@pytest.mark.parametrize('shape', [(3, 5), (5, 3)])
def test_load_cohort_cifti_pconn_fast_path_handles_non_square_matrices(
    tmp_path, monkeypatch, shape
) -> None:
    rows, cols = [f'r{i}' for i in range(shape[0])], [f'c{i}' for i in range(shape[1])]
    matrices = [
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * (i + 1) for i in range(3)
    ]
    paths = [
        _write_pconn(tmp_path / f'sub{i}.pconn.nii', rows, cols, matrix)
        for i, matrix in enumerate(matrices)
    ]
    cohort = pd.DataFrame({'scalar_name': 'conn', 'source_file': paths})

    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    np.testing.assert_array_equal(scalars['conn'], np.stack([m.ravel() for m in matrices]))
    assert scalars.stats['conn'].fast_path_rows == 2


def test_load_cohort_cifti_pconn_rejects_permuted_row_parcels(tmp_path, monkeypatch) -> None:
    # Regression: the raw path only hashed the dimension-1 map, so a subject
    # whose row (dimension-0) parcels were permuted was silently accepted.
    rows, cols = list('ABCD'), list('WXYZ')
    asym = np.arange(16, dtype=np.float32).reshape(4, 4)
    sym = asym + asym.T
    paths = [
        _write_pconn(tmp_path / 'sub0.pconn.nii', rows, cols, sym),
        _write_pconn(tmp_path / 'sub1.pconn.nii', rows, cols, asym),
        _write_pconn(tmp_path / 'sub2.pconn.nii', rows[::-1], cols, sym),
    ]
    cohort = pd.DataFrame({'scalar_name': 'conn', 'source_file': paths})

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    with pytest.raises(cifti_utils.SourceLoadError, match='Inconsistent spatial axis'):
        scalars['conn']
    assert paths[2] not in raw_reads


def test_load_cohort_cifti_pconn_rejects_permuted_column_parcels(tmp_path, monkeypatch) -> None:
    rows, cols = list('ABCD'), list('WXYZ')
    asym = np.arange(16, dtype=np.float32).reshape(4, 4)
    paths = [
        _write_pconn(tmp_path / 'sub0.pconn.nii', rows, cols, asym),
        _write_pconn(tmp_path / 'sub1.pconn.nii', rows, cols[::-1], asym),
    ]
    cohort = pd.DataFrame({'scalar_name': 'conn', 'source_file': paths})

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    with pytest.raises(cifti_utils.SourceLoadError, match='Inconsistent spatial axis'):
        scalars['conn']
    assert paths[1] not in raw_reads


def test_to_modelarray_pconn_regression_end_to_end(tmp_path, monkeypatch) -> None:
    # The review repro: the CLI must reject the permuted subject and store an
    # asymmetric follower in row-major order.
    import h5py

    from modelarrayio.cli.to_modelarray import to_modelarray

    rows, cols = list('ABCD'), list('WXYZ')
    asym = np.arange(16, dtype=np.float32).reshape(4, 4)
    sym = asym + asym.T
    good = [
        _write_pconn(tmp_path / 'sub0.pconn.nii', rows, cols, sym),
        _write_pconn(tmp_path / 'sub1.pconn.nii', rows, cols, asym),
    ]
    permuted = _write_pconn(tmp_path / 'sub2.pconn.nii', rows[::-1], cols, sym)

    bad_cohort = tmp_path / 'bad.csv'
    pd.DataFrame({'scalar_name': 'conn', 'source_file': [*good, permuted]}).to_csv(
        bad_cohort, index=False
    )
    with pytest.raises(cifti_utils.SourceLoadError, match='Inconsistent spatial axis'):
        to_modelarray(bad_cohort, output=tmp_path / 'bad.h5')
    assert not (tmp_path / 'bad.h5').exists()

    good_cohort = tmp_path / 'good.csv'
    pd.DataFrame({'scalar_name': 'conn', 'source_file': good}).to_csv(good_cohort, index=False)
    to_modelarray(good_cohort, output=tmp_path / 'good.h5')
    with h5py.File(tmp_path / 'good.h5', 'r') as h5_file:
        np.testing.assert_array_equal(h5_file['scalars/conn/values'][1], asym.ravel())


def test_raw_read_of_pconn_matches_nibabel_order(tmp_path) -> None:
    # The on-disk block is column-major; the raw read must return nibabel's
    # row-major flattening, whatever the reference matrix looks like.
    rows, cols = list('ABC'), list('WXYZ')
    reference = _write_pconn(tmp_path / 'ref.pconn.nii', rows, cols, np.ones((3, 4)))
    follower_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    follower = _write_pconn(tmp_path / 'f.pconn.nii', rows, cols, follower_matrix)
    plan = _build_cifti_raw_read_plan(
        nb.load(reference).get_fdata(dtype=np.float32).ravel(), reference, 'float32'
    )
    assert plan is not None
    assert plan.matrix_shape == (3, 4)
    raw = _read_cifti_row_raw(follower, plan)
    np.testing.assert_array_equal(raw, follower_matrix.ravel())
    np.testing.assert_array_equal(raw, nb.load(follower).get_fdata(dtype=np.float32).ravel())


def _cifti_xml(*maps: tuple[str, str, str]) -> bytes:
    """Return a minimal CIFTI XML prefix with the given (dimension, type, body) maps."""
    body = ''.join(
        f'<MatrixIndicesMap AppliesToMatrixDimension="{dimension}" '
        f'IndicesMapToDataType="{index_type}">{inner}</MatrixIndicesMap>'
        for dimension, index_type, inner in maps
    )
    return f'HEADER<CIFTI Version="2"><Matrix>{body}</Matrix></CIFTI>'.encode()


_SCALARS = 'CIFTI_INDEX_TYPE_SCALARS'
_BRAIN_MODELS = 'CIFTI_INDEX_TYPE_BRAIN_MODELS'
_PARCELS = 'CIFTI_INDEX_TYPE_PARCELS'


def test_layout_digest_accepts_scalar_plus_spatial_map() -> None:
    def digest(xml):
        layout = cifti_utils._layout_digest(xml)
        assert layout is not None
        assert layout[0] == 1  # exactly one matrix row
        return layout[1]

    dscalar = digest(
        _cifti_xml(('0', _SCALARS, '<NamedMap>a</NamedMap>'), ('1', _BRAIN_MODELS, 'x'))
    )
    renamed = digest(
        _cifti_xml(('0', _SCALARS, '<NamedMap>b</NamedMap>'), ('1', _BRAIN_MODELS, 'x'))
    )
    moved = digest(
        _cifti_xml(('0', _SCALARS, '<NamedMap>a</NamedMap>'), ('1', _BRAIN_MODELS, 'y'))
    )
    pscalar = digest(_cifti_xml(('0', _SCALARS, ''), ('1', _PARCELS, 'x')))

    # Scalar map names may differ per subject; the spatial map may not.
    assert renamed == dscalar
    assert moved != dscalar
    assert pscalar != dscalar


def test_layout_digest_accepts_pconn_and_covers_every_map() -> None:
    def digest(xml):
        layout = cifti_utils._layout_digest(xml)
        assert layout is not None
        assert layout[0] == 0  # any number of matrix rows
        return layout[1]

    separate = digest(_cifti_xml(('0', _PARCELS, 'rows'), ('1', _PARCELS, 'cols')))
    rows_changed = digest(_cifti_xml(('0', _PARCELS, 'ROWS'), ('1', _PARCELS, 'cols')))
    cols_changed = digest(_cifti_xml(('0', _PARCELS, 'rows'), ('1', _PARCELS, 'COLS')))
    shared = digest(_cifti_xml(('0,1', _PARCELS, 'rows')))
    shared_changed = digest(_cifti_xml(('0,1', _PARCELS, 'ROWS')))

    assert len({separate, rows_changed, cols_changed, shared, shared_changed}) == 5


@pytest.mark.parametrize(
    'head',
    [
        pytest.param(b'no cifti xml here', id='no-cifti-element'),
        pytest.param(_cifti_xml(('1', _BRAIN_MODELS, 'x')), id='missing-dimension-0-map'),
        pytest.param(_cifti_xml(('0', _SCALARS, '')), id='missing-dimension-1-map'),
        pytest.param(_cifti_xml(('0', _BRAIN_MODELS, 'r'), ('1', _BRAIN_MODELS, 'c')), id='dconn'),
        pytest.param(_cifti_xml(('0', _PARCELS, 'r'), ('1', _BRAIN_MODELS, 'c')), id='pdconn'),
        pytest.param(_cifti_xml(('0,1', _BRAIN_MODELS, 'p')), id='shared-dense-map'),
        pytest.param(_cifti_xml(('0', _SCALARS, ''), ('1', _SCALARS, '')), id='scalar-scalar'),
        pytest.param(
            _cifti_xml(('0', 'CIFTI_INDEX_TYPE_LABELS', ''), ('1', _BRAIN_MODELS, 'x')),
            id='label-map',
        ),
        pytest.param(
            _cifti_xml(('0', _SCALARS, ''), ('1', 'CIFTI_INDEX_TYPE_SERIES', '')),
            id='series-spatial-map',
        ),
        pytest.param(
            _cifti_xml(('0', _SCALARS, ''), ('0', _SCALARS, ''), ('1', _BRAIN_MODELS, 'x')),
            id='duplicate-dimension',
        ),
        pytest.param(
            b'<CIFTI><MatrixIndicesMap AppliesToMatrixDimension="0" '
            b'IndicesMapToDataType="CIFTI_INDEX_TYPE_SCALARS">',
            id='unterminated-map',
        ),
        pytest.param(
            b'<CIFTI><MatrixIndicesMap IndicesMapToDataType="CIFTI_INDEX_TYPE_SCALARS">'
            b'</MatrixIndicesMap>',
            id='missing-dimension-attribute',
        ),
    ],
)
def test_layout_digest_declines_other_layouts(head) -> None:
    assert cifti_utils._layout_digest(head) is None


def test_shared_axis_pconn_is_written_as_one_map(tmp_path) -> None:
    # make_pconn uses one axis for rows and columns, which nibabel writes as a
    # single AppliesToMatrixDimension="0,1" map; the plan must accept it.
    path = str(tmp_path / 'combined.pconn.nii')
    make_pconn(['a', 'b'], np.arange(4, dtype=np.float32)).to_filename(path)
    with open(path, 'rb') as handle:
        assert b'AppliesToMatrixDimension="0,1"' in handle.read(4096)
    row = nb.load(path).get_fdata(dtype=np.float32).ravel()
    assert _build_cifti_raw_read_plan(row, path, 'float32') is not None
    assert _build_cifti_raw_read_plan(row, str(tmp_path / 'missing.nii'), 'float32') is None


def _reference_plan(tmp_path, mask, values):
    reference = _write_dscalar(tmp_path / 'ref.dscalar.nii', mask, values)
    image = nb.load(reference)
    row = image.get_fdata(dtype=np.float32).reshape(-1)
    plan = _build_cifti_raw_read_plan(row, reference, 'float32')
    assert plan is not None
    return reference, plan


def test_read_cifti_row_raw_declines_dtype_dims_scaling_and_truncation(tmp_path) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    _reference, plan = _reference_plan(tmp_path, mask, np.arange(n, dtype=np.float32))

    # Different on-disk dtype.
    header = nb.cifti2.Cifti2Header.from_axes((ScalarAxis(['m']), BrainModelAxis.from_mask(mask)))
    float64_path = str(tmp_path / 'float64.dscalar.nii')
    nb.Cifti2Image(np.arange(n, dtype=np.float64).reshape(1, -1), header=header).to_filename(
        float64_path
    )
    assert _read_cifti_row_raw(float64_path, plan) is None

    # Different element count (dims).
    bigger = _toy_mask()
    bigger[0, 1, 2] = True
    dims_path = _write_dscalar(
        tmp_path / 'dims.dscalar.nii', bigger, np.arange(n + 1, dtype=np.float32)
    )
    assert _read_cifti_row_raw(dims_path, plan) is None

    # On-disk scaling the reference does not have.
    scaled_path = _write_dscalar(
        tmp_path / 'scaled.dscalar.nii', mask, np.arange(n, dtype=np.float32)
    )
    with open(scaled_path, 'r+b') as handle:
        nifti_header = nb.Nifti2Header.from_fileobj(handle, check=False)
        nifti_header.set_slope_inter(2.0, 1.0)
        handle.seek(0)
        handle.write(nifti_header.binaryblock)
    assert _read_cifti_row_raw(scaled_path, plan) is None

    # A real file cut short inside its data block.
    truncated_path = _write_dscalar(
        tmp_path / 'truncated.dscalar.nii', mask, np.arange(n, dtype=np.float32)
    )
    with open(truncated_path, 'r+b') as handle:
        handle.truncate(nb.load(truncated_path).dataobj.offset + 4)
    assert _read_cifti_row_raw(truncated_path, plan) is None


def test_load_cohort_cifti_falls_back_for_mismatched_followers(tmp_path, monkeypatch) -> None:
    # Followers the raw path declines are still decoded by nibabel and stored
    # correctly; only matching followers count as fast-path rows.
    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'sub0.dscalar.nii', mask, np.arange(n) + 0.0)
    matching = _write_dscalar(tmp_path / 'sub1.dscalar.nii', mask, np.arange(n) + 1.0)
    header = nb.cifti2.Cifti2Header.from_axes((ScalarAxis(['m']), BrainModelAxis.from_mask(mask)))
    float64_path = str(tmp_path / 'sub2.dscalar.nii')
    nb.Cifti2Image((np.arange(n) + 2.0).reshape(1, -1), header=header).to_filename(float64_path)
    cohort = pd.DataFrame(
        {'scalar_name': 'FA', 'source_file': [reference, matching, float64_path]}
    )

    raw_reads = _spy_raw_reads(monkeypatch)
    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    matrix = scalars['FA']

    np.testing.assert_array_equal(
        matrix, np.stack([np.arange(n) + offset for offset in (0.0, 1.0, 2.0)])
    )
    # The reference itself is read once as the plan's probe.
    assert raw_reads == [reference, matching]
    assert scalars.stats['FA'].fast_path_rows == 1


def test_load_cohort_cifti_rejects_label_follower(tmp_path) -> None:
    # A dlabel-style follower (label map on dimension 0) with the same brain
    # models must not be read as a scalar row.
    from nibabel.cifti2.cifti2_axes import LabelAxis

    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'sub0.dscalar.nii', mask, np.arange(n) + 0.0)
    label_header = nb.cifti2.Cifti2Header.from_axes(
        (LabelAxis(['labels'], {0: ('???', (0, 0, 0, 0))}), BrainModelAxis.from_mask(mask))
    )
    label_path = str(tmp_path / 'sub1.dscalar.nii')
    nb.Cifti2Image(np.zeros((1, n), dtype=np.float32), header=label_header).to_filename(label_path)
    cohort = pd.DataFrame({'scalar_name': 'FA', 'source_file': [reference, label_path]})

    scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
    with pytest.raises(cifti_utils.SourceLoadError):
        scalars['FA']


def test_read_cifti_row_raw_rejects_non_matching_sources(tmp_path) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'ref.dscalar.nii', mask, np.arange(n, dtype=np.float32))
    ref_image = nb.load(reference)
    ref_row = ref_image.get_fdata(dtype=np.float32, caching='unchanged').reshape(-1)
    plan = _build_cifti_raw_read_plan(ref_row, reference, 'float32')
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
    assert _build_cifti_raw_read_plan(row, 'memory.dscalar.nii', 'float32') is None


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
    plan = _build_cifti_raw_read_plan(ref_row, reference, 'float32')
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


def test_build_cifti_raw_read_plan_declines_ineligible_references(tmp_path) -> None:
    mask = _toy_mask()
    n = int(mask.sum())
    reference = _write_dscalar(tmp_path / 'ref.dscalar.nii', mask, np.arange(n, dtype=np.float32))
    image = nb.load(reference)
    row = image.get_fdata(dtype=np.float32).reshape(-1)

    # Remote and compressed sources never get a plan.
    assert _build_cifti_raw_read_plan(row, 's3://bucket/ref.dscalar.nii', 'float32') is None
    assert _build_cifti_raw_read_plan(row, f'{reference}.gz', 'float32') is None
    # A row that does not match the on-disk dims is declined.
    assert _build_cifti_raw_read_plan(row[:-1], reference, 'float32') is None
    assert _build_cifti_raw_read_plan(row.reshape(1, -1), reference, 'float32') is None
    # The final reference probe must reproduce the decoded row exactly.
    assert _build_cifti_raw_read_plan(row + 1.0, reference, 'float32') is None

    # Integer on-disk data is left to nibabel.
    header = nb.cifti2.Cifti2Header.from_axes((ScalarAxis(['m']), BrainModelAxis.from_mask(mask)))
    int_path = str(tmp_path / 'int.dscalar.nii')
    nb.Cifti2Image(np.arange(n, dtype=np.int16).reshape(1, -1), header=header).to_filename(
        int_path
    )
    int_image = nb.load(int_path)
    int_row = int_image.get_fdata(dtype=np.float32).reshape(-1)
    assert _build_cifti_raw_read_plan(int_row, int_path, 'float32') is None


def _set_slope_inter(path, slope, inter) -> None:
    with open(path, 'r+b') as handle:
        nifti_header = nb.Nifti2Header.from_fileobj(handle, check=False)
        nifti_header.set_slope_inter(slope, inter)
        handle.seek(0)
        handle.write(nifti_header.binaryblock)


def test_load_cohort_cifti_scaled_sources_use_nibabel(tmp_path, monkeypatch) -> None:
    # The raw path serves unscaled data only. A scaled reference gets no plan,
    # and a scaled follower of an unscaled reference is declined; both are
    # decoded (and scaled) by nibabel.
    mask = _toy_mask()
    n = int(mask.sum())

    def _cohort(scaled):
        paths = []
        for i, is_scaled in enumerate(scaled):
            path = _write_dscalar(
                tmp_path / f'{"".join(map(str, map(int, scaled)))}_sub{i}.dscalar.nii',
                mask,
                np.arange(n, dtype=np.float32) + i,
            )
            if is_scaled:
                _set_slope_inter(path, 2.0, 0.5)
            paths.append(path)
        return paths

    for scaled, expected_fast_rows in (((True, True, True), 0), ((False, True, False), 1)):
        paths = _cohort(scaled)
        cohort = pd.DataFrame({'scalar_name': 'FA', 'source_file': paths})
        raw_reads = _spy_raw_reads(monkeypatch)
        scalars, _ = load_cohort_cifti(cohort, workers=1, storage_dtype='float32')
        matrix = scalars['FA']

        expected = np.stack([nb.load(path).get_fdata(dtype=np.float32).ravel() for path in paths])
        np.testing.assert_array_equal(matrix, expected)
        if scaled[1]:
            np.testing.assert_array_equal(matrix[1], (np.arange(n) + 1) * 2.0 + 0.5)
        assert not [
            path
            for path, is_scaled in zip(paths, scaled, strict=True)
            if is_scaled and path in raw_reads
        ]
        assert scalars.stats['FA'].fast_path_rows == expected_fast_rows


@pytest.mark.parametrize('kind', ['dscalar', 'pscalar', 'pconn'])
def test_fast_path_followers_never_touch_nibabel(tmp_path, monkeypatch, kind) -> None:
    # Per-file nibabel decoding (XML parse + axis rebuild) is what made large
    # CIFTI cohorts unusable. Only each scalar's first source may be decoded;
    # nibabel work must not grow with the number of followers.
    mask = _toy_mask()
    n = int(mask.sum())
    parcel_names = [f'p{i}' for i in range(n)]

    def _write(path, values):
        if kind == 'dscalar':
            return _write_dscalar(path, mask, values)
        if kind == 'pconn':
            matrix = np.outer(values, np.arange(1, 4, dtype=np.float32))
            return _write_pconn(path, parcel_names, ['x', 'y', 'z'], matrix)
        make_pscalar(parcel_names, values).to_filename(str(path))
        return str(path)

    def _stage(n_sources):
        paths = [
            _write(tmp_path / f'{n_sources}_sub{i}.{kind}.nii', np.arange(n, dtype=np.float32) + i)
            for i in range(n_sources)
        ]
        cohort = pd.DataFrame({'scalar_name': 'FA', 'source_file': paths})
        loads: list[str] = []
        real_load = cifti_utils.load_nibabel
        monkeypatch.setattr(
            cifti_utils,
            'load_nibabel',
            lambda source, **kwargs: (loads.append(str(source)), real_load(source, **kwargs))[1],
        )
        axis_builds = []
        real_get_axis = nb.cifti2.Cifti2Header.get_axis
        monkeypatch.setattr(
            nb.cifti2.Cifti2Header,
            'get_axis',
            lambda self, index: (axis_builds.append(index), real_get_axis(self, index))[1],
        )
        scalars, _ = load_cohort_cifti(cohort, workers=2, storage_dtype='float32')
        scalars['FA']
        monkeypatch.undo()
        assert loads == [paths[0]]
        assert scalars.stats['FA'].fast_path_rows == n_sources - 1
        return len(axis_builds)

    assert _stage(3) == _stage(8)
