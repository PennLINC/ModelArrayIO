"""Unit tests for HDF5 storage helpers."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from modelarrayio.storage.h5_storage import (
    compute_chunk_shape_full_subjects,
    create_scalar_matrix_dataset,
    resolve_compression,
    resolve_dtype,
    write_rows_in_column_stripes,
)


def test_resolve_dtype() -> None:
    assert resolve_dtype('float32') == np.float32
    assert resolve_dtype('float64') == np.float64
    assert resolve_dtype('FLOAT32') == np.float32
    with pytest.raises(ValueError, match='Unsupported storage dtype'):
        resolve_dtype('unknown')


def test_resolve_compression_gzip_and_none() -> None:
    comp, level, shuffle = resolve_compression('gzip', 9, True)
    assert comp == 'gzip'
    assert level == 9
    assert shuffle is True

    comp, level, shuffle = resolve_compression('none', 4, True)
    assert comp is None
    assert shuffle is False

    comp, level, shuffle = resolve_compression('lzf', 4, False)
    assert comp == 'lzf'
    assert shuffle is False


def test_resolve_compression_invalid_gzip_level_falls_back() -> None:
    comp, level, _shuffle = resolve_compression('gzip', 'bad', True)
    assert comp == 'gzip'
    assert level == 4


def test_compute_chunk_shape_full_subjects() -> None:
    chunk = compute_chunk_shape_full_subjects(
        n_files=3,
        n_elements=100,
        item_chunk=10,
        target_chunk_mb=2.0,
        storage_np_dtype=np.float32,
    )
    assert chunk == (3, 10)


def test_compute_chunk_shape_auto_item_chunk() -> None:
    chunk = compute_chunk_shape_full_subjects(
        n_files=2,
        n_elements=50,
        item_chunk=0,
        target_chunk_mb=1.0,
        storage_np_dtype=np.float32,
    )
    assert chunk[0] == 2
    assert 1 <= chunk[1] <= 50


@pytest.mark.parametrize(
    ('n_files', 'n_elements'),
    [(0, 10), (3, 0), (0, 0)],
)
def test_compute_chunk_shape_rejects_zero_dimension(n_files: int, n_elements: int) -> None:
    with pytest.raises(ValueError, match='zero-length'):
        compute_chunk_shape_full_subjects(
            n_files,
            n_elements,
            item_chunk=0,
            target_chunk_mb=2.0,
            storage_np_dtype=np.float32,
        )


def test_create_scalar_matrix_dataset_writes_data_and_attrs(tmp_path) -> None:
    path = tmp_path / 'out.h5'
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    with h5py.File(path, 'w') as h5:
        create_scalar_matrix_dataset(
            h5,
            'scalars/FA/values',
            data,
            ['sub-01', 'sub-02'],
            compression='none',
            shuffle=False,
        )
    with h5py.File(path, 'r') as h5:
        dset = h5['scalars/FA/values']
        np.testing.assert_array_equal(dset[...], data)
        assert list(dset.attrs['column_names']) == ['sub-01', 'sub-02']


def test_write_rows_in_column_stripes_matches_dense_write(tmp_path) -> None:
    """Stripe writer should match assigning the full matrix."""
    path = tmp_path / 'stripe.h5'
    n_files, num_elements = 3, 17
    full = np.arange(n_files * num_elements, dtype=np.float64).reshape(n_files, num_elements)
    rows = [full[i].copy() for i in range(n_files)]

    with h5py.File(path, 'w') as h5:
        dset = h5.create_dataset('m', shape=(n_files, num_elements), dtype='f8', chunks=(3, 5))
        write_rows_in_column_stripes(dset, rows)

    with h5py.File(path, 'r') as h5:
        np.testing.assert_array_equal(h5['m'][...], full)


def test_write_rows_in_column_stripes_length_mismatch_raises(tmp_path) -> None:
    with h5py.File(tmp_path / 'x.h5', 'w') as h5:
        dset = h5.create_dataset('m', shape=(2, 4), dtype='f4')
    with h5py.File(tmp_path / 'x.h5', 'a') as h5:
        dset = h5['m']
        with pytest.raises(ValueError, match='rows length'):
            write_rows_in_column_stripes(dset, [np.zeros(4)])
