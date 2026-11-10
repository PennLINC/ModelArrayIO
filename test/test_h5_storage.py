"""Unit tests for HDF5 storage helpers."""

from __future__ import annotations

import json
import logging
import zlib

import h5py
import numpy as np
import pytest

from modelarrayio.storage import h5_storage
from modelarrayio.storage.h5_storage import (
    compute_chunk_shape_full_subjects,
    create_scalar_matrix_dataset,
    resolve_compression,
    resolve_dtype,
    write_full_matrix,
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


def test_resolve_compression_defaults_and_none() -> None:
    assert resolve_compression('gzip', None, True) == ('gzip', 4, True)
    # At the storage level, None means "no compression", not "backend default".
    assert resolve_compression(None, 4, True) == (None, None, False)


@pytest.mark.parametrize(
    ('compression', 'level', 'message'),
    [
        ('gzip', 'bad', 'compression level must be an integer'),
        ('gzip', 12, 'gzip compression level must be between 0 and 9'),
        ('zstd', 5, "Compression 'zstd' is not supported by hdf5"),
    ],
)
def test_resolve_compression_rejects_invalid_settings(compression, level, message) -> None:
    # Shared with the CLI's resolver: invalid settings raise instead of being
    # silently clamped or replaced.
    with pytest.raises(ValueError, match=message):
        resolve_compression(compression, level, True)


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


def test_create_scalar_matrix_dataset_writes_data_and_column_names_dataset(tmp_path) -> None:
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
        assert 'column_names' not in dset.attrs
        assert h5['scalars/FA/column_names'].asstr()[...].tolist() == ['sub-01', 'sub-02']


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_parallel_direct_chunks_match_normal_hdf5_and_pad_partial_chunk(tmp_path, dtype) -> None:
    path = tmp_path / 'direct.h5'
    data = np.arange(21, dtype=dtype).reshape(3, 7)

    with h5py.File(path, 'w') as h5:
        normal = h5.create_dataset(
            'normal',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 4),
            compression='gzip',
            compression_opts=4,
            shuffle=True,
        )
        normal[...] = data
        direct = h5.create_dataset(
            'direct',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 4),
            compression='gzip',
            compression_opts=4,
            shuffle=True,
        )
        metrics = write_full_matrix(direct, data, compression_workers=2)

        np.testing.assert_array_equal(direct[...], normal[...])
        assert direct.compression == normal.compression == 'gzip'
        assert direct.compression_opts == normal.compression_opts == 4
        assert direct.shuffle is normal.shuffle is True
        assert direct.chunks == normal.chunks == (2, 4)
        assert metrics['direct_chunk'] is True
        assert metrics['fallback'] is False
        assert metrics['chunk_count'] == 4
        assert metrics['compression_cpu_seconds'] >= 0.0
        json.dumps(metrics)

        for offset in ((0, 0), (0, 4), (2, 0), (2, 4)):
            direct_mask, direct_bytes = direct.id.read_direct_chunk(offset)
            normal_mask, normal_bytes = normal.id.read_direct_chunk(offset)
            assert direct_mask == normal_mask == 0
            # Deflate streams are not guaranteed byte-for-byte stable across zlib
            # builds. Their decoded shuffled bytes are the portable equivalence.
            assert zlib.decompress(direct_bytes) == zlib.decompress(normal_bytes)

        _filter_mask, compressed = direct.id.read_direct_chunk((2, 4))
        shuffled = zlib.decompress(compressed)
        unshuffled = (
            np.frombuffer(shuffled, dtype=np.uint8).reshape(data.dtype.itemsize, -1).T.reshape(-1)
        )
        padded_chunk = unshuffled.view(data.dtype).reshape(2, 4)
        np.testing.assert_array_equal(padded_chunk[0, :3], data[2, 4:])
        np.testing.assert_array_equal(padded_chunk[0, 3], np.zeros((), dtype=data.dtype))
        np.testing.assert_array_equal(padded_chunk[1], np.zeros(4, dtype=data.dtype))


def test_no_compression_bypasses_direct_chunk_path(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'uncompressed.h5'
    data = np.arange(10, dtype=np.float32).reshape(2, 5)

    def _unexpected_direct_write(*_args, **_kwargs):
        raise AssertionError('direct chunk path should not be called')

    monkeypatch.setattr(h5_storage, '_write_direct_gzip_chunks', _unexpected_direct_write)
    with h5py.File(path, 'w') as h5:
        dataset = h5.create_dataset('values', shape=data.shape, dtype=data.dtype, chunks=(2, 3))
        metrics = write_full_matrix(dataset, data)
        np.testing.assert_array_equal(dataset[...], data)

    assert metrics['direct_chunk'] is False
    assert metrics['fallback'] is False
    assert metrics['compression_seconds'] == 0.0


def test_internal_compression_worker_environment_controls_pool(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'workers.h5'
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    monkeypatch.setenv('MODELARRAYIO_HDF5_COMPRESSION_WORKERS', '2')

    with h5py.File(path, 'w') as h5:
        dataset = h5.create_dataset(
            'values',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 2),
            compression='gzip',
            compression_opts=1,
            shuffle=True,
        )
        metrics = write_full_matrix(dataset, data)

    assert metrics['compression_workers'] == 2
    with h5py.File(path, 'r') as h5:
        np.testing.assert_array_equal(data, h5['values'][...])


@pytest.mark.parametrize('configured_workers', ['invalid', '0', '-1'])
def test_invalid_compression_worker_environment_is_not_silently_ignored(
    tmp_path, monkeypatch, configured_workers
) -> None:
    path = tmp_path / 'invalid-workers.h5'
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    monkeypatch.setenv('MODELARRAYIO_HDF5_COMPRESSION_WORKERS', configured_workers)

    with h5py.File(path, 'w') as h5:
        dataset = h5.create_dataset(
            'values',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(1, 2),
            compression='gzip',
        )
        with pytest.raises(ValueError, match='positive integer'):
            write_full_matrix(dataset, data)


def test_default_compression_workers_respect_available_cpu_count(tmp_path, monkeypatch) -> None:
    path = tmp_path / 'available-cpus.h5'
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    monkeypatch.delenv('MODELARRAYIO_HDF5_COMPRESSION_WORKERS', raising=False)
    monkeypatch.setattr(h5_storage, '_available_cpu_count', lambda: 2)

    with h5py.File(path, 'w') as h5:
        dataset = h5.create_dataset(
            'values',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(1, 2),
            compression='gzip',
        )
        metrics = write_full_matrix(dataset, data)

    assert metrics['compression_workers'] == 2


def test_column_names_are_written_beside_nested_values_dataset(tmp_path) -> None:
    path = tmp_path / 'nested.h5'
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    with h5py.File(path, 'w') as h5:
        create_scalar_matrix_dataset(
            h5,
            'scalars/group/FA/values',
            data,
            sources_list=['sub-01', 'sub-02'],
            compression=None,
        )

    with h5py.File(path, 'r') as h5:
        assert h5['scalars/group/FA/column_names'].asstr()[...].tolist() == [
            'sub-01',
            'sub-02',
        ]


def test_direct_chunk_worker_failure_logs_and_falls_back(tmp_path, monkeypatch, caplog) -> None:
    path = tmp_path / 'fallback.h5'
    data = np.arange(10, dtype=np.float64).reshape(2, 5)

    def _failed_worker(*_args, **_kwargs):
        raise zlib.error('injected compression failure')

    monkeypatch.setattr(h5_storage, '_encode_gzip_chunk', _failed_worker)
    with h5py.File(path, 'w') as h5, caplog.at_level(logging.WARNING):
        dataset = h5.create_dataset(
            'values',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 3),
            compression='gzip',
            compression_opts=1,
            shuffle=True,
        )
        metrics = write_full_matrix(dataset, data, compression_workers=2)
        np.testing.assert_array_equal(dataset[...], data)

    assert metrics['direct_chunk'] is False
    assert metrics['fallback'] is True
    assert metrics['compression_seconds'] is None
    assert metrics['fallback_reason'] == 'gzip encoding failed: injected compression failure'
    assert any('Falling back to ordinary HDF5' in record.message for record in caplog.records)


def test_direct_chunk_programming_errors_propagate(tmp_path, monkeypatch) -> None:
    # Only zlib/h5py failures fall back; a bug in the encoder must surface.
    def _buggy_worker(*_args, **_kwargs):
        raise AttributeError('injected bug')

    monkeypatch.setattr(h5_storage, '_encode_gzip_chunk', _buggy_worker)
    with h5py.File(tmp_path / 'bug.h5', 'w') as h5:
        dataset = h5.create_dataset(
            'values', shape=(2, 5), dtype='f8', chunks=(2, 3), compression='gzip'
        )
        with pytest.raises(AttributeError, match='injected bug'):
            write_full_matrix(dataset, np.zeros((2, 5)), compression_workers=2)


class _FailingDatasetID:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def write_direct_chunk(self, *_args, **_kwargs):
        raise OSError('injected h5py failure')


class _DatasetWithFailingDirectWrite:
    def __init__(self, real):
        self._real = real
        self.id = _FailingDatasetID(real.id)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setitem__(self, key, value):
        self._real[key] = value


def test_direct_chunk_h5py_failure_logs_and_falls_back(tmp_path, caplog) -> None:
    data = np.arange(10, dtype=np.float32).reshape(2, 5)
    with h5py.File(tmp_path / 'h5py-failure.h5', 'w') as h5, caplog.at_level(logging.WARNING):
        dataset = h5.create_dataset(
            'values', shape=data.shape, dtype=data.dtype, chunks=(2, 3), compression='gzip'
        )
        metrics = write_full_matrix(_DatasetWithFailingDirectWrite(dataset), data)
        np.testing.assert_array_equal(dataset[...], data)

    assert metrics['fallback'] is True
    assert 'injected h5py failure' in metrics['fallback_reason']
    assert any('Falling back to ordinary HDF5' in record.message for record in caplog.records)


def test_lzf_writes_through_h5py_without_fallback_warning(tmp_path, caplog) -> None:
    path = tmp_path / 'lzf.h5'
    data = np.arange(10, dtype=np.float32).reshape(2, 5)

    with h5py.File(path, 'w') as h5, caplog.at_level(logging.WARNING):
        dataset = h5.create_dataset(
            'values',
            shape=data.shape,
            dtype=data.dtype,
            chunks=(2, 3),
            compression='lzf',
            shuffle=True,
        )
        metrics = write_full_matrix(dataset, data)
        np.testing.assert_array_equal(dataset[...], data)

    # lzf is a supported option, not a direct-writer failure: no fallback, no warning.
    assert metrics['direct_chunk'] is False
    assert metrics['fallback'] is False
    assert metrics['fallback_reason'] is None
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_full_matrix_coercion_preserves_staged_array() -> None:
    matrix = np.empty((2, 3), dtype=np.float32, order='C')
    assert h5_storage._coerce_full_matrix(matrix, dtype=np.float32) is matrix


def test_write_rows_in_column_stripes_matches_dense_write(tmp_path) -> None:
    """Stripe writer should match assigning the full matrix."""
    path = tmp_path / 'stripe.h5'
    n_files, n_elements = 3, 17
    full = np.arange(n_files * n_elements, dtype=np.float64).reshape(n_files, n_elements)
    rows = [full[i].copy() for i in range(n_files)]

    with h5py.File(path, 'w') as h5:
        dset = h5.create_dataset('m', shape=(n_files, n_elements), dtype='f8', chunks=(3, 5))
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
