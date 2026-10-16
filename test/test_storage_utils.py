"""Unit tests for shared storage helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelarrayio.storage import utils as storage_utils


def test_resolve_dtype_rejects_unknown_values() -> None:
    assert storage_utils.resolve_dtype('float32') == np.float32
    assert storage_utils.resolve_dtype(np.float64) == np.float64
    with pytest.raises(ValueError, match='Unsupported storage dtype'):
        storage_utils.resolve_dtype('int16')


def test_compute_full_subject_chunk_shape_auto() -> None:
    chunk = storage_utils.compute_full_subject_chunk_shape(
        n_files=4,
        n_elements=128,
        item_chunk=0,
        target_chunk_mb=1.0,
        storage_np_dtype='float32',
    )
    assert chunk[0] == 4
    assert 1 <= chunk[1] <= 128


@pytest.mark.parametrize(
    ('item_chunk', 'target_chunk_mb'),
    [(-1, 1.0), (1.5, 1.0), (True, 1.0), (0, 0.0), (0, -1.0), (0, np.nan)],
)
def test_compute_full_subject_chunk_shape_rejects_invalid_layout(
    item_chunk, target_chunk_mb
) -> None:
    with pytest.raises(ValueError, match=r'item chunk|target chunk size'):
        storage_utils.compute_full_subject_chunk_shape(
            n_files=4,
            n_elements=128,
            item_chunk=item_chunk,
            target_chunk_mb=target_chunk_mb,
            storage_np_dtype='float32',
        )


def test_normalize_column_names_accepts_series_and_sequences() -> None:
    assert storage_utils.normalize_column_names(['a', 2]) == ['a', '2']
    assert storage_utils.normalize_column_names(pd.Series(['x', 'y'])).__class__ is list


@pytest.mark.parametrize('name', ['', '   ', '.', '..', 'a/b', r'a\b', 'a\x00b'])
def test_validate_scalar_name_rejects_unsafe_path_segments(name: str) -> None:
    with pytest.raises(ValueError, match=r'Scalar names|Unsafe scalar name'):
        storage_utils.validate_scalar_name(name)


def test_validate_scalar_name_preserves_valid_identifier() -> None:
    assert storage_utils.validate_scalar_name('mean FA') == 'mean FA'


@pytest.mark.parametrize(
    ('backend', 'compression', 'level', 'expected'),
    [
        ('hdf5', None, None, ('gzip', 4)),
        ('tiledb', None, None, ('zstd', 5)),
        ('hdf5', 'lzf', 9, ('lzf', None)),
        ('tiledb', 'none', 9, ('none', None)),
        ('tiledb', 'gzip', 2, ('gzip', 2)),
    ],
)
def test_resolve_backend_compression_defaults_and_levels(
    backend, compression, level, expected
) -> None:
    assert storage_utils.resolve_backend_compression(backend, compression, level) == expected


@pytest.mark.parametrize(
    ('backend', 'compression', 'level'),
    [
        ('hdf5', 'zstd', None),
        ('tiledb', 'lzf', None),
        ('hdf5', 'gzip', 10),
        ('hdf5', 'gzip', 2.0),
    ],
)
def test_resolve_backend_compression_rejects_invalid_combinations(
    backend, compression, level
) -> None:
    with pytest.raises(ValueError, match=r'not supported|between 0 and 9|must be an integer'):
        storage_utils.resolve_backend_compression(backend, compression, level)
