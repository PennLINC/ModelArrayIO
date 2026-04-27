"""Unit tests for TileDB storage helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tiledb

from modelarrayio.storage import tiledb_storage


def test_build_filter_list_variants() -> None:
    no_filters = tiledb_storage._build_filter_list(None, None, shuffle=False)
    assert isinstance(no_filters, tiledb.FilterList)
    assert len(no_filters) == 0

    zstd = tiledb_storage._build_filter_list('zstd', 9, shuffle=True)
    assert len(zstd) >= 2

    fallback = tiledb_storage._build_filter_list('not-a-codec', 'bad', shuffle=False)
    assert len(fallback) == 0


def test_create_empty_scalar_matrix_array_writes_metadata_and_overwrites(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    uri = tiledb_storage.create_empty_scalar_matrix_array(
        str(base),
        'scalars/FA/values',
        n_files=2,
        n_elements=3,
        storage_dtype='float32',
        compression='gzip',
        compression_level=1,
        shuffle=True,
        tile_voxels=2,
        target_tile_mb=0.5,
        sources_list=['s1', 's2'],
    )
    assert tiledb.object_type(uri) == 'array'
    with tiledb.open(uri, 'r') as array:
        assert json.loads(array.meta['column_names']) == ['s1', 's2']

    uri_again = tiledb_storage.create_empty_scalar_matrix_array(
        str(base),
        'scalars/FA/values',
        n_files=2,
        n_elements=3,
    )
    assert uri_again == uri
    assert tiledb.object_type(uri_again) == 'array'


def test_write_rows_in_column_stripes_round_trip(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    uri = tiledb_storage.create_empty_scalar_matrix_array(
        str(base),
        'scalars/FA/values',
        n_files=3,
        n_elements=5,
        storage_dtype='float32',
        tile_voxels=2,
    )

    rows = [
        np.array([1, 2, 3, 4, 5], dtype=np.float32),
        np.array([6, 7, 8, 9, 10], dtype=np.float32),
        np.array([11, 12, 13, 14, 15], dtype=np.float32),
    ]
    tiledb_storage.write_rows_in_column_stripes(uri, rows)

    with tiledb.open(uri, 'r') as array:
        np.testing.assert_array_equal(array[:]['values'], np.vstack(rows))


def test_write_rows_in_column_stripes_rejects_wrong_row_count(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    uri = tiledb_storage.create_empty_scalar_matrix_array(
        str(base),
        'scalars/FA/values',
        n_files=2,
        n_elements=3,
    )
    with pytest.raises(ValueError, match='rows length does not match'):
        tiledb_storage.write_rows_in_column_stripes(uri, [np.array([1, 2, 3], dtype=np.float32)])


def test_write_parcel_names_and_column_names(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    with pytest.raises(ValueError, match='must not be empty'):
        tiledb_storage.write_parcel_names(str(base), 'parcels/parcel_id', [])

    tiledb_storage.write_parcel_names(str(base), 'parcels/parcel_id', ['P1', 'P2'])
    parcel_uri = base / 'parcels' / 'parcel_id'
    assert tiledb.object_type(str(parcel_uri)) == 'array'
    with tiledb.open(str(parcel_uri), 'r') as array:
        np.testing.assert_array_equal(array[:]['values'], np.array(['P1', 'P2'], dtype=object))

    tiledb.group_create(str(base / 'scalars' / 'FA'))
    tiledb_storage.write_column_names(str(base), 'FA', ['sub-1', 'sub-2'])
    column_uri = base / 'scalars' / 'FA' / 'column_names'
    with tiledb.open(str(column_uri), 'r') as array:
        np.testing.assert_array_equal(
            array[:]['values'], np.array(['sub-1', 'sub-2'], dtype=object)
        )
    with tiledb.Group(str(base / 'scalars' / 'FA'), 'r') as group:
        assert json.loads(group.meta['column_names']) == ['sub-1', 'sub-2']


def test_create_scalar_matrix_array_writes_values_and_metadata(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    uri = tiledb_storage.create_scalar_matrix_array(
        str(base),
        'scalars/MD/values',
        values,
        ['first', 'second'],
        storage_dtype='float32',
        compression='zstd',
        compression_level=3,
    )
    with tiledb.open(uri, 'r') as array:
        np.testing.assert_array_equal(array[:]['values'], values)
        assert json.loads(array.meta['column_names']) == ['first', 'second']
