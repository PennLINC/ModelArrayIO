"""Unit tests for TileDB storage helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tiledb

from modelarrayio.storage import tiledb_storage


def assert_domain_dtypes(array: tiledb.Array, expected: list[str]) -> None:
    for idx, dtype in enumerate(expected):
        assert array.schema.domain.dim(idx).dtype == np.dtype(dtype)


def test_build_filter_list_variants() -> None:
    no_filters = tiledb_storage._build_filter_list(None, None, shuffle=False)
    assert isinstance(no_filters, tiledb.FilterList)
    assert len(no_filters) == 0

    # ``none`` is a true no-filter fast path even when the CLI's default
    # shuffle flag remains enabled.
    assert len(tiledb_storage._build_filter_list('none', None, shuffle=True)) == 0

    zstd = tiledb_storage._build_filter_list('zstd', 9, shuffle=True)
    assert len(zstd) >= 2

    fallback = tiledb_storage._build_filter_list('not-a-codec', 'bad', shuffle=False)
    assert len(fallback) == 0


@pytest.mark.parametrize(
    ('compression', 'level', 'expected'),
    [
        ('zstd', None, ('zstd', 5)),
        ('gzip', None, ('gzip', 4)),
        ('zstd', 'bad', ('zstd', 5)),
        (None, 9, ('none', None)),
        ('unknown', 9, ('none', None)),
    ],
)
def test_resolve_compression_settings_reports_effective_values(
    compression, level, expected
) -> None:
    assert tiledb_storage._resolve_compression_settings(compression, level) == expected


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
        assert_domain_dtypes(array, ['int32', 'int32'])
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
    metrics = tiledb_storage.write_rows_in_column_stripes(uri, rows)

    with tiledb.open(uri, 'r') as array:
        np.testing.assert_array_equal(array[:]['values'], np.vstack(rows))
    assert len(tiledb.array_fragments(uri)) == 1
    assert metrics['dense_submissions'] == 1
    assert metrics['fragment_count'] == 1
    json.dumps(metrics)


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
        assert_domain_dtypes(array, ['int32'])
        np.testing.assert_array_equal(array[:]['values'], np.array(['P1', 'P2'], dtype=object))

    # Some TileDB builds do not implicitly create missing parent directories.
    (base / 'scalars').mkdir(parents=True, exist_ok=True)
    tiledb.group_create(str(base / 'scalars' / 'FA'))
    tiledb_storage.write_column_names(str(base), 'FA', ['sub-1', 'sub-2'])
    column_uri = base / 'scalars' / 'FA' / 'column_names'
    with tiledb.open(str(column_uri), 'r') as array:
        assert_domain_dtypes(array, ['int32'])
        np.testing.assert_array_equal(
            array[:]['values'], np.array(['sub-1', 'sub-2'], dtype=object)
        )
    with tiledb.Group(str(base / 'scalars' / 'FA'), 'r') as group:
        assert json.loads(group.meta['column_names']) == ['sub-1', 'sub-2']


def test_create_scalar_matrix_array_writes_values_and_metadata(tmp_path: Path) -> None:
    base = tmp_path / 'store.tdb'
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metrics = {}
    uri = tiledb_storage.create_scalar_matrix_array(
        str(base),
        'scalars/MD/values',
        values,
        ['first', 'second'],
        storage_dtype='float32',
        compression='zstd',
        compression_level=3,
        metrics=metrics,
    )
    with tiledb.open(uri, 'r') as array:
        assert_domain_dtypes(array, ['int32', 'int32'])
        np.testing.assert_array_equal(array[:]['values'], values)
        assert json.loads(array.meta['column_names']) == ['first', 'second']
    assert len(tiledb.array_fragments(uri)) == 1
    assert metrics['dense_submissions'] == 1
    assert metrics['fragment_count'] == 1
    assert metrics['matrix_shape'] == [2, 2]
    json.dumps(metrics)


def test_create_scalar_matrix_array_reports_default_compression_level(tmp_path: Path) -> None:
    metrics = {}
    tiledb_storage.create_scalar_matrix_array(
        str(tmp_path / 'store.tdb'),
        'scalars/MD/values',
        np.ones((2, 2), dtype=np.float32),
        ['first', 'second'],
        compression='zstd',
        compression_level=None,
        metrics=metrics,
    )

    assert metrics['compression'] == 'zstd'
    assert metrics['compression_level'] == 5


def test_create_scalar_matrix_array_reports_effective_shuffle_for_no_compression(
    tmp_path: Path,
) -> None:
    metrics = {}
    uri = tiledb_storage.create_scalar_matrix_array(
        str(tmp_path / 'store.tdb'),
        'scalars/MD/values',
        np.ones((2, 2), dtype=np.float32),
        ['first', 'second'],
        compression='none',
        compression_level=None,
        shuffle=True,
        metrics=metrics,
    )

    with tiledb.open(uri, 'r') as array:
        assert len(array.schema.attr('values').filters) == 0
    assert metrics['compression'] == 'none'
    assert metrics['shuffle'] is False


def test_create_scalar_matrix_array_skips_output_stats_without_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    def _unexpected_stats(*_args, **_kwargs):
        raise AssertionError('output statistics should only be collected for telemetry')

    monkeypatch.setattr(tiledb_storage, '_local_output_stats', _unexpected_stats)
    monkeypatch.setattr(tiledb_storage.tiledb, 'array_fragments', _unexpected_stats)

    uri = tiledb_storage.create_scalar_matrix_array(
        str(tmp_path / 'store.tdb'),
        'scalars/MD/values',
        np.ones((2, 2), dtype=np.float32),
        ['first', 'second'],
    )
    with tiledb.open(uri, 'r') as array:
        np.testing.assert_array_equal(array[:]['values'], np.ones((2, 2), dtype=np.float32))


def test_write_full_matrix_rejects_non_matrix_schema(tmp_path: Path) -> None:
    uri = str(tmp_path / 'one-dimensional.tdb')
    dimension = tiledb.Dim(name='items', domain=(0, 2), tile=3, dtype=np.int32)
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(dimension),
        attrs=[tiledb.Attr(name='values', dtype=np.float32)],
        sparse=False,
    )
    tiledb.Array.create(uri, schema)

    with pytest.raises(ValueError, match='must have two dimensions'):
        tiledb_storage.write_full_matrix(uri, np.ones((1, 3), dtype=np.float32))


def test_write_full_matrix_rejects_unexpected_attributes(tmp_path: Path) -> None:
    uri = str(tmp_path / 'wrong-attributes.tdb')
    dimensions = [
        tiledb.Dim(name='subjects', domain=(0, 1), tile=2, dtype=np.int32),
        tiledb.Dim(name='items', domain=(0, 2), tile=3, dtype=np.int32),
    ]
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dimensions),
        attrs=[tiledb.Attr(name='other', dtype=np.float32)],
        sparse=False,
    )
    tiledb.Array.create(uri, schema)

    with pytest.raises(ValueError, match="attribute named 'values'"):
        tiledb_storage.write_full_matrix(uri, np.ones((2, 3), dtype=np.float32))


def test_full_matrix_coercion_preserves_staged_array() -> None:
    matrix = np.empty((2, 3), dtype=np.float32, order='C')
    assert tiledb_storage._coerce_full_matrix(matrix, dtype=np.float32) is matrix
