"""Unit tests for shared CLI helper utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from modelarrayio.cli import utils as cli_utils


def test_prepare_output_directory_warns_for_existing_path(tmp_path: Path, caplog) -> None:
    output_dir = tmp_path / 'results'
    output_dir.mkdir()
    logger = logging.getLogger('test_cli_utils')

    with caplog.at_level(logging.WARNING):
        result = cli_utils.prepare_output_directory(output_dir, logger)

    assert result == output_dir
    assert output_dir.exists()
    assert any('Output directory exists' in record.message for record in caplog.records)


def test_prefixed_output_path_sanitizes_prefix(tmp_path: Path) -> None:
    output_path = tmp_path / 'stats.h5'
    prefixed = cli_utils.prefixed_output_path(output_path, 'p.value/result')
    assert prefixed.name == 'p.value_result_stats.h5'


def test_write_table_dataset_writes_transposed_data_and_attrs(tmp_path: Path) -> None:
    table = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    h5_path = tmp_path / 'table.h5'

    with h5py.File(h5_path, 'w') as h5_file:
        dataset = cli_utils.write_table_dataset(
            h5_file,
            'table',
            table,
            extra_attrs={'labels': ['left', 'right']},
        )
        np.testing.assert_array_equal(dataset[...], np.array([[1, 2], [3, 4]]))
        assert [str(value) for value in dataset.attrs['column_names']] == ['a', 'b']
        assert [str(value) for value in dataset.attrs['labels']] == ['left', 'right']


def test_write_hdf5_scalar_matrices_skips_empty_rows_and_writes_values(tmp_path: Path) -> None:
    h5_path = tmp_path / 'scalars.h5'
    with h5py.File(h5_path, 'w') as h5_file:
        cli_utils.write_hdf5_scalar_matrices(
            h5_file,
            scalars={
                'FA': [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)],
                'MD': [],
            },
            sources_by_scalar={'FA': ['sub-1', 'sub-2'], 'MD': []},
            storage_dtype='float32',
            compression='gzip',
            compression_level=1,
            shuffle=True,
            chunk_voxels=0,
            target_chunk_mb=0.5,
        )

    with h5py.File(h5_path, 'r') as h5_file:
        assert 'scalars/FA/values' in h5_file
        assert 'scalars/MD/values' not in h5_file
        values = h5_file['scalars/FA/values'][...]
        np.testing.assert_array_equal(values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_write_tiledb_scalar_matrices_calls_column_name_writer(monkeypatch, tmp_path: Path) -> None:
    called = {'create': [], 'columns': [], 'write': []}

    def _fake_create(*args, **kwargs):
        called['create'].append((args, kwargs))

    def _fake_write_columns(*args, **kwargs):
        called['columns'].append((args, kwargs))

    def _fake_write_rows(*args, **kwargs):
        called['write'].append((args, kwargs))

    monkeypatch.setattr(cli_utils.tiledb_storage, 'create_empty_scalar_matrix_array', _fake_create)
    monkeypatch.setattr(cli_utils.tiledb_storage, 'write_column_names', _fake_write_columns)
    monkeypatch.setattr(cli_utils.tiledb_storage, 'write_rows_in_column_stripes', _fake_write_rows)

    cli_utils.write_tiledb_scalar_matrices(
        tmp_path / 'out.tdb',
        scalars={'FA': [np.array([1.0, 2.0], dtype=np.float32)], 'MD': []},
        sources_by_scalar={'FA': ['sub-1.nii.gz'], 'MD': []},
        storage_dtype='float32',
        compression='zstd',
        compression_level=3,
        shuffle=False,
        chunk_voxels=16,
        target_chunk_mb=1.0,
        write_column_name_arrays=True,
    )

    assert len(called['create']) == 1
    assert len(called['columns']) == 1
    assert len(called['write']) == 1


def test_write_hdf5_and_tiledb_parcel_arrays(monkeypatch, tmp_path: Path) -> None:
    parcel_arrays = {'parcel_id': np.array(['A', 'B'])}
    h5_path = tmp_path / 'parcels.h5'
    with h5py.File(h5_path, 'w') as h5_file:
        cli_utils.write_hdf5_parcel_arrays(h5_file, parcel_arrays)

    with h5py.File(h5_path, 'r') as h5_file:
        np.testing.assert_array_equal(h5_file['parcels/parcel_id'][...].astype(str), np.array(['A', 'B']))

    calls = []

    def _fake_write_parcel_names(base_uri, array_path, names):
        calls.append((base_uri, array_path, names))

    monkeypatch.setattr(cli_utils.tiledb_storage, 'write_parcel_names', _fake_write_parcel_names)
    cli_utils.write_tiledb_parcel_arrays(tmp_path / 'out.tdb', {'parcel_id': np.array(['1', '2'])})
    assert calls == [(str(tmp_path / 'out.tdb'), 'parcels/parcel_id', ['1', '2'])]


def test_decode_names_handles_scalar_and_bytes() -> None:
    decoded = cli_utils._decode_names(np.array([b'a\x00', b' b ', b'']))
    assert decoded == ['a', 'b']
    assert cli_utils._decode_names(' value ') == ['value']


def test_read_result_names_prefers_attrs_then_fallback_paths(tmp_path: Path, caplog) -> None:
    h5_path = tmp_path / 'results.h5'
    logger = logging.getLogger('test_cli_utils.read_result_names')

    with h5py.File(h5_path, 'w') as h5_file:
        group = h5_file.require_group('results/lm')
        matrix = group.create_dataset(
            'results_matrix',
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

        matrix.attrs['colnames'] = np.array([b'first', b'second'])
        assert cli_utils.read_result_names(h5_file, 'lm', matrix, logger=logger) == ['first', 'second']

        del matrix.attrs['colnames']
        group.create_dataset(
            'column_names',
            data=np.array(['alpha', 'beta'], dtype=h5py.string_dtype('utf-8')),
        )
        assert cli_utils.read_result_names(h5_file, 'lm', matrix, logger=logger) == ['alpha', 'beta']

    with h5py.File(h5_path, 'a') as h5_file:
        group = h5_file.require_group('results/lm_nested')
        matrix = group.create_dataset(
            'matrix',
            data=np.array([[1.0, 2.0]], dtype=np.float32),
        )
        group.require_group('results_matrix').create_dataset(
            'column_names',
            data=np.array(['gamma'], dtype=h5py.string_dtype('utf-8')),
        )
        assert cli_utils.read_result_names(h5_file, 'lm_nested', matrix, logger=logger) == ['gamma']

        del h5_file['results/lm_nested/results_matrix/column_names']
        with caplog.at_level(logging.WARNING):
            fallback = cli_utils.read_result_names(h5_file, 'lm_nested', matrix, logger=logger)
        assert fallback == ['component001']
        assert any('Unable to read column names' in record.message for record in caplog.records)
