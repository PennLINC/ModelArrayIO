"""Smoke tests for shared argparse helpers."""

from __future__ import annotations

import argparse

from modelarrayio.cli import parser_utils


def _parser_with_cohort() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    parser_utils.add_output_hdf5_arg(p)
    parser_utils.add_cohort_arg(p)
    parser_utils.add_storage_args(p)
    parser_utils.add_backend_arg(p)
    parser_utils.add_s3_workers_arg(p)
    return p


def test_minimal_invocation_defaults() -> None:
    p = _parser_with_cohort()
    args = p.parse_args(['--cohort-file', 'cohort.csv'])
    assert args.cohort_file == 'cohort.csv'
    assert args.storage_dtype == 'float32'
    assert args.compression == 'gzip'
    assert args.compression_level == 4
    assert args.shuffle is True
    assert args.chunk_voxels == 0
    assert args.backend == 'hdf5'
    assert args.s3_workers == 1
    assert args.log_level == 'INFO'


def test_storage_aliases_and_no_shuffle() -> None:
    p = _parser_with_cohort()
    args = p.parse_args(
        [
            '--cohort-file',
            'c.csv',
            '--dtype',
            'float64',
            '--compression',
            'lzf',
            '--no-shuffle',
            '--chunk-voxels',
            '128',
            '--log-level',
            'WARNING',
        ]
    )
    assert args.storage_dtype == 'float64'
    assert args.compression == 'lzf'
    assert args.shuffle is False
    assert args.chunk_voxels == 128
    assert args.log_level == 'WARNING'


def test_output_hdf5_default_name_override() -> None:
    p = argparse.ArgumentParser()
    parser_utils.add_output_hdf5_arg(p, default_name='custom.h5')
    args = p.parse_args([])
    assert args.output_hdf5 == 'custom.h5'


def test_tiledb_args_group() -> None:
    p = argparse.ArgumentParser()
    parser_utils.add_output_tiledb_arg(p, default_name='arrays.tdb')
    parser_utils.add_tiledb_storage_args(p)
    args = p.parse_args([])
    assert args.output_tiledb == 'arrays.tdb'
    assert args.tdb_compression == 'zstd'
    assert args.tdb_tile_voxels == 0


def test_scalar_columns_optional() -> None:
    p = argparse.ArgumentParser()
    parser_utils.add_cohort_arg(p)
    parser_utils.add_scalar_columns_arg(p)
    args = p.parse_args(['--cohort-file', 'c.csv', '--scalar-columns', 'c1', 'c2'])
    assert args.scalar_columns == ['c1', 'c2']
