"""Tests for shared argparse helpers and CLI argument parsers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from modelarrayio.cli import parser_utils
from modelarrayio.cli.export_results import _parse_export_results
from modelarrayio.cli.to_modelarray import _parse_to_modelarray


def test_add_log_level_arg_registers_choices():
    parser = argparse.ArgumentParser()
    parser_utils.add_log_level_arg(parser)
    args = parser.parse_args(['--log-level', 'DEBUG'])
    assert args.log_level == 'DEBUG'
    with pytest.raises(SystemExit):
        parser.parse_args(['--log-level', 'INVALID'])


def test_parse_to_modelarray_minimal_defaults(tmp_path):
    cohort = tmp_path / 'cohort.csv'
    cohort.write_text('scalar_name,source_file\nx,file.nii.gz\n')
    parser = _parse_to_modelarray()
    args = parser.parse_args(['--cohort-file', str(cohort)])
    assert args.cohort_file == cohort.resolve()
    assert args.output == Path('modelarray.h5')
    assert args.backend == 'hdf5'
    assert args.storage_dtype == 'float32'
    assert args.compression == 'gzip'
    assert args.compression_level == 4
    assert args.shuffle is True
    assert args.chunk_voxels == 0
    assert args.target_chunk_mb == 2.0
    assert args.workers == 1
    assert args.s3_workers == 1
    assert args.scalar_columns is None
    assert args.group_mask_file is None
    assert args.index_file is None
    assert args.directions_file is None
    assert args.log_level == 'INFO'


def test_parse_to_modelarray_cohort_file_underscore_alias(tmp_path):
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    parser = _parse_to_modelarray()
    args = parser.parse_args(['--cohort_file', str(cohort)])
    assert args.cohort_file == cohort.resolve()


def test_parse_to_modelarray_optional_paths_and_flags(tmp_path):
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    mask = tmp_path / 'mask.nii.gz'
    mask.touch()
    idx = tmp_path / 'index.nii.gz'
    idx.touch()
    dirs = tmp_path / 'dirs.nii.gz'
    dirs.touch()
    parser = _parse_to_modelarray()
    args = parser.parse_args(
        [
            '--cohort-file',
            str(cohort),
            '--output',
            str(tmp_path / 'out.h5'),
            '--backend',
            'tiledb',
            '--dtype',
            'float64',
            '--compression',
            'zstd',
            '--compression-level',
            '7',
            '--no-shuffle',
            '--chunk-voxels',
            '64',
            '--workers',
            '4',
            '--s3-workers',
            '2',
            '--scalar-columns',
            'col_a',
            'col_b',
            '--mask',
            str(mask),
            '--index-file',
            str(idx),
            '--directions-file',
            str(dirs),
            '--log-level',
            'WARNING',
        ]
    )
    assert args.output == tmp_path / 'out.h5'
    assert args.backend == 'tiledb'
    assert args.storage_dtype == 'float64'
    assert args.compression == 'zstd'
    assert args.compression_level == 7
    assert args.shuffle is False
    assert args.chunk_voxels == 64
    assert args.workers == 4
    assert args.s3_workers == 2
    assert args.scalar_columns == ['col_a', 'col_b']
    assert args.group_mask_file == mask.resolve()
    assert args.index_file == idx.resolve()
    assert args.directions_file == dirs.resolve()
    assert args.log_level == 'WARNING'


def test_parse_to_modelarray_target_chunk_mb_branch(tmp_path):
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    parser = _parse_to_modelarray()
    args = parser.parse_args(['--cohort-file', str(cohort), '--target-chunk-mb', '8.5'])
    assert args.target_chunk_mb == 8.5
    assert args.chunk_voxels == 0


def test_parse_to_modelarray_requires_cohort_file(tmp_path):
    parser = _parse_to_modelarray()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parse_to_modelarray_rejects_missing_cohort_path(tmp_path):
    parser = _parse_to_modelarray()
    missing = tmp_path / 'nope.csv'
    with pytest.raises(SystemExit):
        parser.parse_args(['--cohort-file', str(missing)])


def test_parse_to_modelarray_rejects_invalid_backend(tmp_path):
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    parser = _parse_to_modelarray()
    with pytest.raises(SystemExit):
        parser.parse_args(['--cohort-file', str(cohort), '--backend', 'sqlite'])


def test_parse_export_results_minimal_defaults(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    out = tmp_path / 'exports'
    parser = _parse_export_results()
    args = parser.parse_args(
        [
            '--analysis-name',
            'lm',
            '--input-hdf5',
            str(h5),
            '--output-dir',
            str(out),
        ]
    )
    assert args.analysis_name == 'lm'
    assert args.in_file == h5.resolve()
    assert args.output_dir == str(out)
    assert args.group_mask_file is None
    assert args.compress is True
    assert args.index_file is None
    assert args.directions_file is None
    assert args.cohort_file is None
    assert args.example_file is None
    assert args.log_level == 'INFO'


def test_parse_export_results_input_hdf5_underscore_alias(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    parser = _parse_export_results()
    args = parser.parse_args(
        [
            '--analysis_name',
            'a',
            '--input_hdf5',
            str(h5),
            '--output_dir',
            str(tmp_path / 'o'),
        ]
    )
    assert args.in_file == h5.resolve()


def test_parse_export_results_nifti_options(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    mask = tmp_path / 'mask.nii.gz'
    mask.touch()
    parser = _parse_export_results()
    args = parser.parse_args(
        [
            '--analysis-name',
            'x',
            '--input-hdf5',
            str(h5),
            '--output-dir',
            str(tmp_path / 'out'),
            '--mask',
            str(mask),
            '--no-compress',
        ]
    )
    assert args.group_mask_file == mask.resolve()
    assert args.compress is False


def test_parse_export_results_mif_files(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    idx = tmp_path / 'index.nii.gz'
    idx.touch()
    dirs = tmp_path / 'dirs.nii.gz'
    dirs.touch()
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    parser = _parse_export_results()
    args = parser.parse_args(
        [
            '--analysis-name',
            'm',
            '--input-hdf5',
            str(h5),
            '--output-dir',
            str(tmp_path / 'out'),
            '--index-file',
            str(idx),
            '--directions-file',
            str(dirs),
            '--cohort-file',
            str(cohort),
        ]
    )
    assert args.index_file == idx.resolve()
    assert args.directions_file == dirs.resolve()
    assert args.cohort_file == cohort.resolve()


def test_parse_export_results_example_file_instead_of_cohort(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    example = tmp_path / 'example.dscalar.nii'
    example.touch()
    parser = _parse_export_results()
    args = parser.parse_args(
        [
            '--analysis-name',
            'c',
            '--input-hdf5',
            str(h5),
            '--output-dir',
            str(tmp_path / 'out'),
            '--example-file',
            str(example),
        ]
    )
    assert args.example_file == example.resolve()
    assert args.cohort_file is None


def test_parse_export_results_cohort_and_example_mutually_exclusive(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    cohort = tmp_path / 'cohort.csv'
    cohort.touch()
    example = tmp_path / 'ex.nii'
    example.touch()
    parser = _parse_export_results()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--analysis-name',
                'x',
                '--input-hdf5',
                str(h5),
                '--output-dir',
                str(tmp_path / 'out'),
                '--cohort-file',
                str(cohort),
                '--example-file',
                str(example),
            ]
        )


def test_parse_export_results_requires_analysis_input_output(tmp_path):
    h5 = tmp_path / 'results.h5'
    h5.touch()
    parser = _parse_export_results()
    with pytest.raises(SystemExit):
        parser.parse_args(['--input-hdf5', str(h5), '--output-dir', str(tmp_path / 'o')])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--analysis-name',
                'a',
                '--output-dir',
                str(tmp_path / 'o'),
            ]
        )


def test_parse_export_results_rejects_missing_hdf5(tmp_path):
    parser = _parse_export_results()
    missing = tmp_path / 'gone.h5'
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--analysis-name',
                'a',
                '--input-hdf5',
                str(missing),
                '--output-dir',
                str(tmp_path / 'out'),
            ]
        )
