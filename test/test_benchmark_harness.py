"""Regression tests for benchmark result gating and artifact isolation."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import harness_common
import output_compare
import run_benchmarks as harness


def test_r_gate_and_missing_output_make_case_fail() -> None:
    assert harness._case_failure_reasons(
        returncode=0,
        outputs_present=True,
        r_gate='failed',
        meets_acceptance_target=True,
    ) == ['r_gate']
    assert harness._case_failure_reasons(
        returncode=0,
        outputs_present=False,
        r_gate='failed_missing_output',
        meets_acceptance_target=None,
    ) == ['missing_output', 'r_gate']
    assert harness._case_status([], 'not_run') == 'unverified'
    assert harness._case_status([], 'passed') == 'passed'
    assert harness._case_status(['r_gate'], 'failed') == 'failed'


def test_missing_r_gate_executable_is_recorded_as_failure(tmp_path: Path) -> None:
    output = tmp_path / 'output.h5'
    output.touch()

    status, _seconds = harness._run_r_gate(
        'definitely-missing-r-command {output}',
        [output],
        'hdf5',
        ['FA'],
        tmp_path / 'r-gate.log',
    )

    assert status == 'failed'
    assert 'Could not execute R gate command' in (tmp_path / 'r-gate.log').read_text()


def test_shard_artifacts_are_isolated(tmp_path: Path) -> None:
    first = harness._run_artifact_paths(tmp_path, 'node-01')
    second = harness._run_artifact_paths(tmp_path, 'node-02')

    assert first[0] == tmp_path / 'shard-node-01'
    assert first[1] == tmp_path / 'results.node-01.csv'
    assert first[2] == tmp_path / 'run_meta.node-01.json'
    assert set(first).isdisjoint(second)


@pytest.mark.parametrize('value', ['../escape', '/absolute', 'contains space', '.hidden'])
def test_shard_identifier_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError, match='result-shard'):
        harness._validated_shard(value)


@pytest.mark.parametrize('value', ['../escape', '/absolute', 'contains space', '.hidden'])
def test_run_identifier_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError, match='run-id'):
        harness._validated_run_id(value)


def test_completed_case_keys_support_resume_and_legacy_rows(tmp_path: Path) -> None:
    results = tmp_path / 'results.csv'
    with results.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_key', 'case_id', 'status'])
        writer.writeheader()
        writer.writerow(
            {
                'case_key': 'stable-case',
                'case_id': 'stable-case-retry-01',
                'status': 'passed',
            }
        )
        writer.writerow({'case_key': '', 'case_id': 'legacy-case', 'status': 'failed'})

    assert harness._completed_case_keys(results) == {'stable-case', 'legacy-case'}
    assert harness._recorded_failure_count(results) == 1


def test_available_cpu_count_prefers_process_allocation(monkeypatch) -> None:
    monkeypatch.setattr(harness.os, 'process_cpu_count', lambda: 3, raising=False)
    monkeypatch.setattr(harness.os, 'cpu_count', lambda: 64)

    assert harness._available_cpu_count() == 3


def test_full_tier_covers_complete_reference_grid(monkeypatch) -> None:
    monkeypatch.setattr(harness, '_available_cpu_count', lambda: 14)
    args = SimpleNamespace(
        tier='full',
        workers=None,
        dtypes=None,
        target_chunk_mb=None,
        backends=['hdf5', 'tiledb'],
        compression_workers=None,
    )

    cases = harness._case_grid(args)

    assert len(cases) == 440
    assert sum(case['backend'] == 'hdf5' for case in cases) == 280
    assert sum(case['backend'] == 'tiledb' for case in cases) == 160
    assert {case['workers'] for case in cases} == {1, 2, 4, 8, 14}
    assert {case['dtype'] for case in cases} == {'float32', 'float64'}
    assert {case['target_chunk_mb'] for case in cases} == {8.0, 32.0, 64.0, 128.0}
    assert {
        (case['compression'], case['compression_level'])
        for case in cases
        if case['backend'] == 'hdf5'
    } == {('gzip', 1), ('gzip', 4), ('none', 0)}
    assert {
        (case['compression'], case['compression_level'])
        for case in cases
        if case['backend'] == 'tiledb'
    } == {('zstd', 1), ('zstd', 3), ('zstd', 5), ('none', 0)}


def test_resolved_executable_keeps_path_lookup_and_absolutizes_paths(tmp_path: Path) -> None:
    executable = tmp_path / 'bin' / 'modelarrayio'
    executable.parent.mkdir()
    executable.touch()

    assert harness._resolved_executable('modelarrayio') == 'modelarrayio'
    assert harness._resolved_executable(str(executable)) == str(executable.resolve())


def test_long_cohort_scalar_names_and_modality_are_recorded() -> None:
    rows = [
        {'scalar_name': 'FA', 'source_file': 'first.nii.gz'},
        {'scalar_name': 'FA', 'source_file': 'second.nii.gz'},
        {'scalar_name': 'MD', 'source_file': 'third.nii.gz'},
    ]

    assert harness_common.scalar_names(rows, []) == ['FA', 'MD']
    assert harness_common.source_modality(harness_common.scalar_source_paths(rows, [])) == 'nifti'
    assert harness._resolve_split_files(None, []) is False
    assert harness._resolve_split_files(None, ['FA', 'MD']) is True
    assert harness._resolve_split_files(False, ['FA', 'MD']) is False


def test_wide_scalar_inference_excludes_demographic_columns() -> None:
    rows = [
        {
            'subject_id': 'sub-01',
            'age': '12',
            'source_mask_file': 'mask.nii.gz',
            'FA': 'fa.nii.gz',
            'MD': 'md.nii.gz',
        }
    ]

    assert harness_common.infer_scalar_columns(rows) == ['FA', 'MD']


def test_scalar_inference_rejects_empty_and_uninferable_cohorts() -> None:
    # Both harnesses share this check, so compare_revisions.py also fails loudly instead of
    # treating a wide cohort with no recognizable scalar columns as long-format.
    with pytest.raises(ValueError, match='no rows'):
        harness_common.infer_scalar_columns([])
    with pytest.raises(ValueError, match='pass --scalar-columns'):
        harness_common.infer_scalar_columns([{'subject_id': 'sub-01', 'age': '12'}])


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('s3://bucket/data.dscalar.nii', 'cifti'),
        ('input.pscalar.nii', 'cifti'),
        ('input.pconn.nii', 'cifti'),
        ('input.mif.gz', 'mif'),
    ],
)
def test_benchmark_modality_supports_cifti_and_mif(path: str, expected: str) -> None:
    assert harness_common.source_modality([path]) == expected


def test_write_csv_uses_lf_and_unions_columns_in_first_seen_order(tmp_path: Path) -> None:
    path = tmp_path / 'nested' / 'rows.csv'
    harness_common.write_csv(path, [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

    assert path.read_bytes() == b'a,b,c\n1,2,\n,3,4\n'


def test_output_path_follows_the_converter_naming() -> None:
    case_dir = Path('case')

    assert harness_common.output_path(case_dir, 'hdf5') == case_dir / 'modelarray.h5'
    assert harness_common.output_path(case_dir, 'tiledb', 'FA') == case_dir / 'FA_modelarray.tdb'


def _write_hdf5_output(path: Path, values: np.ndarray, names: list[str]) -> Path:
    with h5py.File(path, 'w') as handle:
        handle['scalars/FA/values'] = values
        handle['scalars/FA/column_names'] = np.asarray(names, dtype=h5py.string_dtype())
    return path


def test_compare_outputs_reports_value_nan_and_source_differences(tmp_path: Path) -> None:
    values = np.asarray([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    names = ['sub-01', 'sub-02']
    reference = _write_hdf5_output(tmp_path / 'reference.h5', values, names)

    same = output_compare.compare_outputs(
        reference, _write_hdf5_output(tmp_path / 'same.h5', values.copy(), names), 'FA', 2
    )
    assert same == {
        'shape_equal': True,
        'dtype_equal': True,
        'sources_equal': True,
        'nan_equal': True,
        'values_equal': True,
        'max_abs_difference': 0.0,
    }

    changed = values.copy()
    changed[1, 2] = 6.5
    changed[0, 1] = 2.0
    different = output_compare.compare_outputs(
        reference, _write_hdf5_output(tmp_path / 'diff.h5', changed, names[::-1]), 'FA', 2
    )
    assert different == {
        'shape_equal': True,
        'dtype_equal': True,
        'sources_equal': False,
        'nan_equal': False,
        'values_equal': False,
        'max_abs_difference': pytest.approx(0.5),
    }
