"""Tests for the CUBIC validation tools: source identity, gates, reference compare."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tiledb

import compare_to_reference as compare_reference
import comparison_runs
import create_comparison_report as comparison_report
import materialize_cohort as materialize
import source_identity
import summarize_scale
import tiledb_read_gate as read_gate

REPO = Path(__file__).resolve().parents[1]


# --- source identity -------------------------------------------------------


def test_source_identity_is_deterministic_and_pins_content(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    (repo / 'src').mkdir(parents=True)
    (repo / 'src' / 'a.py').write_text('x = 1\n')
    (repo / 'pyproject.toml').write_text('[project]\nname = "x"\n')
    suite = repo / 'benchmarks' / 'to_modelarray'
    suite.mkdir(parents=True)
    (suite / 'run.py').write_text('print()\n')
    # Ignored-style directories inside a suite must not affect the hash even without git.
    (suite / 'results').mkdir()
    (suite / 'results' / 'big.csv').write_text('1,2\n')

    first = source_identity.source_identity(repo)
    assert first['commit'] is None  # no git repository
    assert len(first['source_tree_sha256']) == 64
    assert source_identity.source_identity(repo) == first

    (suite / 'results' / 'big.csv').write_text('changed\n')
    assert (
        source_identity.source_identity(repo)['source_tree_sha256'] == first['source_tree_sha256']
    )

    (repo / 'src' / 'a.py').write_text('x = 2\n')
    assert (
        source_identity.source_identity(repo)['source_tree_sha256'] != first['source_tree_sha256']
    )


def test_source_identity_on_this_repository_reports_commit() -> None:
    identity = source_identity.source_identity(REPO)
    assert identity['commit'] is None or len(identity['commit']) == 40
    assert len(identity['source_tree_sha256']) == 64


# --- helpers to build outputs ----------------------------------------------


def _write_tiledb(root: Path, scalar: str, values: np.ndarray, names: list[str], *, rows=None):
    uri = root / 'scalars' / scalar / 'values'
    uri.parent.mkdir(parents=True, exist_ok=True)
    n_rows = rows if rows is not None else values.shape[0]
    domain = tiledb.Domain(
        tiledb.Dim(name='subject', domain=(0, n_rows - 1), tile=n_rows, dtype=np.int32),
        tiledb.Dim(name='element', domain=(0, values.shape[1] - 1), tile=4, dtype=np.int32),
    )
    schema = tiledb.ArraySchema(
        domain=domain, attrs=[tiledb.Attr(name='values', dtype=np.float32)], sparse=False
    )
    tiledb.Array.create(str(uri), schema)
    with tiledb.open(str(uri), 'w') as array:
        array[0 : values.shape[0], 0 : values.shape[1]] = {'values': values}
        array.meta['column_names'] = json.dumps(names)
    return uri


def _write_hdf5(path: Path, scalar: str, values: np.ndarray, names: list[str]) -> Path:
    with h5py.File(path, 'w') as handle:
        group = handle.create_group(f'scalars/{scalar}')
        group.create_dataset('values', data=values)
        group.create_dataset('column_names', data=np.array(names, dtype='S'))
    return path


# --- read gate ---------------------------------------------------------------


def test_read_gate_passes_and_fails_on_both_backends(tmp_path: Path, capsys) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    names = ['s1', 's2', 's3']
    _write_tiledb(tmp_path / 'ok.tdb', 'x', values, names)
    assert read_gate._gate_tiledb(tmp_path / 'ok.tdb', 8) == 0
    h5 = _write_hdf5(tmp_path / 'ok.h5', 'x', values, names)
    assert read_gate._gate_hdf5(h5, 8) == 0

    _write_tiledb(tmp_path / 'bad.tdb', 'x', values, ['only-one'])
    assert read_gate._gate_tiledb(tmp_path / 'bad.tdb', 8) == 1
    assert 'column names' in capsys.readouterr().err


# --- compare to reference ----------------------------------------------------


def _reference(tmp_path: Path):
    ref = tmp_path / 'ref.tdb'
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    values[1, 3] = np.nan
    names = ['a', 'b', 'c', 'd']
    _write_tiledb(ref, 'area', values, names)
    return ref, values, names


def test_compare_aligns_rows_by_name_and_accepts_subsets(tmp_path: Path) -> None:
    ref, values, _names = _reference(tmp_path)
    # Candidate holds rows d and b, in that order — alignment must use names.
    candidate = _write_hdf5(tmp_path / 'cand.h5', 'area', values[[3, 1]], ['d', 'b'])
    result = compare_reference.compare_scalar(ref, candidate, 'area', block_columns=3)
    assert result['passed']
    assert result['values_equal']
    assert result['nan_equal']
    assert result['rows_compared'] == 2
    assert result['max_abs_difference'] == 0.0
    assert result['reference_complete'] is True


def test_compare_detects_value_change_and_misaligned_names(tmp_path: Path) -> None:
    ref, values, names = _reference(tmp_path)
    changed = values.copy()
    changed[2, 5] += 0.5
    candidate = _write_hdf5(tmp_path / 'changed.h5', 'area', changed, names)
    result = compare_reference.compare_scalar(ref, candidate, 'area')
    assert not result['values_equal']
    assert result['max_abs_difference'] == pytest.approx(0.5)
    assert not result['passed']

    bad_names = _write_hdf5(tmp_path / 'names.h5', 'area', values, ['a', 'b', 'c', 'zzz'])
    result = compare_reference.compare_scalar(ref, bad_names, 'area')
    assert result['sources_aligned'] is False
    assert result['rows_missing_in_reference'] == 1
    assert not result['passed']


def test_compare_flags_partial_reference_and_lists_completeness(tmp_path: Path) -> None:
    ref = tmp_path / 'partial.tdb'
    values = np.arange(16, dtype=np.float32).reshape(2, 8)
    # Domain has 4 rows but only 2 were written (a stalled original run).
    _write_tiledb(ref, 'curv', values, ['a', 'b'], rows=4)
    listing = compare_reference.reference_completeness(ref)
    assert listing == [
        {
            'scalar': 'curv',
            'rows': 4,
            'columns': 8,
            'dtype': 'float32',
            'complete': False,
            'nonempty_domain': listing[0]['nonempty_domain'],
        }
    ]
    candidate = _write_hdf5(tmp_path / 'c.h5', 'curv', values, ['a', 'b'])
    result = compare_reference.compare_scalar(ref, candidate, 'curv')
    assert result['values_equal']
    assert result['reference_complete'] is False
    assert not result['passed']


def test_compare_accepts_tiledb_candidate(tmp_path: Path) -> None:
    ref, values, names = _reference(tmp_path)
    cand = tmp_path / 'cand.tdb'
    _write_tiledb(cand, 'area', values, names)
    result = compare_reference.compare_scalar(ref, cand, 'area', block_columns=5)
    assert result['passed']
    assert result['rows_compared'] == 4


# --- materialize cohort -------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def download_file(self, bucket: str, key: str, target: str) -> None:
        self.calls.append((bucket, key))
        Path(target).write_bytes(b'data:' + key.encode())


def test_materialize_rewrites_s3_cells_and_downloads_once(tmp_path: Path) -> None:
    cohort = tmp_path / 'cohort.csv'
    with cohort.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subject_id', 'source_mask_file', 'alff'])
        writer.writeheader()
        writer.writerow(
            {
                'subject_id': 's1',
                'source_mask_file': 's3://bkt/p/func_mask/s1_func_mask.nii.gz',
                'alff': 's3://bkt/p/alff/s1_alff.nii.gz',
            }
        )
        writer.writerow(
            {
                'subject_id': 's2',
                'source_mask_file': 's3://bkt/p/func_mask/s1_func_mask.nii.gz',
                'alff': 's3://bkt/p/alff/s2_alff.nii.gz',
            }
        )
    client = _FakeClient()
    args = SimpleNamespace(
        cohort=cohort,
        data_dir=tmp_path / 'data',
        output=tmp_path / 'local.csv',
        workers=2,
        extra_key=['s3://bkt/p/func_mask/group.nii.gz'],
    )
    count = materialize.materialize(args, client=client)
    assert count == 4  # 2 alff + 1 shared mask + 1 extra key
    assert sorted(client.calls) == sorted(
        [
            ('bkt', 'p/alff/s1_alff.nii.gz'),
            ('bkt', 'p/alff/s2_alff.nii.gz'),
            ('bkt', 'p/func_mask/s1_func_mask.nii.gz'),
            ('bkt', 'p/func_mask/group.nii.gz'),
        ]
    )
    with (tmp_path / 'local.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['alff'].endswith('data/alff/s1_alff.nii.gz')
    assert Path(rows[0]['alff']).is_absolute()
    assert Path(rows[0]['alff']).exists()
    assert rows[1]['source_mask_file'] == rows[0]['source_mask_file']


def test_compare_reports_missing_reference_scalar(tmp_path: Path) -> None:
    ref, values, names = _reference(tmp_path)
    candidate = _write_hdf5(tmp_path / 'c.h5', 'thickness', values, names)
    result = compare_reference.compare_scalar(ref, candidate, 'thickness')
    assert result['reference_missing'] is True
    assert result['passed'] is False


# --- summarize_scale -----------------------------------------------------------


def test_summarize_scale_collects_every_ladder_step(tmp_path: Path) -> None:
    root = tmp_path / 'results'
    compare = root / 'cubic-03-area-compare'
    compare.mkdir(parents=True)
    cohort = tmp_path / 'area.csv'
    cohort.write_text('subject_id,scalar_name,source_file\ns1,area,/a\ns2,area,/b\n')
    (compare / 'comparison_meta.json').write_text(
        json.dumps({'cohort': str(cohort), 'scalar_names': ['area'], 'workers': 1})
    )
    with (compare / 'comparison_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['implementation', 'backend', 'wall_seconds'])
        writer.writeheader()
        writer.writerows(
            [
                {'implementation': 'baseline', 'backend': 'hdf5', 'wall_seconds': '100'},
                {'implementation': 'optimized', 'backend': 'hdf5', 'wall_seconds': '4'},
            ]
        )
    sweep = root / 'cubic-03-area-sweep'
    sweep.mkdir()
    with (sweep / 'results.csv').open('w', newline='') as stream:
        fields = [
            'scalar_names',
            'input_file_count',
            'backend',
            'workers',
            'wall_seconds',
            'input_files_per_minute',
            'peak_memory_bytes',
            'fragment_count',
            'fast_path_rows',
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                'scalar_names': 'area',
                'input_file_count': '2',
                'backend': 'tiledb',
                'workers': '8',
                'wall_seconds': '3.5',
                'input_files_per_minute': '34.3',
                'peak_memory_bytes': str(2**30),
                'fragment_count': '1',
                'fast_path_rows': '1',
            }
        )
    shard = root / 'cubic-04-shard' / 'curv'
    shard.mkdir(parents=True)
    (shard / 'wall.csv').write_text('wall_seconds,hdf5,60.5\nwall_seconds,tiledb,50.0\n')
    (shard / 'PASSED').write_text('ok')
    prod = root / 'production' / 'checks'
    prod.mkdir(parents=True)
    (prod / 'wall_seconds.csv').write_text('sulc,70.25\n')

    output = tmp_path / 'scale.csv'
    args = ['--results-root', str(root), '--output', str(output)]

    old_argv = sys.argv
    sys.argv = ['summarize_scale.py', *args]
    try:
        assert summarize_scale.main() == 0
    finally:
        sys.argv = old_argv
    with output.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    steps = [row['step'] for row in rows]
    assert steps == [
        '3_full_scalar_original_vs_optimized',
        '3_full_scalar_worker_sweep',
        '4_scalar_sharded_array',
        '4_scalar_sharded_array',
        '5_production_build',
    ]
    assert rows[0]['speedup'] == '25.0'
    assert rows[0]['subjects'] == '2'
    assert rows[1]['peak_memory_gb'] == '1.0'
    assert rows[2]['passed'] == 'True'
    assert rows[4]['passed'] == 'False'


# --- comparison report: production-scale section ------------------------------


def test_scale_section_renders_every_step_and_is_empty_without_rows(tmp_path: Path) -> None:
    styles = comparison_report._styles()
    assert comparison_report._scale_section([], 'note', styles) == []

    scale = tmp_path / 'scale.csv'
    with scale.open('w', newline='') as stream:
        fields = [
            'step',
            'scalars',
            'subjects',
            'backend',
            'workers',
            'baseline_median_wall_seconds',
            'optimized_median_wall_seconds',
            'speedup',
            'input_files_per_minute',
            'peak_memory_gb',
            'fragment_count',
            'fast_path_rows',
            'passed',
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                'step': '3_full_scalar_original_vs_optimized',
                'scalars': 'area',
                'subjects': '8693',
                'backend': 'hdf5',
                'workers': '1',
                'baseline_median_wall_seconds': '11854.8',
                'optimized_median_wall_seconds': '429.9',
                'speedup': '27.57',
            }
        )
        writer.writerow(
            {
                'step': '3_full_scalar_worker_sweep',
                'scalars': 'area',
                'subjects': '8693',
                'backend': 'tiledb',
                'workers': '8',
                'optimized_median_wall_seconds': '68.6',
                'input_files_per_minute': '7603',
                'peak_memory_gb': '20.4',
                'fragment_count': '1',
                'fast_path_rows': '8692',
            }
        )
        writer.writerow(
            {
                'step': '4_scalar_sharded_array',
                'scalars': 'curv',
                'subjects': '8693',
                'backend': 'hdf5',
                'workers': '8',
                'optimized_median_wall_seconds': '134.5',
                'passed': 'True',
            }
        )
    rows = comparison_report._load_scale(scale)
    assert len(rows) == 3
    flowables = comparison_report._scale_section(rows, 'Original run: 112 h first scalar.', styles)
    # A page break, heading, note, and one heading+table per populated step.
    assert len(flowables) >= 8
    assert comparison_report._load_scale(None) == []


# --- comparison report: text follows the measured runs ----------------------------------


def _group(modality: str, backend: str, baseline: float, optimized: float, *, equal: bool = True):
    return comparison_runs.Group(
        modality=modality,
        backend=backend,
        n_subjects=100,
        n_elements=10,
        n_scalars=1,
        baseline_wall=[baseline],
        optimized_wall=[optimized],
        values_equal=equal,
    )


def _story_text(flowables) -> str:
    """Flatten the report story, including paragraphs nested in table cells."""
    parts = []
    for flowable in flowables:
        if hasattr(flowable, 'getPlainText'):
            parts.append(flowable.getPlainText())
        parts.extend(
            _story_text(cell for cell in row if not isinstance(cell, str))
            for row in getattr(flowable, '_cellvalues', [])
        )
    return ' '.join(parts)


def _report_meta(grouped) -> dict[str, object]:
    return {
        'baseline_revision': '9ab13dd',
        'platform': 'CUBIC compute nodes',
        'versions': dict.fromkeys(('python', 'numpy', 'nibabel', 'h5py', 'tiledb'), '1.0'),
        'settings': {'repetitions': '3', 'warmups': '1', 'workers': '1', 'target_chunk_mb': '2'},
        'cohorts': {
            modality: {
                'cohort': f'/c/{modality}.csv',
                'cohort_sha256': 'ab' * 32,
                'scalar_names': ['x'],
            }
            for modality in grouped
        },
        'cohort_sources': {'cifti': 'RBC subset, internal & real'},
    }


def test_report_text_names_what_was_measured() -> None:
    # The pinned CUBIC medians: TileDB beats HDF5 for NIfTI and MIF, so no HDF5 claim is made.
    cubic = [
        _group('nifti', 'hdf5', 3.02, 2.17),
        _group('nifti', 'tiledb', 3.41, 2.33),
        _group('cifti', 'hdf5', 385.78, 12.12),
        _group('cifti', 'tiledb', 396.47, 12.56),
        _group('mif', 'hdf5', 10.76, 2.91),
        _group('mif', 'tiledb', 14.52, 3.53),
    ]
    assert comparison_report._scope(cubic) == 'NIfTI, CIFTI, and MIF on both HDF5 and TileDB'
    assert comparison_report._largest_gain(cubic) == (
        'The largest gain is CIFTI on HDF5 (31.8x), where the raw-read fast path removes the '
        'per-file XML header parse.'
    )
    mif_only = [_group('mif', 'hdf5', 2.0, 1.0), _group('mif', 'tiledb', 4.0, 1.0)]
    assert comparison_report._scope(mif_only) == 'MIF on both HDF5 and TileDB'
    assert 'one dense fragment' in comparison_report._largest_gain(mif_only)
    assert comparison_report._scope([_group('nifti', 'hdf5', 2.0, 1.0)]) == 'NIfTI on HDF5'

    assert comparison_report._count('3', 'repetition') == 'three repetitions'
    assert comparison_report._count('1', 'warm-up') == 'one warm-up'
    assert comparison_report._count('0', 'warm-up') == 'no warm-ups'
    assert comparison_report._count('1/3', 'repetition') == '1/3 repetitions'
    settings = [{'target_chunk_mb': 32}, {'target_chunk_mb': 2.0}, {}]
    assert comparison_report._run_setting(settings, 'target_chunk_mb') == '2/32'
    with pytest.raises(SystemExit, match='MODALITY=TEXT'):
        comparison_report._cohort_sources(['dwi=somewhere'])


def test_report_verdict_follows_the_results() -> None:
    styles = comparison_report._styles()
    passing = {'cifti': [_group('cifti', 'hdf5', 30.0, 3.0), _group('cifti', 'tiledb', 30.0, 3.0)]}
    text = _story_text(comparison_report._report_story(passing, _report_meta(passing), styles))
    assert 'It is safe to adopt' in text
    assert 'for CIFTI on both HDF5 and TileDB' in text
    assert 'three counterbalanced repetitions after one discarded warm-up' in text
    assert 'RBC subset, internal & real' in text
    assert 'synth' not in text
    assert 'Local' not in text

    failing = {
        'nifti': [
            _group('nifti', 'hdf5', 1.0, 2.0),
            _group('nifti', 'tiledb', 2.0, 1.0, equal=False),
        ]
    }
    text = _story_text(comparison_report._report_story(failing, _report_meta(failing), styles))
    assert 'safe to adopt' not in text
    assert 'in 1 of the 2 modality x backend configurations' in text
    assert 'but only 1 of 2 optimized outputs were bitwise equal' in text
    assert 'faster in 1 of 2 configurations and bitwise equal in 1 of 2' in text
    assert '1 of 2 configurations passed every check' in text
    assert '100 subjects x 1 scalar:' in text
