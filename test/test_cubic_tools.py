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
