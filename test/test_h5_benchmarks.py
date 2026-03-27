"""HDF5 benchmark sweeps with persisted artifacts for later analysis."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from modelarrayio.storage.h5_storage import (
    create_empty_scalar_matrix_dataset,
)
from test.benchmark_data_utils import (
    fill_realistic_voxel_stripe,
    finalize_plan_metadata,
    make_realistic_voxel_benchmark_plan,
)

RESULT_COLUMNS = [
    'timestamp_utc',
    'run_kind',
    'num_input_files',
    'target_chunk_mb',
    'compression',
    'compression_level',
    'shuffle',
    'dtype',
    'num_items',
    'chunk_subjects',
    'chunk_items',
    'elapsed_seconds',
    'data_generation_seconds',
    'hdf5_write_seconds',
    'output_size_bytes',
    'output_size_gb',
    'throughput_values_per_second',
    'throughput_mb_per_second',
    'seed',
    'volume_shape',
    'group_mask_voxels',
    'sampled_voxels',
    'noise_std',
    'dropout_range',
    'mean_missing_fraction',
    'std_missing_fraction',
    'workflow_reference',
    'python_version',
    'h5py_version',
    'platform',
]

_SCHEMA_VALIDATED_PATHS: set[Path] = set()
_BENCHMARK_SEED_BASE = 20260313


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmark_results_dir() -> Path:
    configured = os.environ.get('MODELARRAYIO_BENCHMARK_RESULTS_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    return _project_root() / 'benchmark_results'


def _xdist_worker_id() -> str | None:
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    return worker if worker else None


def _results_csv_path(results_dir: Path) -> Path:
    worker = _xdist_worker_id()
    if worker:
        return results_dir / f'h5_benchmark_results_{worker}.csv'
    return results_dir / 'h5_benchmark_results.csv'


def _run_meta_path(results_dir: Path) -> Path:
    worker = _xdist_worker_id()
    if worker:
        return results_dir / f'run_meta_{worker}.json'
    return results_dir / 'run_meta.json'


def _append_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    _append_csv_rows(csv_path, [row])


def _append_csv_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path = csv_path.resolve()
    if csv_path not in _SCHEMA_VALIDATED_PATHS:
        _ensure_csv_schema(csv_path)
        _SCHEMA_VALIDATED_PATHS.add(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _ensure_csv_schema(csv_path: Path) -> None:
    """Migrate older benchmark CSV headers to the current schema in place."""
    if not csv_path.exists():
        return

    with csv_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    old_header = rows[0]
    if old_header == RESULT_COLUMNS and all(len(r) == len(RESULT_COLUMNS) for r in rows[1:]):
        return

    migrated_rows: list[dict[str, object]] = []
    for values in rows[1:]:
        if not values:
            continue
        old_map = {
            key: values[idx] if idx < len(values) else '' for idx, key in enumerate(old_header)
        }
        migrated_rows.append({column: old_map.get(column, '') for column in RESULT_COLUMNS})

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated_rows)


def _update_run_meta(meta_path: Path, run_kind: str, csv_path: Path) -> None:
    benchmark_env_prefix = 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MODELARRAYIO_S3_ANON=1 PYTHONPATH=src'
    payload = {
        'updated_utc': datetime.now(UTC).isoformat(),
        'last_run_kind': run_kind,
        'csv_path': str(csv_path),
        'plots_dir': str(meta_path.parent / 'plots'),
        'commands': {
            'quick': (
                f'{benchmark_env_prefix} pytest -m benchmark_quick test/test_h5_benchmarks.py -q'
            ),
            'medium': (
                f'{benchmark_env_prefix} pytest -m benchmark_medium test/test_h5_benchmarks.py -q'
            ),
            'full': f'{benchmark_env_prefix} pytest -m benchmark_full test/test_h5_benchmarks.py -q',
            'parallel': (
                f'{benchmark_env_prefix} pytest -n auto -m benchmark_full test/test_h5_benchmarks.py -q'
            ),
            'plot': 'Rscript test/plot_h5_benchmarks.R',
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _benchmark_seed(num_input_files: int) -> int:
    """Use one deterministic seed per cohort size for fair storage-setting comparisons."""
    return _BENCHMARK_SEED_BASE + int(num_input_files)


def _run_single_benchmark(
    tmp_path: Path,
    *,
    run_kind: str,
    num_input_files: int,
    target_chunk_mb: float,
    compression: str,
    compression_level: int,
    shuffle: bool,
    seed: int,
) -> dict[str, object]:
    plan = make_realistic_voxel_benchmark_plan(
        num_input_files=num_input_files,
        seed=seed,
        max_voxels=0,
        require_s3_templates=True,
    )
    num_items = plan.num_items
    output_file = tmp_path / (
        f'{run_kind}_{num_input_files}_{target_chunk_mb}_{compression}_{compression_level}_{int(shuffle)}.h5'
    )

    started = time.perf_counter()
    data_generation_seconds = 0.0
    hdf5_write_seconds = 0.0
    with h5py.File(output_file, 'w') as h5f:
        create_started = time.perf_counter()
        dset = create_empty_scalar_matrix_dataset(
            h5f,
            dataset_path='scalars/alff/values',
            num_subjects=num_input_files,
            num_items=num_items,
            storage_dtype='float32',
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            target_chunk_mb=target_chunk_mb,
            sources_list=[f'sub-{idx:06d}' for idx in range(num_input_files)],
        )
        hdf5_write_seconds += time.perf_counter() - create_started
        stripe_width = dset.chunks[1] if dset.chunks is not None else max(1, num_items // 8)
        stripe_buffer = np.empty((num_input_files, stripe_width), dtype=np.float32)
        missing_counts = np.zeros(num_input_files, dtype=np.int64)

        for start in range(0, num_items, stripe_width):
            end = min(start + stripe_width, num_items)
            view = stripe_buffer[:, : end - start]
            gen_started = time.perf_counter()
            missing_counts += fill_realistic_voxel_stripe(plan, start=start, end=end, out=view)
            data_generation_seconds += time.perf_counter() - gen_started
            write_started = time.perf_counter()
            dset[:, start:end] = view
            hdf5_write_seconds += time.perf_counter() - write_started

        chunk_subjects, chunk_items = dset.chunks
    elapsed = time.perf_counter() - started

    output_size_bytes = output_file.stat().st_size
    output_size_gb = output_size_bytes / (1024.0**3)
    output_file.unlink(missing_ok=True)
    values_written = float(num_input_files * num_items)
    throughput_values_per_second = values_written / elapsed if elapsed > 0 else float('inf')
    throughput_mb_per_second = (
        values_written * 4.0 / (1024.0 * 1024.0) / elapsed if elapsed > 0 else float('inf')
    )
    missing_fractions = missing_counts / float(num_items)
    metadata = finalize_plan_metadata(
        plan,
        mean_missing_fraction=float(np.mean(missing_fractions)),
        std_missing_fraction=float(np.std(missing_fractions)),
    )

    return {
        'timestamp_utc': datetime.now(UTC).isoformat(),
        'run_kind': run_kind,
        'num_input_files': num_input_files,
        'target_chunk_mb': float(target_chunk_mb),
        'compression': compression,
        'compression_level': int(compression_level),
        'shuffle': int(shuffle),
        'dtype': 'float32',
        'num_items': int(num_items),
        'chunk_subjects': int(chunk_subjects),
        'chunk_items': int(chunk_items),
        'elapsed_seconds': float(elapsed),
        'data_generation_seconds': float(data_generation_seconds),
        'hdf5_write_seconds': float(hdf5_write_seconds),
        'output_size_bytes': int(output_size_bytes),
        'output_size_gb': float(output_size_gb),
        'throughput_values_per_second': float(throughput_values_per_second),
        'throughput_mb_per_second': float(throughput_mb_per_second),
        'seed': int(seed),
        'volume_shape': json.dumps(metadata['volume_shape']),
        'group_mask_voxels': int(metadata['group_mask_voxels']),
        'sampled_voxels': int(metadata['sampled_voxels']),
        'noise_std': float(metadata['noise_std']),
        'dropout_range': json.dumps(metadata['dropout_range']),
        'mean_missing_fraction': float(metadata['mean_missing_fraction']),
        'std_missing_fraction': float(metadata['std_missing_fraction']),
        'workflow_reference': str(metadata['workflow_reference']),
        'python_version': sys.version.split()[0],
        'h5py_version': h5py.__version__,
        'platform': platform.platform(),
    }


@pytest.mark.benchmark
@pytest.mark.benchmark_quick
@pytest.mark.parametrize(
    ('num_input_files', 'target_chunk_mb', 'compression', 'compression_level', 'shuffle'),
    [
        (100, 16.0, 'gzip', 1, True),
        (100, 32.0, 'gzip', 1, True),
        (100, 32.0, 'gzip', 4, False),
        (100, 64.0, 'gzip', 9, True),
        (100, 32.0, 'lzf', 0, True),
        (100, 64.0, 'lzf', 0, False),
        (100, 32.0, 'none', 0, False),
        (1000, 16.0, 'gzip', 1, True),
        (1000, 32.0, 'gzip', 4, True),
        (1000, 64.0, 'gzip', 9, False),
        (1000, 32.0, 'lzf', 0, True),
        (1000, 32.0, 'none', 0, False),
    ],
)
def test_h5_benchmark_quick_subset(
    tmp_path,
    num_input_files: int,
    target_chunk_mb: float,
    compression: str,
    compression_level: int,
    shuffle: bool,
) -> None:
    """Fast benchmark subset for interactive inspection."""
    results_dir = _benchmark_results_dir()
    csv_path = _results_csv_path(results_dir)
    meta_path = _run_meta_path(results_dir)
    row = _run_single_benchmark(
        tmp_path=tmp_path,
        run_kind='quick',
        num_input_files=num_input_files,
        target_chunk_mb=target_chunk_mb,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        seed=_benchmark_seed(num_input_files),
    )
    _append_csv_row(csv_path, row)
    _update_run_meta(meta_path, run_kind='quick', csv_path=csv_path)

    assert row['elapsed_seconds'] > 0
    assert row['output_size_bytes'] > 0
    assert row['output_size_gb'] >= 0


@pytest.mark.benchmark
@pytest.mark.benchmark_medium
@pytest.mark.parametrize('num_input_files', [100, 1000, 10000])
@pytest.mark.parametrize('target_chunk_mb', [4.0, 8.0, 16.0, 32.0, 64.0])
@pytest.mark.parametrize('compression', ['gzip', 'lzf', 'none'])
@pytest.mark.parametrize('shuffle', [True, False])
def test_h5_benchmark_medium_sweep(
    tmp_path,
    num_input_files: int,
    target_chunk_mb: float,
    compression: str,
    shuffle: bool,
) -> None:
    """Medium benchmark sweep excluding the largest cohort size."""
    results_dir = _benchmark_results_dir()
    csv_path = _results_csv_path(results_dir)
    meta_path = _run_meta_path(results_dir)
    compression_levels = [1, 4, 9] if compression == 'gzip' else [0]
    rows: list[dict[str, object]] = []
    for compression_level in compression_levels:
        row = _run_single_benchmark(
            tmp_path=tmp_path,
            run_kind='medium',
            num_input_files=num_input_files,
            target_chunk_mb=target_chunk_mb,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            seed=_benchmark_seed(num_input_files),
        )
        rows.append(row)

        assert row['elapsed_seconds'] > 0
        assert row['output_size_bytes'] > 0
        assert row['output_size_gb'] >= 0
    _append_csv_rows(csv_path, rows)
    _update_run_meta(meta_path, run_kind='medium', csv_path=csv_path)


@pytest.mark.benchmark
@pytest.mark.benchmark_full
@pytest.mark.parametrize('num_input_files', [100, 1000, 10000, 40000])
@pytest.mark.parametrize('target_chunk_mb', [4.0, 8.0, 16.0, 32.0, 64.0])
@pytest.mark.parametrize('compression', ['gzip', 'lzf', 'none'])
@pytest.mark.parametrize('shuffle', [True, False])
def test_h5_benchmark_full_sweep(
    tmp_path,
    num_input_files: int,
    target_chunk_mb: float,
    compression: str,
    shuffle: bool,
) -> None:
    """Full benchmark sweep for publication-grade comparisons."""
    results_dir = _benchmark_results_dir()
    csv_path = _results_csv_path(results_dir)
    meta_path = _run_meta_path(results_dir)
    compression_levels = [1, 4, 9] if compression == 'gzip' else [0]
    rows: list[dict[str, object]] = []
    for compression_level in compression_levels:
        row = _run_single_benchmark(
            tmp_path=tmp_path,
            run_kind='full',
            num_input_files=num_input_files,
            target_chunk_mb=target_chunk_mb,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            seed=_benchmark_seed(num_input_files),
        )
        rows.append(row)

        assert row['elapsed_seconds'] > 0
        assert row['output_size_bytes'] > 0
        assert row['output_size_gb'] >= 0
    _append_csv_rows(csv_path, rows)
    _update_run_meta(meta_path, run_kind='full', csv_path=csv_path)
