"""Run reproducible end-to-end ``to-modelarray`` benchmark sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import re
import resource
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import harness_common as common
import source_identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--mask', help='Group mask for NIfTI cohorts (local path or S3 URI).')
    parser.add_argument('--index-file', help='Index MIF for fixel cohorts (local path or S3 URI).')
    parser.add_argument(
        '--directions-file', help='Directions MIF for fixel cohorts (local path or S3 URI).'
    )
    parser.add_argument('--scalar-columns', nargs='+')
    parser.add_argument('--tier', choices=['quick', 'medium', 'full'], default='quick')
    parser.add_argument(
        '--backends', nargs='+', choices=['hdf5', 'tiledb'], default=['hdf5', 'tiledb']
    )
    parser.add_argument('--workers', nargs='+', type=int)
    parser.add_argument('--dtypes', nargs='+', choices=['float32', 'float64'])
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--target-chunk-mb', nargs='+', type=float)
    parser.add_argument(
        '--compression-workers',
        nargs='+',
        type=int,
        help='Internal HDF5 compression-pool sizes to sweep (not a public CLI option).',
    )
    parser.add_argument('--acceptance-files-per-minute', type=float)
    parser.add_argument('--run-id', default=datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--modelarrayio', default='modelarrayio')
    parser.add_argument(
        '--cache-state', choices=['warm', 'cold', 'unspecified'], default='unspecified'
    )
    parser.add_argument('--repeat-factor', type=int, default=1)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--split-files', dest='split_files', action='store_true')
    split_group.add_argument('--no-split-files', dest='split_files', action='store_false')
    parser.set_defaults(split_files=None)
    parser.add_argument('--skip-input-bytes', action='store_true')
    parser.add_argument('--result-shard', help='Write results.<shard>.csv for parallel sweeps.')
    parser.add_argument(
        '--r-gate-command',
        help=(
            'Optional command template run for every output. Available fields: '
            '{output}, {backend}, and {scalars}.'
        ),
    )
    return parser


def _validated_shard(value: str | None) -> str | None:
    """Return a path-safe shard identifier."""
    if value is None:
        return None
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(
            '--result-shard must start with an alphanumeric character and contain only '
            'letters, digits, underscores, dots, or hyphens.'
        )
    return value


def _validated_run_id(value: str) -> str:
    """Return a path-safe benchmark run identifier."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(
            '--run-id must start with an alphanumeric character and contain only '
            'letters, digits, underscores, dots, or hyphens.'
        )
    return value


def _run_artifact_paths(run_dir: Path, shard: str | None) -> tuple[Path, Path, Path]:
    """Return isolated artifact, result, and metadata paths for one shard."""
    artifact_dir = run_dir / f'shard-{shard}' if shard else run_dir
    result_name = f'results.{shard}.csv' if shard else 'results.csv'
    meta_name = f'run_meta.{shard}.json' if shard else 'run_meta.json'
    return artifact_dir, run_dir / result_name, run_dir / meta_name


def _available_cpu_count() -> int:
    """Return CPUs allocated to this process rather than the host total."""
    process_cpu_count = getattr(os, 'process_cpu_count', None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return max(1, int(count))

    sched_getaffinity = getattr(os, 'sched_getaffinity', None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def _machine_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        'platform': platform.platform(),
        'allocated_cpus': _available_cpu_count(),
        'versions': common.package_versions(),
    }
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        page_count = os.sysconf('SC_PHYS_PAGES')
        metadata['physical_memory_bytes'] = int(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        metadata['physical_memory_bytes'] = None
    try:
        import tiledb

        config = tiledb.Config()
        metadata['tiledb_compute_concurrency_level'] = int(config['sm.compute_concurrency_level'])
        metadata['tiledb_io_concurrency_level'] = int(config['sm.io_concurrency_level'])
    except (ImportError, KeyError, TypeError, ValueError):
        metadata['tiledb_compute_concurrency_level'] = None
        metadata['tiledb_io_concurrency_level'] = None
    return metadata


def _source_paths(rows: list[dict[str, str]], scalar_columns: list[str]) -> list[str]:
    paths: list[str] = []
    for row in rows:
        if scalar_columns:
            paths.extend(row[column] for column in scalar_columns if row.get(column))
        elif row.get('source_file'):
            paths.append(row['source_file'])
        if row.get('source_mask_file'):
            paths.append(row['source_mask_file'])
    return paths


def _resolve_split_files(requested: bool | None, scalar_columns: list[str]) -> bool:
    """Mirror the converter's wide-split/long-combined default."""
    return bool(scalar_columns) if requested is None else requested


def _resolved_input(value: str | None) -> str | None:
    if value is None or value.startswith('s3://'):
        return value
    return str(Path(value).resolve())


def _resolved_executable(value: str) -> str:
    """Resolve an explicitly pathed executable before converter cwd changes."""
    candidate = Path(value).expanduser()
    if candidate.exists() or candidate.parent != Path('.'):
        return str(candidate.resolve())
    return value


def _input_bytes(
    paths: list[str], auxiliary_paths: list[str], *, source_root: Path
) -> tuple[int | None, int]:
    sizes: dict[str, int] = {}
    s3_client = None
    total = 0
    all_paths = [*auxiliary_paths, *paths]
    for source in all_paths:
        if source not in sizes:
            if source.startswith('s3://'):
                if s3_client is None:
                    import boto3
                    from botocore import UNSIGNED
                    from botocore.config import Config

                    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                parsed = urlparse(source)
                response = s3_client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
                sizes[source] = int(response['ContentLength'])
            else:
                try:
                    local_source = Path(source)
                    if not local_source.is_absolute():
                        local_source = source_root / local_source
                    sizes[source] = local_source.stat().st_size
                except OSError:
                    return None, len(all_paths)
        total += sizes[source]
    return total, len(all_paths)


def _case_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    available_cpus = _available_cpu_count()
    default_workers = {
        'quick': [1],
        'medium': [1, 4, 8],
        'full': sorted({1, 2, 4, 8, available_cpus}),
    }[args.tier]
    workers = args.workers or default_workers
    dtypes = args.dtypes or (['float32', 'float64'] if args.tier != 'quick' else ['float32'])
    chunk_targets = args.target_chunk_mb or (
        [8.0, 32.0, 64.0, 128.0] if args.tier == 'full' else [2.0]
    )
    cases = []
    for backend in args.backends:
        if args.tier == 'full':
            codecs = (
                [('gzip', 1), ('gzip', 4), ('none', 0)]
                if backend == 'hdf5'
                else [
                    ('zstd', 1),
                    ('zstd', 3),
                    ('zstd', 5),
                    ('none', 0),
                ]
            )
        else:
            codecs = [('gzip', 4)] if backend == 'hdf5' else [('zstd', 5)]
        for worker_count, dtype, chunk_target, codec in itertools.product(
            workers, dtypes, chunk_targets, codecs
        ):
            compression, level = codec
            compression_workers = [None]
            if backend == 'hdf5' and compression != 'none':
                compression_workers = args.compression_workers or (
                    sorted({1, 4, available_cpus}) if args.tier == 'full' else [None]
                )
            cases.extend(
                {
                    'backend': backend,
                    'workers': worker_count,
                    'dtype': dtype,
                    'target_chunk_mb': chunk_target,
                    'compression': compression,
                    'compression_level': level,
                    'compression_workers_requested': compression_worker_count,
                }
                for compression_worker_count in compression_workers
            )
    return cases


def _load_timing(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    scalars = [record for record in records if record.get('event') == 'scalar_complete']
    complete = next(
        (record for record in reversed(records) if record.get('event') == 'conversion_complete'),
        {},
    )
    aggregate: dict[str, object] = dict(complete)
    for field in (
        'source_load_seconds',
        'staging_cpu_seconds',
        'validation_seconds',
        'staging_allocation_seconds',
        'write_seconds',
        'compression_seconds',
        'submission_seconds',
        'fragment_count',
        'fast_path_rows',
    ):
        values = [
            record[field] for record in scalars if isinstance(record.get(field), (int, float))
        ]
        aggregate[field] = sum(values) if values else None
    aggregate['compression_workers'] = max(
        (int(record.get('compression_workers', 0)) for record in scalars), default=0
    )
    return aggregate


def _append_csv(path: Path, row: dict[str, object]) -> None:
    existing: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            existing = list(reader)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + '.tmp')
    common.write_csv(temporary, [*existing, row], fieldnames)
    os.replace(temporary, path)


def _completed_case_keys(path: Path) -> set[str]:
    """Return stable case keys already recorded by an interrupted sweep."""
    if not path.exists():
        return set()
    with path.open(newline='', encoding='utf-8') as stream:
        return {
            row.get('case_key') or row['case_id']
            for row in csv.DictReader(stream)
            if row.get('case_key') or row.get('case_id')
        }


def _recorded_failure_count(path: Path) -> int:
    """Count failed rows so a resumed run preserves its nonzero status."""
    if not path.exists():
        return 0
    with path.open(newline='', encoding='utf-8') as stream:
        return sum(row.get('status') == 'failed' for row in csv.DictReader(stream))


def _write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON metadata file."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, path)


def _run_r_gate(
    template: str | None,
    outputs: list[Path],
    backend: str,
    scalar_columns: list[str],
    log_path: Path,
) -> tuple[str, float | None]:
    if not template:
        return 'not_run', None
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        for output in outputs:
            command = shlex.split(
                template.format(
                    output=str(output),
                    backend=backend,
                    scalars=','.join(scalar_columns),
                )
            )
            try:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except (OSError, ValueError) as exc:
                log.write(f'Could not execute R gate command: {exc}\n')
                return 'failed', time.perf_counter() - started
            if completed.returncode != 0:
                return 'failed', time.perf_counter() - started
    return 'passed', time.perf_counter() - started


def _case_failure_reasons(
    *,
    returncode: int,
    outputs_present: bool,
    r_gate: str,
    meets_acceptance_target: bool | None,
) -> list[str]:
    """Return every correctness or performance gate that failed for a case."""
    reasons = []
    if returncode != 0:
        reasons.append('converter')
    if not outputs_present:
        reasons.append('missing_output')
    if r_gate.startswith('failed'):
        reasons.append('r_gate')
    if meets_acceptance_target is False:
        reasons.append('acceptance_target')
    return reasons


def _case_status(failure_reasons: list[str], r_gate: str) -> str:
    """Distinguish failed, R-verified, and smoke-only benchmark rows."""
    if failure_reasons:
        return 'failed'
    if r_gate == 'not_run':
        return 'unverified'
    return 'passed'


def main() -> int:
    args = _parser().parse_args()
    if args.repetitions < 1 or args.repeat_factor < 1:
        raise ValueError('--repetitions and --repeat-factor must be positive')
    result_shard = _validated_shard(args.result_shard)
    run_id = _validated_run_id(args.run_id)
    repo = Path(__file__).resolve().parents[2]
    root = args.results_dir or Path(
        os.environ.get(
            'MODELARRAYIO_BENCHMARK_RESULTS_DIR', repo / 'benchmarks' / 'to_modelarray' / 'results'
        )
    )
    root = root.expanduser().resolve()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir, results_csv, meta_path = _run_artifact_paths(run_dir, result_shard)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows = common.read_cohort(args.cohort)
    scalar_columns = args.scalar_columns or common.infer_scalar_columns(rows)
    scalar_names = common.scalar_names(rows, scalar_columns)
    split_files = _resolve_split_files(args.split_files, scalar_columns)
    scalar_sources = common.scalar_source_paths(rows, scalar_columns)
    modality = common.source_modality(scalar_sources)
    mask = _resolved_input(args.mask)
    index_file = _resolved_input(args.index_file)
    directions_file = _resolved_input(args.directions_file)
    if modality == 'nifti' and mask is None:
        raise ValueError('--mask is required for a NIfTI benchmark cohort.')
    if modality == 'mif' and (index_file is None or directions_file is None):
        raise ValueError(
            '--index-file and --directions-file are required for a MIF benchmark cohort.'
        )
    auxiliary_paths = (
        [mask]
        if modality == 'nifti'
        else [index_file, directions_file]
        if modality == 'mif'
        else []
    )
    paths = _source_paths(rows, scalar_columns)
    input_bytes, input_file_count = (
        (None, len(paths) + len(auxiliary_paths))
        if args.skip_input_bytes
        else _input_bytes(paths, auxiliary_paths, source_root=args.cohort.resolve().parent)
    )
    machine = _machine_metadata()
    identity = source_identity.source_identity(repo)
    cases = _case_grid(args)
    modelarrayio_command = _resolved_executable(args.modelarrayio)
    configuration = {
        'cases': cases,
        'repetitions': args.repetitions,
        'cache_state': args.cache_state,
        'acceptance_files_per_minute': args.acceptance_files_per_minute,
        'split_files': split_files,
        'skip_input_bytes': args.skip_input_bytes,
        'mask': mask,
        'index_file': index_file,
        'directions_file': directions_file,
        'r_gate_command': args.r_gate_command,
        'modelarrayio': modelarrayio_command,
    }
    fresh_metadata = {
        'run_id': run_id,
        'created_at': datetime.now(UTC).isoformat(),
        'run_kind': 'end_to_end',
        'result_shard': result_shard,
        'tier': args.tier,
        'modality': modality,
        'cohort': str(args.cohort.resolve()),
        'cohort_sha256': common.sha256_file(args.cohort),
        'repeat_factor': args.repeat_factor,
        'acceptance_files_per_minute': args.acceptance_files_per_minute,
        'scalar_columns': scalar_columns,
        'scalar_names': scalar_names,
        'modelarrayio_commit': identity['commit'],
        'source_tree_sha256': identity['source_tree_sha256'],
        'source_dirty': identity['dirty'],
        **machine,
        'configuration': configuration,
        'commands': [],
    }
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        identity_keys = (
            'run_id',
            'run_kind',
            'result_shard',
            'tier',
            'modality',
            'cohort',
            'cohort_sha256',
            'repeat_factor',
            'scalar_columns',
            'scalar_names',
            'modelarrayio_commit',
            'source_tree_sha256',
        )
        mismatches = [key for key in identity_keys if metadata.get(key) != fresh_metadata.get(key)]
        previous_configuration = metadata.get('configuration')
        if previous_configuration is not None and previous_configuration != configuration:
            mismatches.append('configuration')
        if mismatches:
            raise ValueError(
                f'Cannot resume benchmark run {run_id!r}: metadata differs for '
                f'{", ".join(mismatches)}. Choose a new --run-id.'
            )
        metadata['configuration'] = configuration
        metadata.setdefault('commands', [])
        metadata.setdefault('resume_events', []).append(
            {
                'resumed_at': datetime.now(UTC).isoformat(),
                'machine': machine,
            }
        )
    else:
        metadata = fresh_metadata
    _write_json(meta_path, metadata)

    completed_case_keys = _completed_case_keys(results_csv)
    environment = os.environ.copy()
    environment.setdefault('MODELARRAYIO_S3_ANON', '1')
    failures = _recorded_failure_count(results_csv)
    for case_index, case in enumerate(cases, start=1):
        for repetition in range(1, args.repetitions + 1):
            case_prefix = f'{result_shard}-' if result_shard else ''
            case_key = f'{case_prefix}case-{case_index:04d}-rep-{repetition:02d}'
            if case_key in completed_case_keys:
                print(f'{case_key}: already recorded; skipping')
                continue

            case_id = case_key
            case_dir = artifact_dir / 'outputs' / case_id
            retry = 1
            while case_dir.exists():
                case_id = f'{case_key}-retry-{retry:02d}'
                case_dir = artifact_dir / 'outputs' / case_id
                retry += 1
            case_dir.mkdir(parents=True, exist_ok=True)
            output = common.output_path(case_dir, case['backend'])
            timing_path = case_dir / 'timing.jsonl'
            command = [
                modelarrayio_command,
                'to-modelarray',
                '--cohort-file',
                str(args.cohort.resolve()),
                '--output',
                str(output),
                '--backend',
                str(case['backend']),
                '--workers',
                str(case['workers']),
                '--dtype',
                str(case['dtype']),
                '--compression',
                str(case['compression']),
                '--compression-level',
                str(case['compression_level']),
                '--target-chunk-mb',
                str(case['target_chunk_mb']),
                '--timing-jsonl',
                str(timing_path),
                '--split-files' if split_files else '--no-split-files',
            ]
            if modality == 'nifti':
                command.extend(['--mask', str(mask)])
            elif modality == 'mif':
                command.extend(
                    [
                        '--index-file',
                        str(index_file),
                        '--directions-file',
                        str(directions_file),
                    ]
                )
            if scalar_columns:
                command.extend(['--scalar-columns', *scalar_columns])

            stdout_path = case_dir / 'stdout.log'
            stderr_path = case_dir / 'stderr.log'
            started = time.perf_counter()
            before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            with (
                stdout_path.open('w', encoding='utf-8') as stdout,
                stderr_path.open('w', encoding='utf-8') as stderr,
            ):
                case_environment = environment.copy()
                if case['compression_workers_requested'] is None:
                    case_environment.pop('MODELARRAYIO_HDF5_COMPRESSION_WORKERS', None)
                else:
                    case_environment['MODELARRAYIO_HDF5_COMPRESSION_WORKERS'] = str(
                        case['compression_workers_requested']
                    )
                completed = subprocess.run(
                    command,
                    cwd=args.cohort.resolve().parent,
                    env=case_environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            wall_seconds = time.perf_counter() - started
            after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)

            outputs = [output] if output.exists() else sorted(case_dir.glob(f'*_{output.name}'))
            output_bytes = 0
            output_files = 0
            for candidate in outputs:
                candidate_bytes, candidate_files = common.output_stats(candidate)
                output_bytes += candidate_bytes
                output_files += candidate_files
            if completed.returncode != 0:
                r_gate, r_read_seconds = 'not_run_converter_failed', None
            elif not outputs:
                r_gate, r_read_seconds = 'failed_missing_output', None
            else:
                r_gate, r_read_seconds = _run_r_gate(
                    args.r_gate_command,
                    outputs,
                    str(case['backend']),
                    scalar_names,
                    case_dir / 'r_gate.log',
                )
            timing = _load_timing(timing_path)
            files_per_minute = 60.0 * input_file_count / wall_seconds
            meets_acceptance_target = (
                None
                if args.acceptance_files_per_minute is None
                else files_per_minute >= args.acceptance_files_per_minute
            )
            failure_reasons = _case_failure_reasons(
                returncode=completed.returncode,
                outputs_present=bool(outputs),
                r_gate=r_gate,
                meets_acceptance_target=meets_acceptance_target,
            )
            case_status = _case_status(failure_reasons, r_gate)
            case_failed = case_status == 'failed'
            row = {
                'run_id': run_id,
                'case_key': case_key,
                'case_id': case_id,
                'run_kind': 'end_to_end',
                'tier': args.tier,
                'status': case_status,
                'failure_reason': ','.join(failure_reasons),
                'returncode': completed.returncode,
                'backend': case['backend'],
                'workers': case['workers'],
                'dtype': case['dtype'],
                'compression': case['compression'],
                'compression_level': case['compression_level'],
                'compression_workers': timing.get('compression_workers'),
                'compression_workers_requested': case['compression_workers_requested'],
                'target_chunk_mb': case['target_chunk_mb'],
                'cache_state': args.cache_state,
                'repeat_factor': args.repeat_factor,
                'repetition': repetition,
                'scalar_names': ','.join(scalar_names),
                'cohort_sha256': metadata['cohort_sha256'],
                'input_file_count': input_file_count,
                'input_bytes': input_bytes,
                'input_files_per_minute': files_per_minute,
                'acceptance_files_per_minute': args.acceptance_files_per_minute,
                'meets_acceptance_target': meets_acceptance_target,
                'wall_seconds': wall_seconds,
                'child_user_cpu_seconds': after_usage.ru_utime - before_usage.ru_utime,
                'child_system_cpu_seconds': after_usage.ru_stime - before_usage.ru_stime,
                'children_peak_rss_platform_units_cumulative': after_usage.ru_maxrss,
                'peak_memory_bytes': timing.get('peak_memory_bytes'),
                'output_bytes': output_bytes,
                'output_file_count': output_files,
                'r_gate': r_gate,
                'r_read_seconds': r_read_seconds,
                'modelarrayio_commit': metadata['modelarrayio_commit'],
                'source_filesystem': (
                    's3'
                    if paths and all(path.startswith('s3://') for path in paths)
                    else 'mixed'
                    if any(path.startswith('s3://') for path in paths)
                    else 'local'
                ),
                'output_filesystem': 'local',
                'tiledb_compute_concurrency_level': machine.get(
                    'tiledb_compute_concurrency_level'
                ),
                'tiledb_io_concurrency_level': machine.get('tiledb_io_concurrency_level'),
                **{
                    key: timing.get(key)
                    for key in (
                        'source_load_seconds',
                        'staging_cpu_seconds',
                        'validation_seconds',
                        'staging_allocation_seconds',
                        'write_seconds',
                        'compression_seconds',
                        'submission_seconds',
                        'fragment_count',
                        'fast_path_rows',
                        'peak_open_file_descriptors_sampled',
                    )
                },
                'source_tree_sha256': metadata.get('source_tree_sha256'),
                'command': shlex.join(command),
            }
            _append_csv(results_csv, row)
            metadata['commands'].append(
                {
                    'case_key': case_key,
                    'case_id': case_id,
                    'command': command,
                    'environment_overrides': {
                        'MODELARRAYIO_S3_ANON': case_environment.get('MODELARRAYIO_S3_ANON'),
                        'MODELARRAYIO_HDF5_COMPRESSION_WORKERS': case_environment.get(
                            'MODELARRAYIO_HDF5_COMPRESSION_WORKERS'
                        ),
                    },
                    'returncode': completed.returncode,
                    'status': row['status'],
                    'r_gate': r_gate,
                    'stdout': str(stdout_path.relative_to(run_dir)),
                    'stderr': str(stderr_path.relative_to(run_dir)),
                }
            )
            _write_json(meta_path, metadata)
            completed_case_keys.add(case_key)
            failures += int(case_failed)
            print(
                f'{case_id}: backend={case["backend"]} workers={case["workers"]} '
                f'dtype={case["dtype"]} status={row["status"]} wall={wall_seconds:.3f}s'
            )
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
