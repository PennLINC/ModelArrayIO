"""Compare original and optimized ``to-modelarray`` executables on identical inputs."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import platform
import resource
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import harness_common as common
import output_compare
import source_identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-executable', type=Path, required=True)
    parser.add_argument('--optimized-executable', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--mask', type=Path, help='Group mask for NIfTI cohorts.')
    parser.add_argument('--index-file', type=Path, help='Index MIF for fixel cohorts.')
    parser.add_argument('--directions-file', type=Path, help='Directions MIF for fixel cohorts.')
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument(
        '--scalar-columns',
        nargs='+',
        help='Wide-format NIfTI scalar columns; omit for long-format CIFTI/MIF cohorts.',
    )
    parser.add_argument(
        '--backends', nargs='+', choices=['hdf5', 'tiledb'], default=['hdf5', 'tiledb']
    )
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument(
        '--target-chunk-mb',
        type=float,
        default=2.0,
        help='Chunk/tile target passed to both executables (production CIFTI used 32).',
    )
    parser.add_argument('--r-gate-script', type=Path, required=True)
    parser.add_argument(
        '--gate-command',
        help=(
            'Optional gate command template with {output} and {backend} fields, used instead '
            'of "Rscript <r-gate-script>" (for example a dispatcher that gates TileDB in '
            'Python where the R tiledb package is unavailable).'
        ),
    )
    parser.add_argument('--baseline-revision', default='HEAD')
    parser.add_argument('--optimized-revision', default='working-tree')
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Reuse completed case directories and only run correctness checks.',
    )
    return parser


def _source_identity(repo: Path) -> dict[str, object]:
    """Pin the exact (possibly uncommitted or rsynced) source tree that ran."""
    identity = source_identity.source_identity(repo)
    return {
        'source_commit': identity['commit'],
        'source_dirty': identity['dirty'],
        'source_tree_sha256': identity['source_tree_sha256'],
    }


def _case_order(repetitions: int) -> list[tuple[str, int]]:
    """Counterbalance adjacent runs while retaining stable repetition identifiers."""
    result = []
    for repetition in range(1, repetitions + 1):
        implementations = (
            ('baseline', 'optimized') if repetition % 2 else ('optimized', 'baseline')
        )
        result.extend((implementation, repetition) for implementation in implementations)
    return result


def _command(
    executable: Path,
    implementation: str,
    backend: str,
    cohort: Path,
    output: Path,
    inputs: dict[str, object],
    workers: int,
) -> list[str]:
    compression, level = ('gzip', 4) if backend == 'hdf5' else ('zstd', 5)
    modality = inputs['modality']
    split = modality == 'nifti'
    command = [
        str(executable),
        'to-modelarray',
        '--cohort-file',
        str(cohort),
        '--output',
        str(output),
        '--backend',
        backend,
        '--workers',
        str(workers),
        '--dtype',
        'float32',
        '--compression',
        compression,
        '--compression-level',
        str(level),
        '--target-chunk-mb',
        str(inputs.get('target_chunk_mb', 2.0)),
        '--split-files' if split else '--no-split-files',
        '--log-level',
        'WARNING',
    ]
    if modality == 'nifti':
        command.extend(['--mask', str(inputs['mask'])])
        command.extend(['--scalar-columns', *inputs['scalar_columns']])
    elif modality == 'mif':
        command.extend(['--index-file', str(inputs['index_file'])])
        command.extend(['--directions-file', str(inputs['directions_file'])])
    if implementation == 'baseline':
        command.extend(['--s3-workers', str(workers)])
    return command


def _expected_outputs(case_dir: Path, backend: str, inputs: dict[str, object]) -> list[Path]:
    if inputs['modality'] == 'nifti':
        return [
            common.output_path(case_dir, backend, scalar) for scalar in inputs['scalar_columns']
        ]
    return [common.output_path(case_dir, backend)]


def _run_gate(
    rscript: str | None,
    script: Path,
    output: Path,
    backend: str,
    log,
    template: str | None = None,
) -> tuple[bool, float]:
    if template:
        command = shlex.split(template.format(output=str(output), backend=backend))
    else:
        command = [str(rscript), str(script), str(output), backend]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    except (OSError, ValueError) as exc:
        log.write(f'Could not execute gate command {command!r}: {exc}\n')
        return False, time.perf_counter() - started
    return completed.returncode == 0, time.perf_counter() - started


def _run_case(
    *,
    implementation: str,
    executable: Path,
    backend: str,
    repetition: int,
    case_dir: Path,
    cohort: Path,
    inputs: dict[str, object],
    workers: int,
    rscript: str | None,
    r_gate_script: Path,
    gate_template: str | None = None,
) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=False)
    requested_output = common.output_path(case_dir, backend)
    command = _command(
        executable,
        implementation,
        backend,
        cohort,
        requested_output,
        inputs,
        workers,
    )
    stdout_path = case_dir / 'stdout.log'
    stderr_path = case_dir / 'stderr.log'
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault('MODELARRAYIO_S3_ANON', '1')
    with (
        stdout_path.open('w', encoding='utf-8') as stdout,
        stderr_path.open('w', encoding='utf-8') as stderr,
    ):
        # Resolve relative cohort paths the same way run_benchmarks.py and the
        # converter's own convention do: from the cohort file's directory.
        completed = subprocess.run(
            command,
            cwd=cohort.parent,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    outputs = _expected_outputs(case_dir, backend, inputs)
    outputs_present = all(path.exists() for path in outputs)
    gate_passed = False
    r_read_seconds = 0.0
    gate_path = case_dir / 'r_gate.log'
    with gate_path.open('w', encoding='utf-8') as gate_log:
        if completed.returncode == 0 and outputs_present:
            gate_results = [
                _run_gate(rscript, r_gate_script, output, backend, gate_log, gate_template)
                for output in outputs
            ]
            gate_passed = all(passed for passed, _ in gate_results)
            r_read_seconds = sum(seconds for _, seconds in gate_results)
    output_bytes = 0
    output_files = 0
    if outputs_present:
        for output in outputs:
            size, count = common.output_stats(output)
            output_bytes += size
            output_files += count
    status = (
        'passed' if completed.returncode == 0 and outputs_present and gate_passed else 'failed'
    )
    return {
        'implementation': implementation,
        'backend': backend,
        'repetition': repetition,
        'status': status,
        'returncode': completed.returncode,
        'wall_seconds': wall_seconds,
        'child_user_cpu_seconds': after.ru_utime - before.ru_utime,
        'child_system_cpu_seconds': after.ru_stime - before.ru_stime,
        'r_gate': 'passed' if gate_passed else 'failed',
        'r_read_seconds': r_read_seconds,
        'output_bytes': output_bytes,
        'output_file_count': output_files,
        'output_paths': json.dumps([str(path) for path in outputs]),
        'stdout_log': str(stdout_path),
        'stderr_log': str(stderr_path),
        'r_gate_log': str(gate_path),
        'command': shlex.join(command),
    }


def _compare_outputs(
    reference: Path, candidate: Path, backend: str, inputs: dict[str, object]
) -> dict[str, object]:
    comparisons = []
    split = inputs['modality'] == 'nifti'
    for scalar in inputs['scalar_names']:
        output_scalar = scalar if split else None
        comparisons.append(
            output_compare.compare_outputs(
                common.output_path(reference, backend, output_scalar),
                common.output_path(candidate, backend, output_scalar),
                scalar,
            )
        )
    result = {
        key: all(bool(comparison[key]) for comparison in comparisons)
        for key in ('shape_equal', 'dtype_equal', 'sources_equal', 'nan_equal', 'values_equal')
    }
    result['max_abs_difference'] = max(
        float(comparison['max_abs_difference']) for comparison in comparisons
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.workers < 1:
        raise ValueError('repetitions/workers must be positive and warmups must be non-negative')
    baseline = args.baseline_executable.expanduser().resolve()
    optimized = args.optimized_executable.expanduser().resolve()
    cohort = args.cohort.expanduser().resolve()
    gate = args.r_gate_script.expanduser().resolve()
    rscript = shutil.which('Rscript')
    if rscript is None and not args.gate_command:
        raise FileNotFoundError('Rscript was not found on PATH (or pass --gate-command)')

    cohort_rows = common.read_cohort(cohort)
    scalar_columns = args.scalar_columns or common.infer_scalar_columns(cohort_rows)
    modality = common.source_modality(common.scalar_source_paths(cohort_rows, scalar_columns))
    scalar_names = common.scalar_names(cohort_rows, scalar_columns)
    mask = args.mask.expanduser().resolve() if args.mask is not None else None
    index_file = args.index_file.expanduser().resolve() if args.index_file is not None else None
    directions_file = (
        args.directions_file.expanduser().resolve() if args.directions_file is not None else None
    )
    if modality == 'nifti' and mask is None:
        raise ValueError('--mask is required for a NIfTI cohort.')
    if modality == 'mif' and (index_file is None or directions_file is None):
        raise ValueError('--index-file and --directions-file are required for a MIF cohort.')
    inputs: dict[str, object] = {
        'modality': modality,
        'mask': mask,
        'index_file': index_file,
        'directions_file': directions_file,
        'scalar_columns': scalar_columns,
        'scalar_names': scalar_names,
        'target_chunk_mb': args.target_chunk_mb,
    }
    required = [baseline, optimized, cohort, gate]
    required += [path for path in (mask, index_file, directions_file) if path is not None]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    run_dir = args.results_dir.expanduser().resolve()
    executables = {'baseline': baseline, 'optimized': optimized}
    if args.verify_only:
        results_path = run_dir / 'comparison_results.csv'
        if not results_path.is_file():
            raise FileNotFoundError(results_path)
        with results_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        case_dirs = {
            (implementation, backend, repetition): run_dir
            / f'{implementation}-{backend}-rep-{repetition:02d}'
            for backend in args.backends
            for implementation in ('baseline', 'optimized')
            for repetition in range(1, args.repetitions + 1)
        }
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        # Warm-up and measured runs share every setting except the case itself.
        run_case = functools.partial(
            _run_case,
            cohort=cohort,
            inputs=inputs,
            workers=args.workers,
            rscript=rscript,
            r_gate_script=gate,
            gate_template=args.gate_command,
        )
        for backend in args.backends:
            for implementation in ('baseline', 'optimized'):
                for warmup in range(1, args.warmups + 1):
                    with tempfile.TemporaryDirectory(
                        prefix=f'warmup-{implementation}-{backend}-{warmup:02d}-', dir=run_dir
                    ) as temporary:
                        temporary_path = Path(temporary)
                        row = run_case(
                            implementation=implementation,
                            executable=executables[implementation],
                            backend=backend,
                            repetition=0,
                            case_dir=temporary_path / 'case',
                        )
                        if row['status'] != 'passed':
                            raise RuntimeError(
                                f'Warm-up failed for {implementation}/{backend}: '
                                f'{row["stderr_log"]}'
                            )

        rows = []
        case_dirs: dict[tuple[str, str, int], Path] = {}
        for backend in args.backends:
            for implementation, repetition in _case_order(args.repetitions):
                case_dir = run_dir / f'{implementation}-{backend}-rep-{repetition:02d}'
                row = run_case(
                    implementation=implementation,
                    executable=executables[implementation],
                    backend=backend,
                    repetition=repetition,
                    case_dir=case_dir,
                )
                rows.append(row)
                case_dirs[(implementation, backend, repetition)] = case_dir
                common.write_csv(run_dir / 'comparison_results.csv', rows)
                print(
                    f'{implementation}/{backend}/rep-{repetition:02d}: '
                    f'status={row["status"]} wall={row["wall_seconds"]:.3f}s',
                    flush=True,
                )
                if row['status'] != 'passed':
                    return 1

    correctness_rows = []
    for backend in args.backends:
        reference = case_dirs[('baseline', backend, 1)]
        for repetition in range(1, args.repetitions + 1):
            candidate = case_dirs[('optimized', backend, repetition)]
            correctness_rows.append(
                {
                    'backend': backend,
                    'optimized_repetition': repetition,
                    **_compare_outputs(reference, candidate, backend, inputs),
                }
            )
    common.write_csv(run_dir / 'correctness_results.csv', correctness_rows)

    metadata = {
        'created_at': datetime.now(UTC).isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'baseline_executable': str(baseline),
        'optimized_executable': str(optimized),
        'baseline_revision': args.baseline_revision,
        'optimized_revision': args.optimized_revision,
        'cohort': str(cohort),
        'cohort_sha256': common.sha256_file(cohort),
        'modality': modality,
        'mask': str(mask) if mask is not None else None,
        'mask_sha256': common.sha256_file(mask) if mask is not None else None,
        'index_file': str(index_file) if index_file is not None else None,
        'directions_file': str(directions_file) if directions_file is not None else None,
        'scalar_columns': scalar_columns,
        'scalar_names': scalar_names,
        'backends': args.backends,
        'workers': args.workers,
        'target_chunk_mb': args.target_chunk_mb,
        'gate_command': args.gate_command,
        'repetitions': args.repetitions,
        'warmups': args.warmups,
        'versions': common.package_versions(),
        **_source_identity(Path(__file__).resolve().parents[2]),
        'case_order': _case_order(args.repetitions),
        'comparison_results': 'comparison_results.csv',
        'correctness_results': 'correctness_results.csv',
    }
    (run_dir / 'comparison_meta.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8'
    )
    correctness_passed = all(
        all(
            bool(row[key])
            for key in ('shape_equal', 'dtype_equal', 'sources_equal', 'nan_equal', 'values_equal')
        )
        for row in correctness_rows
    )
    return int(not correctness_passed)


if __name__ == '__main__':
    raise SystemExit(main())
