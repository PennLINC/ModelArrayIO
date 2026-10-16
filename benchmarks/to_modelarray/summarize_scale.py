"""Pin the production-scale CUBIC results (full-scalar and scalar-sharded runs) to one CSV.

Reads the step-3 single-scalar comparison and worker sweep, the step-4 scalar-sharded
job-array walls, and (when present) the step-5 production-build walls, and writes one
row per measurement so the numbers are committed and reproducible alongside the
cross-modality summary. Missing inputs are skipped, so the CSV can be regenerated as
the ladder progresses.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import harness_common as common


def _rows_compare(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / 'comparison_results.csv'
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    meta = json.loads((run_dir / 'comparison_meta.json').read_text(encoding='utf-8'))
    n_rows = sum(1 for _ in open(meta['cohort'], encoding='utf-8')) - 1
    out = []
    for backend in sorted({r['backend'] for r in rows}):
        walls = {
            impl: statistics.median(
                float(r['wall_seconds'])
                for r in rows
                if r['backend'] == backend and r['implementation'] == impl
            )
            for impl in ('baseline', 'optimized')
        }
        out.append(
            {
                'step': '3_full_scalar_original_vs_optimized',
                'scalars': ','.join(meta.get('scalar_names') or []),
                'subjects': n_rows,
                'backend': backend,
                'workers': meta.get('workers'),
                'baseline_median_wall_seconds': round(walls['baseline'], 1),
                'optimized_median_wall_seconds': round(walls['optimized'], 1),
                'speedup': round(walls['baseline'] / walls['optimized'], 2),
            }
        )
    return out


def _rows_sweep(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / 'results.csv'
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return [
            {
                'step': '3_full_scalar_worker_sweep',
                'scalars': r['scalar_names'],
                'subjects': int(r['input_file_count']),
                'backend': r['backend'],
                'workers': int(r['workers']),
                'baseline_median_wall_seconds': '',
                'optimized_median_wall_seconds': round(float(r['wall_seconds']), 1),
                'speedup': '',
                'input_files_per_minute': round(float(r['input_files_per_minute'])),
                'peak_memory_gb': round(float(r['peak_memory_bytes'] or 0) / 2**30, 1),
                'fragment_count': r.get('fragment_count'),
                'fast_path_rows': r.get('fast_path_rows'),
            }
            for r in csv.DictReader(stream)
        ]


def _wall_row(
    step: str, scalar: str, backend: str, seconds: str, passed: bool
) -> dict[str, object]:
    """One job-array conversion; steps 4 and 5 convert every subject with 8 loader workers."""
    return {
        'step': step,
        'scalars': scalar,
        'subjects': 8693,
        'backend': backend,
        'workers': 8,
        'baseline_median_wall_seconds': '',
        'optimized_median_wall_seconds': round(float(seconds), 1),
        'speedup': '',
        'passed': passed,
    }


def _rows_shards(shard_root: Path, step: str) -> list[dict[str, object]]:
    if not shard_root.exists():
        return []
    out = []
    for wall in sorted(shard_root.glob('*/wall.csv')):
        for line in wall.read_text(encoding='utf-8').splitlines():
            _, backend, seconds = line.split(',')
            passed = (wall.parent / 'PASSED').exists()
            out.append(_wall_row(step, wall.parent.name, backend, seconds, passed))
    return out


def _rows_production(prod_root: Path) -> list[dict[str, object]]:
    wall = prod_root / 'checks' / 'wall_seconds.csv'
    if not wall.exists():
        return []
    out = []
    for line in wall.read_text(encoding='utf-8').splitlines():
        scalar, seconds = line.split(',')
        passed = (prod_root / 'checks' / f'{scalar}.PASSED').exists()
        out.append(_wall_row('5_production_build', scalar, 'hdf5', seconds, passed))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--scalar', default='area')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    rows: list[dict[str, object]] = []
    rows += _rows_compare(root / f'cubic-03-{args.scalar}-compare')
    rows += _rows_sweep(root / f'cubic-03-{args.scalar}-sweep')
    rows += _rows_shards(root / 'cubic-04-shard', '4_scalar_sharded_array')
    rows += _rows_production(root / 'production')
    if not rows:
        raise SystemExit('no production-scale results found')
    common.write_csv(args.output, rows)
    print(f'Wrote {len(rows)} rows to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
